import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    ExecResult,
    Observation,
    Perturbation,
    SkillCall,
    SkillTarget,
)
from isaacs_mom.orchestrator import cli, loop
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import evaluate, run_loop
from isaacs_mom.seeds import DEV_SEED_START, SealedSeedError, assert_not_heldout, heldout_seeds
from isaacs_mom.sim.fake import BIN_OUTER_HALF, FLOOR_Z, HELD_Z, PLACE_X, FakeSim, _bin_pose
from isaacs_mom.student import StubStudent, stub_finetune
from isaacs_mom.teacher import choose_scenes
from isaacs_mom.teacher.argon import Argon, Part, Purpose, stub_generate
from isaacs_mom.validator import replay, validate
from isaacs_mom.validator.rules import recording_error_verdict
from isaacs_mom.validator.training import build_training_set, training_examples


class ScriptedPolicy:
    """Test-only: lines up, hugs, lifts, turns to the bin and places, using ground truth real policies never see."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        gt = self.sim.ground_truth()
        work_y, closed, lifted = obs.joint_state
        x, y, heading = self.sim.pose()
        if gt.box_on_bin_floor:
            return None
        if not closed:
            dy = -math.sin(heading) * (gt.box_xyz[0] - x) + math.cos(heading) * (gt.box_xyz[1] - y) - work_y
            if abs(dy) > 0.01:
                return SkillCall(skill="reach_left" if dy > 0 else "reach_right", params={"distance_cm": abs(dy) * 100})
            return SkillCall(skill="hug")
        if not lifted:
            return SkillCall(skill="lift")
        if abs(gt.box_xyz[0] - 0.4) < 0.2 and abs(gt.box_xyz[1]) < 0.25:  # still over the table
            return SkillCall(skill="walk_back", params={"distance_cm": 30})
        dist = math.hypot(gt.bin_xy[0] - x, gt.bin_xy[1] - y)
        turn = math.degrees(math.remainder(math.atan2(gt.bin_xy[1] - y, gt.bin_xy[0] - x) - heading, math.tau))
        if abs(turn) > 2:
            return SkillCall(skill="turn_left" if turn > 0 else "turn_right", params={"angle_deg": abs(turn)})
        if dist > PLACE_X + 0.05:
            return SkillCall(skill="walk_forward", params={"distance_cm": (dist - PLACE_X - 0.02) * 100})
        return SkillCall(skill="place")


class GiveUp:
    controller: Controller = "student"
    checkpoint: str | None = "ckpt-r0"

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return SkillCall(skill="reobserve") if obs.step == 0 else None


def test_same_seed_same_scene_and_perturbation_changes_the_frame() -> None:
    sim = FakeSim()
    sim.reset(1234)
    first = sim.observe().head_rgb_jpeg
    sim.reset(1234)
    assert sim.observe().head_rgb_jpeg == first
    sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.observe().head_rgb_jpeg != first


def test_observation_carries_no_object_poses() -> None:
    assert not {"box_xyz", "bin_xy", "box", "bin"} & set(Observation.model_fields)


def test_skill_calls_are_whitelisted_and_clamped() -> None:
    with pytest.raises(ValidationError):
        SkillCall.model_validate({"skill": "fly"})
    with pytest.raises(ValidationError):
        SkillCall(skill="reach_left", params={"torque": 5.0})
    assert SkillCall(skill="walk_forward", params={"distance_cm": 500}).params["distance_cm"] == 150
    assert SkillCall(skill="turn_left", params={"angle_deg": 500}).params["angle_deg"] == 180


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_skill_params_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        SkillCall(skill="turn_left", params={"angle_deg": value})


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_params_in_model_output_are_rejected(text: str) -> None:
    # json.loads accepts these, so a model's raw answer can carry them.
    raw = json.loads(f'{{"skill": "reach_left", "params": {{"distance_cm": {text}}}}}')
    with pytest.raises(ValidationError):
        SkillCall.model_validate(raw)


def test_fake_place_without_walking_drops_the_box_outside_the_bin() -> None:
    sim = FakeSim()
    sim.reset(1234)
    box_y = sim.ground_truth().box_xyz[1]
    sim.execute(SkillCall(skill="reach_left" if box_y > 0 else "reach_right", params={"distance_cm": abs(box_y) * 100}))
    for skill in ("hug", "lift", "place"):
        assert sim.execute(SkillCall(skill=skill)).ok
    gt = sim.ground_truth()
    assert not gt.box_on_bin_floor
    assert not validate(gt).accepted


def test_fake_refuses_a_reach_past_the_limit_without_moving() -> None:
    sim = FakeSim()
    sim.reset(1234)
    before = sim.observe().joint_state
    result = sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 20}))
    assert not result.ok and result.error and not result.executed_actions
    assert sim.observe().joint_state == before


def test_fake_refuses_a_walk_into_the_table_without_moving() -> None:
    sim = FakeSim()
    sim.reset(1234)
    before = (sim.observe().joint_state, sim.pose())
    result = sim.execute(SkillCall(skill="walk_forward", params={"distance_cm": 30}))
    assert not result.ok and result.error and "table" in result.error and not result.executed_actions
    assert (sim.observe().joint_state, sim.pose()) == before


def test_fake_walks_and_turns_the_requested_amounts() -> None:
    sim = FakeSim()
    sim.reset(1234)
    sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 90}))
    sim.execute(SkillCall(skill="walk_forward", params={"distance_cm": 50}))
    assert sim.pose() == pytest.approx((-0.30, 0.50, math.pi / 2))


def _to_the_bin(sim: FakeSim) -> None:
    policy = ScriptedPolicy(sim)
    while call := policy.act(sim.observe(), "", ""):
        assert sim.execute(call).ok


def test_fake_moves_the_box_only_while_it_rests_on_the_table_out_of_the_hands() -> None:
    sim = FakeSim()
    sim.reset(1234)
    start = sim.ground_truth()
    assert sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth().box_xyz != start.box_xyz

    sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    sim.execute(SkillCall(skill="hug"))  # 30 cm short of the box, so the hands close on nothing, as in MujocoSim
    missed = sim.ground_truth()
    assert sim.closed and not sim.held
    assert sim.perturb(Perturbation(kind="move_box", seed=100))
    assert sim.ground_truth().box_xyz != missed.box_xyz

    sim.reset(1234)
    box_y = sim.ground_truth().box_xyz[1]
    sim.execute(SkillCall(skill="reach_left" if box_y > 0 else "reach_right", params={"distance_cm": abs(box_y) * 100}))
    sim.execute(SkillCall(skill="hug"))
    hugged = sim.ground_truth()
    assert sim.held
    assert not sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth() == hugged

    sim.reset(1234)
    _to_the_bin(sim)
    placed = sim.ground_truth()
    assert placed.box_on_bin_floor
    assert not sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth() == placed


def test_fake_moves_the_bin_under_a_box_held_over_it_mid_carry() -> None:
    sim = FakeSim()
    sim.reset(1234)
    policy = ScriptedPolicy(sim)
    while (call := policy.act(sim.observe(), "", "")) and call.skill != "place":
        assert sim.execute(call).ok
    carried = sim.ground_truth()
    assert sim.held and sim.lifted
    assert sim.perturb(Perturbation(kind="move_bin", seed=99))  # held up in the hands, it is above any bin
    moved = sim.ground_truth()
    assert moved.bin_xy != carried.bin_xy and moved.box_xyz == carried.box_xyz

    # Even right over where the new bin lands, as MujocoSim allows: a new bin is 1-1.4 m from the robot, so this is
    # set up rather than carried there.
    sim.reset(1234)
    landing = np.random.default_rng(99)
    sim.box, sim.box_z = _bin_pose(landing, sim.xy)[0], HELD_Z
    assert sim.perturb(Perturbation(kind="move_bin", seed=99))


def test_fake_moves_the_bin_only_clear_of_the_box() -> None:
    sim = FakeSim()
    sim.reset(1234)
    _to_the_bin(sim)
    placed = sim.ground_truth()
    assert not sim.perturb(Perturbation(kind="move_bin", seed=99))
    assert sim.ground_truth() == placed and validate(placed).accepted

    sim.reset(1234)
    assert sim.perturb(Perturbation(kind="move_bin", seed=99))  # the box is on the table, above the bin
    landing = sim.ground_truth()
    outward = np.array([math.cos(landing.bin_yaw), math.sin(landing.bin_yaw)])
    # On the floor beside where the bin would go: centred 10 cm past its wall the box still reaches under it.
    for past_wall, clear in [(0.10, False), (0.30, True)]:
        sim.reset(1234)
        sim.box, sim.box_z = np.array(landing.bin_xy) + (BIN_OUTER_HALF + past_wall) * outward, FLOOR_Z
        on_floor = sim.ground_truth()
        assert sim.perturb(Perturbation(kind="move_bin", seed=99)) is clear
        assert (sim.ground_truth().bin_xy == landing.bin_xy) is clear
        assert sim.ground_truth().box_xyz == on_floor.box_xyz


def test_a_perturbation_the_sim_refuses_is_reported_not_recorded(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug"}
    steps = len(run_episode(sim, ScriptedPolicy(sim), seed=1505, **common).steps)
    late = run_episode(sim, ScriptedPolicy(sim), seed=1505, perturb_at=steps, **common)  # the box is in the bin
    early = run_episode(sim, ScriptedPolicy(sim), seed=1505, perturb_at=0, **common)  # the box is on the table
    logged = [json.loads(line) for line in events.path.read_text().splitlines()]

    def of(manifest: EpisodeManifest, type_: str) -> list[int]:
        return [
            e["data"]["step"] for e in logged if e["type"] == type_ and e["data"]["episode_id"] == manifest.episode_id
        ]

    assert late.validation and late.validation.accepted
    assert late.perturbations == [] and of(late, "perturbation") == [] and of(late, "error") == [steps]
    assert [p.before_step for p in early.perturbations] == [0] and of(early, "perturbation") == [0]
    assert not list((tmp_path / "episodes" / late.episode_id).glob("**/before_move*"))  # a refused move has no frame


def test_an_applied_move_names_the_frame_seen_right_before_it_which_is_never_a_step(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug", "seed": 1501}
    still = run_episode(sim, ScriptedPolicy(sim), **common)  # step 0 reaches, so at step 1 the box rests on the table
    moved = run_episode(sim, ScriptedPolicy(sim), perturb_at=1, **common)
    (move,) = [json.loads(line)["data"] for line in events.path.read_text().splitlines() if '"perturbation"' in line]
    before = (tmp_path / move["before_frame"]).read_bytes()
    episode_dir = tmp_path / "episodes" / moved.episode_id

    # What the robot saw at that pose without the move, and not what it saw after it.
    assert before == (tmp_path / "episodes" / still.episode_id / still.steps[1].frame).read_bytes()
    assert before != (episode_dir / moved.steps[1].frame).read_bytes()
    # Never a step and never trained on (rule 3): the steps and their frames are only the executed decisions.
    assert moved.validation and moved.validation.accepted
    assert sorted(p.name for p in (episode_dir / "frames").iterdir()) == [Path(s.frame).name for s in moved.steps]
    images = {Path(e.image) for e in training_examples(episode_dir, moved)}
    assert images == {episode_dir / s.frame for s in moved.steps} and tmp_path / move["before_frame"] not in images
    assert replay(FakeSim(), episode_dir / "manifest.json")


def test_a_move_due_when_the_student_stops_is_made_once_not_again_when_the_teacher_takes_over(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug"}
    manifest = run_episode(sim, GiveUp(), takeover=ScriptedPolicy(sim), seed=1505, perturb_at=1, **common)
    moves = [line for line in events.path.read_text().splitlines() if '"perturbation"' in line]
    assert manifest.takeover_step == 1  # the student stopped at step 1, where the move was due
    assert [p.before_step for p in manifest.perturbations] == [1] and len(moves) == 1


class Calls:
    """Plays a fixed list of skills, then stops."""

    controller: Controller = "student"
    checkpoint: str | None = None

    def __init__(self, *skills: str) -> None:
        self.skills = list(skills)

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return SkillCall.model_validate({"skill": self.skills.pop(0)}) if self.skills else None


def _move_bin_events(events: EventLog, episode_id: str) -> list[dict[str, Any]]:
    logged = [json.loads(line) for line in events.path.read_text().splitlines()]
    return [e for e in logged if e["data"].get("episode_id") == episode_id and e["type"] in ("perturbation", "error")]


def test_the_bin_moves_once_right_after_a_lift_that_worked_and_the_episode_replays(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug", "seed": 1505}
    still = run_episode(sim, ScriptedPolicy(sim), **common)
    bin_unmoved = sim.ground_truth().bin_xy
    moved = run_episode(sim, ScriptedPolicy(sim), move_bin_after_lift=True, **common)
    lift = next(i for i, s in enumerate(moved.steps) if s.skill_call.skill == "lift")
    assert [(p.before_step, p.perturbation.kind) for p in moved.perturbations] == [(lift + 1, "move_bin")]
    (move,) = _move_bin_events(events, moved.episode_id)
    assert move["type"] == "perturbation" and move["data"]["kind"] == "move_bin"
    assert moved.validation and moved.validation.accepted  # it re-aims every step, so it finds the bin again
    assert sim.ground_truth().bin_xy != bin_unmoved and len(moved.steps) >= len(still.steps)
    assert replay(FakeSim(), tmp_path / "episodes" / moved.episode_id / "manifest.json")


def test_without_a_lift_the_bin_never_moves(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug", "seed": 1505}
    manifest = run_episode(sim, GiveUp(), move_bin_after_lift=True, **common)
    assert manifest.perturbations == [] and _move_bin_events(events, manifest.episode_id) == []  # trigger not reached


def test_the_trigger_reads_the_box_not_the_skill_name(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug", "seed": 1505}
    manifest = run_episode(sim, Calls("lift", "reobserve", "reobserve"), move_bin_after_lift=True, **common)
    assert manifest.steps[0].skill_call.skill == "lift" and manifest.perturbations == []  # lifted nothing


def test_a_bin_move_the_sim_refuses_is_reported_not_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = FakeSim()
    monkeypatch.setattr(sim, "perturb", lambda perturbation: False)
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug", "seed": 1505}
    manifest = run_episode(sim, ScriptedPolicy(sim), move_bin_after_lift=True, **common)
    (refused,) = _move_bin_events(events, manifest.episode_id)
    assert manifest.perturbations == [] and refused["type"] == "error"
    assert str(refused["data"]["error"]).startswith("move_bin not applied")


def test_scripted_episode_succeeds_and_takeover_is_recorded(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    manifest = run_episode(
        sim,
        GiveUp(),
        takeover=ScriptedPolicy(sim),
        run_dir=tmp_path,
        events=events,
        seed=1500,
        split="dev",
        round_=0,
        instruction="hug",
    )
    assert manifest.validation and manifest.validation.accepted
    assert manifest.takeover_step == 1
    assert [s.controller for s in manifest.steps][:1] == ["student"]


def test_heldout_seeds_are_sealed(tmp_path: Path) -> None:
    seed = min(heldout_seeds())
    with pytest.raises(SealedSeedError):
        assert_not_heldout(seed)
    assert not set(choose_scenes([], 30, start=0)) & heldout_seeds()


def test_training_set_takes_only_accepted_teacher_steps_from_dev(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "round_": 0, "instruction": "hug"}
    accepted = run_episode(sim, GiveUp(), takeover=ScriptedPolicy(sim), seed=1501, split="dev", **common)
    run_episode(sim, GiveUp(), seed=1502, split="dev", **common)  # rejected
    run_episode(sim, ScriptedPolicy(sim), seed=min(heldout_seeds()), split="heldout", **common)  # held out
    lines = build_training_set(tmp_path, 1).read_text().splitlines()
    teacher_steps = [s for s in accepted.steps if s.controller == "teacher"]
    assert len(lines) == len(teacher_steps) > 0
    assert all(accepted.episode_id in Path(json.loads(line)["image"]).parts for line in lines)


def test_skeleton_loop_runs_end_to_end(tmp_path: Path) -> None:
    calls: list[object] = []

    def fake(purpose: Purpose, parts: list[Part]) -> str:
        calls.append(parts)
        return stub_generate(purpose, parts)

    argon = Argon(EventLog(tmp_path, "t"), generate=fake, sleep=lambda s: pytest.fail(f"waited {s} s"))
    results = run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=2,
        dev_trials=2,
        argon=argon,
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    assert calls  # the teacher ran, on the fake: no live Gemini call even with a key in .env
    assert [r.round for r in results] == [0, 1]
    assert all(r.n == len(heldout_seeds()) for r in results)
    event_types = {json.loads(line)["type"] for line in (tmp_path / "events.jsonl").read_text().splitlines()}
    assert {"evaluation_result", "finetune_completed", "validation"} <= event_types


def test_decisions_carry_reason_and_latency_but_training_targets_drop_the_reason(tmp_path: Path) -> None:
    class Explains(ScriptedPolicy):
        def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
            call = super().act(obs, instruction, feedback)
            return call.model_copy(update={"reason": "block is right of the hand"}) if call else None

    sim = FakeSim()
    events = EventLog(tmp_path, "t")
    manifest = run_episode(
        sim,
        GiveUp(),
        takeover=Explains(sim),
        run_dir=tmp_path,
        events=events,
        seed=1504,
        split="dev",
        round_=0,
        instruction="push",
    )
    events_seen = [json.loads(line) for line in events.path.read_text().splitlines()]
    decisions = [e["data"] for e in events_seen if e["type"] == "decision"]
    assert all("latency_ms" in d for d in decisions if not d.get("takeover"))
    assert all("reason" in d for d in decisions)
    assert any(d["reason"] == "block is right of the hand" for d in decisions)
    examples = training_examples(tmp_path / "episodes" / manifest.episode_id, manifest)
    assert examples and all(type(e.target) is SkillTarget for e in examples)
    lines = build_training_set(tmp_path, 1).read_text().splitlines()
    assert lines and all("reason" not in json.loads(line)["target"] for line in lines)


def test_fake_sim_reports_the_robot_turned_by_its_waist() -> None:
    sim = FakeSim()
    sim.reset(1700)
    assert sim.ground_truth().robot_yaw == 0.0
    sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 30}))
    gt = sim.ground_truth()
    assert gt.robot_xy == (0.0, 0.0)
    assert gt.robot_yaw == pytest.approx(math.radians(30))


class CrashesOn(FakeSim):
    """A sim whose reset fails on some seeds, standing in for any crash mid-episode."""

    def __init__(self, seeds: set[int]) -> None:
        self.crash_seeds = seeds  # before super().__init__(), which resets
        super().__init__()

    def reset(self, seed: int) -> str:
        if seed in self.crash_seeds:
            raise RuntimeError(f"sim broke on {seed}")
        return super().reset(seed)


def test_a_heldout_episode_that_crashes_is_errored_and_the_evaluation_goes_on(tmp_path: Path) -> None:
    seeds = sorted(heldout_seeds())
    sim = CrashesOn({seeds[0], seeds[1]})
    events = EventLog(tmp_path, "t")
    result = evaluate(sim, "ckpt-r0", 0, tmp_path, events, lambda checkpoint: ScriptedPolicy(sim))
    assert (result.successes, result.errored, result.n) == (len(seeds) - 2, 2, len(seeds))
    assert result.rate == 1.0  # successes out of n - errored (SPEC §3.5)
    errors = [json.loads(line)["data"] for line in events.path.read_text().splitlines() if '"error"' in line]
    assert {e["seed"] for e in errors if e.get("split") == "heldout"} == {seeds[0], seeds[1]}


def test_a_recording_error_verdict_is_errored_not_a_student_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = loop.run_episode
    refused = sorted(heldout_seeds())[0]

    def run_episode_refusing_one(*args: object, **kwargs: object) -> EpisodeManifest:
        manifest = real(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs["seed"] == refused:
            return manifest.model_copy(update={"validation": recording_error_verdict(ValueError("timestamps"))})
        return manifest

    monkeypatch.setattr(loop, "run_episode", run_episode_refusing_one)
    result = evaluate(FakeSim(), "ckpt-r0", 0, tmp_path, EventLog(tmp_path, "t"), lambda checkpoint: GiveUp())
    assert (result.successes, result.errored) == (0, 1)


class MiscountsOn(FakeSim):
    """A sim bug on some seeds: every skill reports one timestamp more than the actions it executed (rule 4)."""

    def __init__(self, seeds: set[int]) -> None:
        self.bad_seeds = seeds  # before super().__init__(), which resets
        super().__init__()

    def reset(self, seed: int) -> str:
        self.seed = seed
        return super().reset(seed)

    def execute(self, call: SkillCall) -> ExecResult:
        result = super().execute(call)
        if self.seed in self.bad_seeds:
            return result.model_copy(update={"timestamps": [*result.timestamps, 0.0]})
        return result


def test_a_step_the_recorder_refuses_ends_that_episode_as_a_recording_error_and_the_run_goes_on(
    tmp_path: Path,
) -> None:
    heldout, trial = sorted(heldout_seeds())[0], DEV_SEED_START
    results = run_loop(
        MiscountsOn({heldout, trial}),
        tmp_path,
        "t",
        max_rounds=1,
        demos=1,
        dev_trials=2,
        argon=silent_argon(tmp_path, "t"),
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    assert [(r.round, r.n, r.errored) for r in results] == [(0, len(heldout_seeds()), 1), (1, len(heldout_seeds()), 1)]
    manifests = [EpisodeManifest.model_validate_json(p.read_text()) for p in tmp_path.glob("episodes/*/manifest.json")]
    refused = {m.episode_id: m for m in manifests if m.seed in (heldout, trial)}
    assert len(refused) == 3  # the held-out seed in both evaluations, and the trial
    for manifest in refused.values():
        assert manifest.validation is not None and not manifest.validation.accepted
        assert manifest.validation.reason.startswith("recording error: ")
        assert len(manifest.steps) == 0  # the refused step is never saved
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    errors = {e["data"].get("episode_id"): e["data"]["error"] for e in events if e["type"] == "error"}
    verdicts = {e["data"]["episode_id"]: e["data"]["reason"] for e in events if e["type"] == "validation"}
    assert all(errors[i] == verdicts[i] == m.validation.reason for i, m in refused.items() if m.validation)
    diagnosed = {e["data"]["episode_id"] for e in events if e["type"] == "diagnosis"}
    assert diagnosed and not diagnosed & refused.keys()  # a simulator bug is never a lesson


def test_a_student_that_cannot_be_built_stops_the_evaluation(tmp_path: Path) -> None:
    def missing(checkpoint: str) -> GiveUp:
        raise FileNotFoundError(f"no adapter for {checkpoint}")

    with pytest.raises(FileNotFoundError):
        evaluate(FakeSim(), "ckpt-r9", 0, tmp_path, EventLog(tmp_path, "t"), missing)


def silent_argon(run_dir: Path, run_id: str) -> Argon:
    """No live Gemini call: these tests are about the student seams."""
    return Argon(EventLog(run_dir, run_id), generate=stub_generate)


def test_loop_runs_the_given_student_and_trainer(tmp_path: Path) -> None:
    built: list[str] = []

    def make_student(checkpoint: str) -> StubStudent:
        built.append(checkpoint)
        return StubStudent(checkpoint)

    def train_student(training_set: Path, previous_checkpoint: str, round_: int) -> str:
        return f"trained-{round_}"

    run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=1,
        dev_trials=1,
        argon=silent_argon(tmp_path, "t"),
        make_student=make_student,
        train_student=train_student,
    )
    assert built[0] == "ckpt-r0"
    assert "trained-1" in built


def test_a_run_is_tested_trialled_and_taught_on_its_own_instruction(tmp_path: Path) -> None:
    instruction = "Drop the orange box into the green bin, gently."
    run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=1,
        dev_trials=1,
        argon=silent_argon(tmp_path, "t"),
        make_student=StubStudent,
        train_student=stub_finetune,
        instruction=instruction,
    )
    manifests = [EpisodeManifest.model_validate_json(p.read_text()) for p in tmp_path.glob("episodes/*/manifest.json")]
    kinds = {(m.split, m.driver) for m in manifests}
    assert kinds == {("heldout", "student"), ("dev", "student"), ("dev", "teacher")}
    assert {m.instruction for m in manifests} == {instruction}


def test_the_cli_refuses_a_run_id_a_checkpoint_id_could_not_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    argv = ["mom", "run", "--sim", "fake", "--student", "stub", "--rounds", "0", "--run-id", "demo 1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_the_cli_sizes_a_round_with_demos_and_dev_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: dict[str, object] = {}

    def run_loop_spy(*args: object, **kwargs: object) -> list[object]:
        sizes.update(demos=kwargs["demos"], dev_trials=kwargs["dev_trials"], share=kwargs["move_bin_share"])
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_loop", run_loop_spy)
    base = ["mom", "run", "--sim", "fake", "--student", "stub", "--rounds", "1"]
    monkeypatch.setattr(sys, "argv", [*base, "--demos", "80", "--dev-trials", "15", "--move-bin-share", "0.5"])
    cli.main()
    assert sizes == {"demos": 80, "dev_trials": 15, "share": 0.5}
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    assert sizes == {"demos": loop.DEMOS_PER_ROUND, "dev_trials": loop.DEV_TRIALS, "share": loop.MOVE_BIN_SHARE}


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--demos", "0"), ("--dev-trials", "0"), ("--move-bin-share", "1.5"), ("--move-bin-share", "-0.1")],
)
def test_the_cli_refuses_a_round_size_below_one_or_a_share_above_all(
    flag: str, value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mom", "run", "--sim", "fake", "--student", "stub", flag, value])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 2


def finetune_events(tmp_path: Path, train_student: Callable[[Path, str, int], str]) -> tuple[dict, dict]:
    run_dir = tmp_path / "some-run"
    run_loop(
        FakeSim(),
        run_dir,
        "some-run",
        max_rounds=1,
        demos=1,
        dev_trials=1,
        argon=silent_argon(run_dir, "some-run"),
        make_student=StubStudent,
        train_student=train_student,
    )
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    [started] = [e["data"] for e in events if e["type"] == "finetune_started"]
    [completed] = [e["data"] for e in events if e["type"] == "finetune_completed"]
    return started, completed


def test_a_skipped_finetune_says_so_and_names_no_adapter(tmp_path: Path) -> None:
    _, completed = finetune_events(tmp_path, stub_finetune)
    assert (completed["skipped"], completed["examples"], completed["checkpoint_path"]) == (True, 0, None)
    assert isinstance(completed["duration_s"], float)


def test_a_finetune_reports_what_it_trained_on_how_long_and_where(tmp_path: Path) -> None:
    started, completed = finetune_events(tmp_path, lambda training_set, previous, round_: f"some-run/ckpt-r{round_}")
    assert (completed["checkpoint"], completed["skipped"]) == ("some-run/ckpt-r1", False)
    assert completed["examples"] == len(Path(started["training_set"]).read_text().splitlines())
    assert isinstance(completed["duration_s"], float) and completed["duration_s"] >= 0
    assert Path(completed["checkpoint_path"]).parts == (*tmp_path.parts, "some-run", "checkpoints", "ckpt-r1")


def test_a_stated_share_of_teacher_demonstrations_gets_the_bin_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = loop.run_episode
    asked: list[tuple[str, bool]] = []

    def recorded(sim: FakeSim, policy: Any, **kwargs: Any) -> EpisodeManifest:
        asked.append((policy.controller, kwargs.get("move_bin_after_lift", False)))
        return real(sim, policy, **kwargs)

    monkeypatch.setattr(loop, "run_episode", recorded)
    run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=8,
        dev_trials=1,
        argon=silent_argon(tmp_path, "t"),
        make_student=StubStudent,
        train_student=stub_finetune,
        move_bin_share=0.25,
    )
    assert [moved for controller, moved in asked if controller == "teacher"] == [False, False, False, True] * 2
    assert not any(moved for controller, moved in asked if controller == "student")  # trials and held-out: unmoved
    logged = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    scheduled = [e["data"]["move_bin_after_lift"] for e in logged if e["type"] == "episode_started"]
    assert sum(scheduled) == 2 and len(scheduled) == len(asked)  # every episode states whether its move was due


@pytest.mark.parametrize(
    ("demos", "share", "moved"), [(20, 0.25, {3, 7, 11, 15, 19}), (5, 0.0, set()), (3, 1.0, {0, 1, 2}), (4, 0.1, set())]
)
def test_the_share_is_spread_evenly(demos: int, share: float, moved: set[int]) -> None:
    assert loop.moves_bin(demos, share) == moved


@pytest.mark.parametrize("share", [1.5, -0.1])
def test_a_share_outside_zero_to_one_is_refused(share: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="share of the demonstrations"):
        run_loop(
            FakeSim(),
            tmp_path,
            "t",
            argon=silent_argon(tmp_path, "t"),
            make_student=StubStudent,
            train_student=stub_finetune,
            max_rounds=0,
            move_bin_share=share,
        )


def test_the_run_reports_the_bin_moves_that_happened_not_the_share_asked_for(tmp_path: Path) -> None:
    events = EventLog(tmp_path, "t")
    for episode, scheduled in [("a", True), ("b", True), ("c", False)]:
        events.emit("episode_started", episode_id=episode, move_bin_after_lift=scheduled)
        events.emit("decision", episode_id=episode, controller="teacher")
    events.emit("perturbation", episode_id="a", kind="move_bin")  # b was scheduled but never lifted
    events.emit("episode_started", episode_id="s", move_bin_after_lift=False)
    events.emit("decision", episode_id="s", controller="student")
    expected = "the bin moved right after the lift in 1 of 3 teacher demonstrations (2 scheduled)"
    assert loop.bin_moves(tmp_path / "events.jsonl") == expected
    assert loop.bin_moves(tmp_path / "missing.jsonl").startswith("the bin moved right after the lift in 0 of 0")
