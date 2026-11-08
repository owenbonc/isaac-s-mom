import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from isaacs_mom.contracts import (
    Controller,
    Diagnosis,
    EpisodeManifest,
    ExecResult,
    GroundTruth,
    Observation,
    Perturbation,
    ReplayReport,
    SkillCall,
    SkillTarget,
    TrainingExample,
    TrainingSetSummary,
    Validation,
)
from isaacs_mom.orchestrator import cli, loop
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import carry_over, check_carry_over, evaluate, lesson_plan, run_loop
from isaacs_mom.policy import Policy
from isaacs_mom.seeds import DEV_SEED_START, SealedSeedError, assert_not_heldout, heldout_seeds
from isaacs_mom.sim.fake import BIN_OUTER_HALF, FLOOR_Z, HELD_Z, PLACE_X, FakeSim, _bin_pose
from isaacs_mom.student import StubStudent, stub_finetune
from isaacs_mom.student.checkpoints import ADAPTER_FILES
from isaacs_mom.tasks.example import OFFLINE_PROPOSAL
from isaacs_mom.tasks.spec import TaskSpec
from isaacs_mom.teacher import FailureMode, choose_scenes
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
    assert "error" not in event_types  # the stub's round diagnosis is usable, so the skeleton run logs no error


def test_a_run_ends_by_replaying_every_accepted_episode(tmp_path: Path) -> None:
    sim = FakeSim()
    run_loop(sim, tmp_path, "t", max_rounds=0, argon=silent_argon(tmp_path, "t"),
             make_student=lambda checkpoint: ScriptedPolicy(sim), train_student=stub_finetune)  # fmt: skip
    saved = [EpisodeManifest.model_validate_json(p.read_text()) for p in tmp_path.glob("episodes/*/manifest.json")]
    accepted = [m for m in saved if m.validation and m.validation.accepted]
    assert (tmp_path / "replay_report.json").exists()
    report = ReplayReport.model_validate_json((tmp_path / "replay_report.json").read_text())
    assert report.replayed == len(accepted) == len(heldout_seeds())  # SPEC §3.6 step 6
    assert report.reproduced == report.replayed and all(m.replay_valid for m in accepted)


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


def _lesson(name: str) -> Diagnosis:
    return Diagnosis(hypothesis=name, evidence="seen", confidence=0.8, lesson=name)


def _mode(name: str, ids: list[str]) -> FailureMode:
    return FailureMode(diagnosis=_lesson(name), explains=ids)


def test_each_failure_is_replayed_first_then_the_modes_share_the_rest_by_the_failures_each_explains() -> None:
    far, short = [f"ep-{i}-aa" for i in range(6)], [f"ep-{i}-bb" for i in range(6, 10)]
    failures = [f for pair in zip(far, short, strict=False) for f in pair] + far[4:]  # the two modes interleaved
    plan = lesson_plan(20, [_mode("far", far), _mode("short", short)], [(f, _lesson(f"own {f}")) for f in failures])
    taught = [d.hypothesis for _, d in plan if d]
    assert [acting_on for acting_on, _ in plan[:10]] == failures  # demo i replays failure i, on its own scene
    assert taught[:10] == ["far" if f in far else "short" for f in failures]  # with its own mode's lesson
    assert taught.count("far") == 12 and taught.count("short") == 8  # 6:4 of 20, the replays counted in
    assert {"far", "short"} <= set(taught[10:13])  # the rest spread through the round, not one mode then the other
    for acting_on, diagnosis in plan[10:]:
        assert diagnosis is not None and acting_on in (far if diagnosis.hypothesis == "far" else short)


def test_a_failure_goes_to_the_first_mode_that_explains_it_and_one_no_mode_explains_keeps_its_own_lesson() -> None:
    own = [(f"ep-{i}-aa", _lesson(f"own {i}")) for i in (1, 2, 3)]
    plan = lesson_plan(4, [_mode("a", ["ep-1-aa"]), _mode("b", ["ep-1-aa", "ep-2-aa"])], own)
    assert [(acting_on, d.hypothesis if d else None) for acting_on, d in plan] == [
        ("ep-1-aa", "a"),
        ("ep-2-aa", "b"),
        ("ep-3-aa", "own 3"),
        ("ep-2-aa", "b"),  # furthest behind its share: b explains two failures and has taught one
    ]


def test_without_modes_the_per_episode_lessons_take_turns() -> None:
    diagnosed = [("ep-1-aa", _lesson("one")), ("ep-2-aa", _lesson("two"))]
    assert lesson_plan(3, [], diagnosed) == [diagnosed[0], diagnosed[1], diagnosed[0]]
    assert lesson_plan(2, [], []) == [(None, None), (None, None)]


def _round_argon(run_dir: Path, answer: Callable[[list[str]], str]) -> tuple[Argon, list[str]]:
    """A stub teacher whose round-level diagnosis is `answer(failed ids)`; act prompts are kept."""
    acted: list[str] = []

    def generate(purpose: Purpose, parts: list[Part]) -> str:
        text = str(parts[0])
        if purpose == "diagnose" and "Group the failures by cause" in text:
            return answer(re.findall(r"trial (ep-\S+) \(failed\)", text))
        if purpose == "act":
            acted.append(text)
        return stub_generate(purpose, parts)

    return Argon(EventLog(run_dir, "t"), generate=generate), acted


def _demos(run_dir: Path) -> list[dict[str, Any]]:
    logged = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    return [e["data"] for e in logged if e["type"] == "episode_started" and e["data"].get("acting_on")]


def _logged(run_dir: Path, kind: str) -> list[dict[str, Any]]:
    logged = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    return [e["data"] for e in logged if e["type"] == kind]


def test_a_round_teaches_by_its_failure_modes_and_records_them(tmp_path: Path) -> None:
    def one_mode(failed: list[str]) -> str:
        mode = {
            "hypothesis": "twists too far",
            "evidence": "every failure",
            "confidence": 0.9,
            "lesson": "stop twisting once the bin is ahead",
            "explains": failed,
        }
        return json.dumps({"modes": [mode]})

    argon, acted = _round_argon(tmp_path, one_mode)
    run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=3,
        dev_trials=2,
        argon=argon,
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    written = json.loads((tmp_path / "diagnosis_r0.json").read_text())
    assert written["fell_back"] is False
    (record,) = written["modes"]
    failed = [d["episode_id"] for d in _logged(tmp_path, "diagnosis")]
    assert record["diagnosis"]["hypothesis"] == "twists too far" and len(record["demo_seeds"]) == 3
    assert {d["acting_on"] for d in _demos(tmp_path)} <= set(failed)  # a failed episode its mode names
    assert acted and all("stop twisting once the bin is ahead" in prompt for prompt in acted)
    assert len(failed) == 2  # the per-episode diagnoses still run, one per failed trial


def test_with_two_modes_each_rejected_scene_gets_its_own_failures_lesson_and_the_record_says_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def two_modes(failed: list[str]) -> str:
        return json.dumps(
            {
                "modes": [
                    {
                        "hypothesis": "a",
                        "evidence": "e",
                        "confidence": 0.9,
                        "lesson": "lesson a",
                        "explains": failed[1:],
                    },
                    {
                        "hypothesis": "b",
                        "evidence": "e",
                        "confidence": 0.9,
                        "lesson": "lesson b",
                        "explains": failed[:1],
                    },
                ]
            }
        )

    lessons: list[str | None] = []
    teacher = loop.ArgonPolicy

    def policy(diagnosis: Diagnosis | None, argon: Argon) -> Any:
        lessons.append(diagnosis.lesson if diagnosis else None)
        return teacher(diagnosis, argon)

    monkeypatch.setattr(loop, "ArgonPolicy", policy)
    argon, _ = _round_argon(tmp_path, two_modes)
    run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=3,
        dev_trials=2,
        argon=argon,
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    seed_of = {d["episode_id"]: d["seed"] for d in _logged(tmp_path, "episode_started")}
    failed = [d["episode_id"] for d in _logged(tmp_path, "diagnosis")]
    demos = _demos(tmp_path)
    assert len(failed) == 2 and [d["acting_on"] for d in demos[:2]] == failed  # demo i replays failure i
    assert all(d["seed"] == seed_of[d["acting_on"]] for d in demos[:2])  # on that failure's own scene
    assert lessons == ["lesson b", "lesson a", "lesson a"]  # with the lesson of the mode that explains it
    taught = {
        lesson: [d["seed"] for d, given in zip(demos, lessons, strict=True) if given == lesson] for lesson in lessons
    }
    record = json.loads((tmp_path / "diagnosis_r0.json").read_text())["modes"]
    assert [(r["diagnosis"]["lesson"], r["explains"], r["demo_seeds"]) for r in record] == [
        ("lesson a", failed[1:], taught["lesson a"]),
        ("lesson b", failed[:1], taught["lesson b"]),
    ]  # rule 6: each mode, and the seeds that got its lesson


def test_a_trial_ended_by_a_recording_error_is_not_in_the_rounds_diagnosis_either(tmp_path: Path) -> None:
    named: list[list[str]] = []

    def one_mode(failed: list[str]) -> str:
        named.append(failed)
        mode = {"hypothesis": "h", "evidence": "e", "confidence": 0.5, "lesson": "l", "explains": failed}
        return json.dumps({"modes": [mode]})

    argon, _ = _round_argon(tmp_path, one_mode)
    run_loop(
        MiscountsOn({DEV_SEED_START}),
        tmp_path,
        "t",
        max_rounds=1,
        demos=1,
        dev_trials=2,
        argon=argon,
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    bad = {d["episode_id"] for d in _logged(tmp_path, "episode_started") if d["seed"] == DEV_SEED_START}
    assert bad and named == [[d["episode_id"] for d in _logged(tmp_path, "diagnosis")]]  # the other trial only
    assert not bad & set(named[0])


def test_a_round_whose_trials_all_pass_records_no_modes_and_no_fallback(tmp_path: Path) -> None:
    sim, built = FakeSim(), []

    def make_student(checkpoint: str) -> Policy:  # fails the held-out evaluation, then passes every trial
        built.append(checkpoint)
        return StubStudent(checkpoint) if len(built) <= len(heldout_seeds()) else ScriptedPolicy(sim)

    run_loop(
        sim,
        tmp_path,
        "t",
        max_rounds=1,
        demos=1,
        dev_trials=2,
        argon=silent_argon(tmp_path, "t"),
        make_student=make_student,
        train_student=stub_finetune,
    )
    assert json.loads((tmp_path / "diagnosis_r0.json").read_text()) == {"fell_back": False, "modes": []}


def test_an_unusable_round_diagnosis_leaves_the_per_episode_lessons(tmp_path: Path) -> None:
    argon, acted = _round_argon(tmp_path, lambda failed: "not json")
    run_loop(
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
    assert json.loads((tmp_path / "diagnosis_r0.json").read_text()) == {"fell_back": True, "modes": []}
    assert [d["acting_on"] for d in _demos(tmp_path)] and acted  # the round still teaches, on the per-episode lessons


class StopsAtOnce:
    """A teacher whose demonstration leaves the box on the table: recorded, rejected."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return SkillCall(skill="reobserve") if obs.step == 0 else None


class SolvesAlone(ScriptedPolicy):
    controller: Controller = "student"


def _finished(checkpoint: Path) -> Path:
    checkpoint.mkdir(parents=True)
    for name in ADAPTER_FILES:
        (checkpoint / name).write_text("{}")
    return checkpoint


def _earlier_run(runs: Path, *policies: str) -> Path:
    """A finished run beside the next one: a demonstration per named policy on its own dev seed, and its checkpoint."""
    prev = runs / "prev"
    sim, events = FakeSim(), EventLog(prev, "prev")
    make = {"taught": ScriptedPolicy, "rejected": lambda _sim: StopsAtOnce(), "alone": SolvesAlone}
    for i, name in enumerate(policies):
        run_episode(sim, make[name](sim), run_dir=prev, events=events, seed=1500 + i, split="dev", round_=1,
                    instruction="hug")  # fmt: skip
    _finished(prev / "checkpoints" / "ckpt-r1")
    return prev


def _manifests(run: Path) -> dict[str, EpisodeManifest]:
    return {
        p.parent.name: EpisodeManifest.model_validate_json(p.read_text()) for p in run.glob("episodes/*/manifest.json")
    }


def _rewrite(run: Path, index: int, **update: Any) -> None:
    path = sorted(run.glob("episodes/*/manifest.json"))[index]
    manifest = EpisodeManifest.model_validate_json(path.read_text())
    path.write_text(manifest.model_copy(update=update).model_dump_json())


def _from_run(tmp_path: Path, prev: Path, name: str = "next") -> tuple[Path, list[str], list[tuple[str, list[str]]]]:
    evaluated: list[str] = []
    trained: list[tuple[str, list[str]]] = []

    def make_student(checkpoint: str) -> StubStudent:
        evaluated.append(checkpoint)
        return StubStudent(checkpoint)

    def train_student(training_set: Path, previous_checkpoint: str, round_: int) -> str:
        images = [TrainingExample.model_validate_json(line).image for line in training_set.read_text().splitlines()]
        trained.append((previous_checkpoint, [Path(image).parents[1].name for image in images]))
        _finished(training_set.parent / "checkpoints" / f"ckpt-r{round_}")
        return f"{training_set.parent.name}/ckpt-r{round_}"

    run_dir = tmp_path / "runs" / name
    run_loop(FakeSim(), run_dir, name, max_rounds=1, demos=1, dev_trials=1, argon=silent_argon(run_dir, name),
             make_student=make_student, train_student=train_student, from_run=prev)  # fmt: skip
    return run_dir, evaluated, trained


def _refused(run_dir: Path, prev: Path) -> BaseException:
    with pytest.raises(Exception) as refused:
        run_loop(FakeSim(), run_dir, run_dir.name, argon=silent_argon(run_dir, "x"), make_student=StubStudent,
                 train_student=stub_finetune, from_run=prev)  # fmt: skip
    assert not (run_dir / "episodes").exists() and not (run_dir / "from_run.json").exists()  # nothing copied
    return refused.value


def test_round_0_is_the_earlier_runs_checkpoint_evaluated_afresh_with_its_demonstrations_still_training(
    tmp_path: Path,
) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught")
    (demo,) = _manifests(prev)
    run_dir, evaluated, trained = _from_run(tmp_path, prev)
    assert evaluated[0] == "prev/ckpt-r1"
    logged = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    results = [e["data"] for e in logged if e["type"] == "evaluation_result"]
    assert results[0]["round"] == 0 and results[0]["n"] == len(heldout_seeds())  # measured in this run
    assert results[0]["from_run"] == {"run": "prev", "checkpoint": "prev/ckpt-r1", "carried": 1}  # rule 6
    assert "from_run" not in results[1]
    held_out = [m for m in _manifests(run_dir).values() if m.split == "heldout" and m.round == 0]
    assert len(held_out) == len(heldout_seeds())
    assert trained and trained[0][0] == "prev/ckpt-r1" and demo in trained[0][1]  # the old demonstration still trains
    summary = TrainingSetSummary.model_validate_json((run_dir / "training_r1.summary.json").read_text())
    assert summary.carried == 1 and summary.accepted >= 1  # not counted as this run's teaching
    record = run_dir / "from_run.json"
    assert record.exists()
    assert json.loads(record.read_text()) == {"from_run": "prev", "checkpoint": "prev/ckpt-r1", "episodes": [demo],
                                              "left_behind": []}  # fmt: skip


def test_copied_demonstrations_stay_apart_from_this_runs_own(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught")
    (demo,) = _manifests(prev)
    run_dir, _, _ = _from_run(tmp_path, prev)
    copied = _manifests(run_dir)[demo]
    assert copied == _manifests(prev)[demo].model_copy(update={"replay_valid": True})  # the earlier run's, replayed
    assert copied.run_id == "prev" and {s.controller for s in copied.steps} == {"teacher"}
    logged = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert demo not in {e["data"].get("episode_id") for e in logged}  # nothing in this run's events claims it


def test_only_accepted_teacher_demonstrations_come_in(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught", "rejected", "alone")
    by_seed = {m.seed: name for name, m in _manifests(prev).items()}
    run_dir, _, trained = _from_run(tmp_path, prev)
    assert set(json.loads((run_dir / "from_run.json").read_text())["episodes"]) == {by_seed[1500]}
    assert not {by_seed[1501], by_seed[1502]} & set(_manifests(run_dir))  # rule 3: never reach the training set
    assert not {by_seed[1501], by_seed[1502]} & set(trained[0][1])


def test_a_real_mom_run_is_a_source_its_exams_stay_behind_and_runs_chain_without_copying_twice(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first, _, _ = _from_run(tmp_path, _earlier_run(runs, "taught"), name="first")  # a mom run with its own exams
    exams = {i for i, m in _manifests(first).items() if m.split == "heldout"}
    assert len(exams) == 2 * len(heldout_seeds())  # round 0 and round 1
    second, evaluated, _ = _from_run(tmp_path, first, name="second")
    record = json.loads((second / "from_run.json").read_text())
    assert set(record["left_behind"]) == exams and not exams & set(_manifests(second))  # rule 2: listed, not copied
    assert evaluated[0] == "first/ckpt-r1"
    taught = {i for i, m in _manifests(first).items() if m.split == "dev" and m.validation and m.validation.accepted
              and any(s.controller == "teacher" for s in m.steps)}  # fmt: skip
    copied = [i for i, m in _manifests(second).items() if m.run_id != "second" and m.split == "dev"]
    assert sorted(copied) == sorted(taught)  # each demonstration once, however many runs it came through


def test_a_dev_episode_on_a_held_out_seed_stops_the_run(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught", "taught")
    _rewrite(prev, 1, seed=min(heldout_seeds()))
    assert isinstance(_refused(tmp_path / "runs" / "next", prev), SealedSeedError)  # rule 2


def test_a_held_out_episode_on_a_seed_that_isnt_sealed_is_refused(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught", "taught")
    _rewrite(prev, 1, split="heldout", seed=900001)
    assert "isn't sealed" in str(_refused(tmp_path / "runs" / "next", prev))


@pytest.mark.parametrize("split", ["dev", "heldout"])
def test_a_source_on_another_task_is_refused_even_by_an_exam_it_leaves_behind(split: str, tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught", "taught")
    seed = 1501 if split == "dev" else min(heldout_seeds())
    _rewrite(prev, 1, sim_version="mujoco-4.0.0 task=across_room", split=split, seed=seed)
    assert "was recorded on" in str(_refused(tmp_path / "runs" / "next", prev))


def test_a_source_with_no_episodes_is_refused(tmp_path: Path) -> None:
    prev = tmp_path / "runs" / "prev"
    _finished(prev / "checkpoints" / "ckpt-r1")
    assert "no episodes" in str(_refused(tmp_path / "runs" / "next", prev))


def test_round_0_starts_from_the_earlier_runs_last_finished_checkpoint(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught")
    run_dir = tmp_path / "runs" / "next"
    (prev / "checkpoints" / "ckpt-r3").mkdir()  # empty: not finished
    assert check_carry_over(prev, run_dir, FakeSim().sim_version).checkpoint == "prev/ckpt-r1"
    for name in ("ckpt-r2", "ckpt-r10", "ckpt-r01", "ckpt-r11.tmp", "ckpt-r\u00b2"):
        _finished(prev / "checkpoints" / name)
    (prev / "checkpoints" / "ckpt-r12").write_text("a file, not a checkpoint")
    assert check_carry_over(prev, run_dir, FakeSim().sim_version).checkpoint == "prev/ckpt-r10"  # numbered, not text


def test_starting_twice_into_one_run_is_refused(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught")
    run_dir = tmp_path / "runs" / "next"
    carry_over(check_carry_over(prev, run_dir, FakeSim().sim_version), run_dir)
    with pytest.raises(Exception) as refused:
        check_carry_over(prev, run_dir, FakeSim().sim_version)
    assert "already holds episodes" in str(refused.value)


def test_a_copy_that_fails_halfway_leaves_nothing_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught", "taught")
    run_dir = tmp_path / "runs" / "next"
    carry = check_carry_over(prev, run_dir, FakeSim().sim_version)
    copies: list[Path] = []
    real = loop.shutil.copytree

    def fails_second(src: Path, dst: Path, *nested: Any) -> Any:
        if nested:  # copytree's own call for a subfolder
            return real(src, dst, *nested)
        copies.append(dst)
        if len(copies) == 2:
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(loop.shutil, "copytree", fails_second)
    with pytest.raises(OSError):
        carry_over(carry, run_dir)
    assert not (run_dir / "episodes").exists() and not (run_dir / "from_run.json").exists()
    monkeypatch.setattr(loop.shutil, "copytree", real)
    carry_over(check_carry_over(prev, run_dir, FakeSim().sim_version), run_dir)
    assert len(_manifests(run_dir)) == 2


def test_what_was_checked_is_what_is_copied(tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / "runs", "taught")
    run_dir = tmp_path / "runs" / "next"
    carry = check_carry_over(prev, run_dir, FakeSim().sim_version)
    rejected = Validation(task_success=False, physics_valid=True, reason="changed after the check")
    _rewrite(prev, 0, validation=rejected)
    carry_over(carry, run_dir)
    (copied,) = _manifests(run_dir).values()
    assert copied == carry.episodes[0][1] and copied.validation and copied.validation.accepted


@pytest.mark.parametrize("where", ["elsewhere", "no checkpoint"])
def test_a_run_refuses_to_start_from_a_run_it_cannot_continue(where: str, tmp_path: Path) -> None:
    prev = _earlier_run(tmp_path / ("runs" if where == "no checkpoint" else "other"), "taught")
    if where == "no checkpoint":
        for name in ADAPTER_FILES:
            (prev / "checkpoints" / "ckpt-r1" / name).unlink()
    refused = _refused(tmp_path / "runs" / "next", prev)
    assert "must sit beside" in str(refused) or "no finished checkpoint" in str(refused)


def test_a_refused_from_run_is_a_usage_error_that_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "prev" / "checkpoints" / "ckpt-r1").mkdir(parents=True)  # not finished
    argv = ["mom", "run", "--sim", "fake", "--student", "stub", "--run-id", "next", "--from-run", "runs/prev"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(BaseException) as stopped:
        cli.main()
    assert isinstance(stopped.value, SystemExit) and stopped.value.code == 2
    assert not (tmp_path / "runs" / "next").exists()


def _rule(reason: str, accepted: bool) -> Callable[[GroundTruth], Validation]:
    return lambda _gt: Validation(task_success=accepted, physics_valid=True, reason=reason)


def test_a_generated_tasks_rule_judges_every_episode_the_loop_runs(tmp_path: Path) -> None:
    run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=1,
        demos=2,
        dev_trials=2,
        argon=silent_argon(tmp_path, "t"),
        make_student=StubStudent,
        train_student=stub_finetune,
        verdict=_rule("spec: box not on the shelf", False),
    )
    manifests = list(_manifests(tmp_path).values())
    kinds = {(m.split, any(s.controller == "teacher" for s in m.steps)) for m in manifests}
    assert kinds == {("heldout", False), ("dev", False), ("dev", True)}  # the exams, the trials and the demos
    assert {m.validation.reason for m in manifests if m.validation} == {"spec: box not on the shelf"}
    assert all(m.validation for m in manifests)


def test_the_loops_replay_judges_by_the_same_rule(tmp_path: Path) -> None:
    results = run_loop(
        FakeSim(),
        tmp_path,
        "t",
        max_rounds=0,
        argon=silent_argon(tmp_path, "t"),
        make_student=StubStudent,
        train_student=stub_finetune,
        verdict=_rule("spec: every rule holds", True),
    )
    report = ReplayReport.model_validate_json((tmp_path / "replay_report.json").read_text())
    assert results[0].successes == len(heldout_seeds()) == report.replayed == report.reproduced


def _spec_file(tmp_path: Path) -> tuple[Path, TaskSpec]:
    words = "put the box on the shelf, even if it moves"
    spec = TaskSpec.model_validate_json(json.dumps({**OFFLINE_PROPOSAL, "instruction": words}))
    path = tmp_path / "spec.json"
    path.write_text(spec.model_dump_json())
    return path, spec


def test_mom_run_teaches_a_generated_task_from_the_tasks_learn_run_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    path, spec = _spec_file(tmp_path)
    _earlier_run(tmp_path / "runs", "taught")
    judged: list[str] = []

    def rule(given: TaskSpec, gt: GroundTruth) -> Validation:
        judged.append(given.name)
        return Validation(task_success=False, physics_valid=True, reason="judged by the spec")

    monkeypatch.setattr(cli, "spec_sim", lambda _spec: FakeSim())  # the scene is MuJoCo's; the plumbing is the same
    monkeypatch.setattr(cli, "judge", rule)
    argv = [
        "mom",
        "run",
        "--sim",
        "mujoco",
        "--student",
        "stub",
        "--teacher",
        "stub",
        "--spec",
        str(path),
        "--from-run",
        "runs/prev",
        "--run-id",
        "next",
        "--rounds",
        "1",
        "--demos",
        "1",
        "--dev-trials",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    run_dir = tmp_path / "runs" / "next"
    assert (run_dir / "spec.json").is_file()
    assert TaskSpec.model_validate_json((run_dir / "spec.json").read_text()) == spec  # rule 6: what it was taught
    logged = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["data"]["text"] for e in logged if e["type"] == "instruction"] == [spec.instruction]
    carried = set(json.loads((run_dir / "from_run.json").read_text())["episodes"])
    new = {k: m for k, m in _manifests(run_dir).items() if k not in carried}
    assert new and {m.instruction for m in new.values()} == {spec.instruction}  # the person's words throughout
    assert {m.validation.reason for m in new.values() if m.validation} == {"judged by the spec"}
    assert judged and set(judged) == {spec.name}
    assert "round 0 started from prev/ckpt-r1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [["--sim", "fake"], ["--task", "near_bin"], ["--instruction", "something else"], ["--spec-missing"]],
)
def test_a_spec_the_run_cant_use_is_a_usage_error_that_writes_nothing(
    extra: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path, _ = _spec_file(tmp_path)
    if extra == ["--spec-missing"]:
        path, extra = tmp_path / "nowhere.json", []
    argv = ["mom", "run", "--student", "stub", "--teacher", "stub", "--spec", str(path), "--run-id", "next", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(BaseException) as stopped:
        cli.main()
    assert isinstance(stopped.value, SystemExit) and stopped.value.code == 2
    assert not (tmp_path / "runs" / "next").exists()
