import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from isaacs_mom.contracts import Controller, EpisodeManifest, Observation, Perturbation, SkillCall, SkillTarget
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.seeds import SealedSeedError, assert_not_heldout, heldout_seeds
from isaacs_mom.sim.fake import BIN_OUTER_HALF, FLOOR_Z, PLACE_X, FakeSim
from isaacs_mom.teacher import choose_scenes
from isaacs_mom.validator import validate
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

    sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 30}))
    sim.execute(SkillCall(skill="hug"))  # turned away from the table, so the hands close on nothing
    sim.execute(SkillCall(skill="turn_right", params={"angle_deg": 30}))
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
    results = run_loop(FakeSim(), tmp_path, "t", max_rounds=1, demos=2, dev_trials=2)
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
