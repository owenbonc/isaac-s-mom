import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from isaacs_mom.contracts import Controller, Observation, Perturbation, SkillCall, SkillTarget
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.seeds import SealedSeedError, assert_not_heldout, heldout_seeds
from isaacs_mom.sim.fake import FakeSim
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
        work_y, waist, closed, lifted = obs.joint_state
        if gt.box_on_bin_floor:
            return None
        if not closed:
            dy = gt.box_xyz[1] - work_y
            if abs(dy) > 0.01:
                return SkillCall(skill="reach_left" if dy > 0 else "reach_right", params={"distance_cm": abs(dy) * 100})
            return SkillCall(skill="hug")
        if not lifted:
            return SkillCall(skill="lift")
        turn = math.degrees(math.atan2(gt.bin_xy[1], gt.bin_xy[0]) - waist)
        if abs(turn) > 2:
            return SkillCall(skill="turn_left" if turn > 0 else "turn_right", params={"angle_deg": abs(turn)})
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
    assert SkillCall(skill="reach_left", params={"distance_cm": 500}).params["distance_cm"] == 20
    assert SkillCall(skill="turn_left", params={"angle_deg": 500}).params["angle_deg"] == 45


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


def test_fake_place_without_turning_drops_the_box_outside_the_bin() -> None:
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


def test_fake_refuses_an_out_of_reach_skill_without_moving() -> None:
    sim = FakeSim()
    sim.reset(1234)
    before = sim.observe().joint_state
    result = sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 45}))
    assert result.ok
    for _ in range(3):
        result = sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 45}))
    assert not result.ok and result.error and not result.executed_actions
    assert sim.observe().joint_state[1] == pytest.approx(before[1] + math.radians(135))


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
