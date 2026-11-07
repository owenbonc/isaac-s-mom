"""Workstream 1: the real MuJoCo scene (SPEC §6 Gate A, rule 7). Skipped with a reason where nothing can render."""

from pathlib import Path

import numpy as np
import pytest

from isaacs_mom.contracts import Perturbation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.validator import replay, validate

try:
    from isaacs_mom.sim.mujoco import MujocoSim
    from isaacs_mom.sim.mujoco.preview import OraclePolicy

    SIM = MujocoSim()
except Exception as e:  # a missing GL context (e.g. no osmesa) surfaces as several exception types
    pytest.skip(f"MuJoCo cannot render here: {e}", allow_module_level=True)

SEED = 1  # a layout the scripted policy completes; the box starts 9 cm left of centre


@pytest.fixture
def sim() -> MujocoSim:
    SIM.reset(SEED)
    return SIM


def test_same_seed_same_scene_and_moving_the_box_changes_the_frame(sim: MujocoSim) -> None:
    first = sim.observe().head_rgb_jpeg
    sim.reset(SEED)
    assert sim.observe().head_rgb_jpeg == first
    sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.observe().head_rgb_jpeg != first


def test_hug_and_lift_raise_the_box_with_physics(sim: MujocoSim) -> None:
    start = sim.ground_truth().box_xyz
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    sim.execute(SkillCall(skill="hug"))
    result = sim.execute(SkillCall(skill="lift"))
    assert result.ok and len(result.executed_actions) == len(result.timestamps) > 0
    assert np.all(np.diff(result.timestamps) > 0)
    assert sim.ground_truth().box_xyz[2] > start[2] + 0.05  # clear of the table, though it slips a little in the hug


def test_scripted_episode_puts_the_box_in_the_bin(sim: MujocoSim) -> None:
    policy = OraclePolicy(sim)
    for _ in range(20):
        call = policy.act(sim.observe(), "", "")
        if call is None:
            break
        assert sim.execute(call).ok
    verdict = validate(sim.ground_truth())
    assert verdict.accepted, verdict.reason


def test_out_of_reach_skills_are_refused_not_executed(sim: MujocoSim) -> None:
    before = sim.observe().joint_state
    result = sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 20}))
    assert not result.ok and result.error and not result.executed_actions
    assert sim.observe().joint_state == before
    assert not sim.ground_truth().joint_limit_violation


def test_a_recorded_episode_replays_exactly(sim: MujocoSim, tmp_path: Path) -> None:
    manifest = run_episode(
        sim,
        OraclePolicy(sim),
        run_dir=tmp_path,
        events=EventLog(tmp_path, "t"),
        seed=SEED,
        split="dev",
        round_=0,
        instruction="hug",
        perturb_at=1,
    )
    assert manifest.validation and manifest.validation.accepted, manifest.validation
    assert replay(sim, tmp_path / "episodes" / manifest.episode_id / "manifest.json")
