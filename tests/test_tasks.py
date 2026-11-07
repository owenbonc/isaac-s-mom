"""Configurable tasks (SPEC §3.1): presets lay out within their ranges, bad configs are refused, and the task is part
of what a recording replays against."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from isaacs_mom.contracts import Observation
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.preview import OraclePolicy
from isaacs_mom.sim.mujoco.scene import (
    BACKED_OFF_CLEARANCE,
    BACKED_OFF_XY,
    BIN_WALL,
    BOX_X,
    TABLE_CLEARANCE,
    layout,
    table_clearance,
)
from isaacs_mom.sim.mujoco.task import BIN_INNER_HALF, TASKS, TaskConfig
from isaacs_mom.validator import replay, validate


@pytest.mark.parametrize("name", sorted(TASKS))
def test_every_preset_lays_out_within_its_ranges(name: str) -> None:
    task = TASKS[name]
    for seed in range(50):
        scene = layout(seed, task)
        assert abs(scene.box_xy[0] - BOX_X) <= 0.02 + 1e-9
        assert abs(scene.box_xy[1]) <= task.box_offset_m + 1e-9
        distance = float(np.hypot(*scene.bin_xy))
        assert task.bin_distance_m[0] - 1e-9 <= distance <= task.bin_distance_m[1] + 1e-9
        bearing = np.degrees(np.arctan2(scene.bin_xy[1], scene.bin_xy[0])) % 360
        assert task.bin_bearing_deg[0] - 1e-6 <= bearing <= task.bin_bearing_deg[1] + 1e-6
        if task.walking:  # room to walk between the table and the bin, and to back off the table first
            assert table_clearance(scene.bin_xy) > BIN_INNER_HALF + BIN_WALL + TABLE_CLEARANCE
            assert (
                np.hypot(scene.bin_xy[0] - BACKED_OFF_XY[0], scene.bin_xy[1] - BACKED_OFF_XY[1]) > BACKED_OFF_CLEARANCE
            )


@pytest.mark.parametrize(
    ("changes", "complaint"),
    [
        ({"box_half": (0.125, 0.20, 0.125)}, "box half-width"),
        ({"box_half": (0.15, 0.17, 0.125)}, "diagonal"),
        ({"box_mass_kg": 5.0}, "box mass"),
        ({"box_offset_m": 0.2}, "box offset"),
        ({"bin_distance_m": (1.4, 1.0)}, "bin distance"),
        ({"bin_distance_m": (0.3, 0.6)}, "bin distance"),
        ({"bin_bearing_deg": (300.0, 60.0)}, "bin bearing"),
        ({"platform_top_m": 0.50}, "platform top"),  # the box lands still between the open hands
        ({"platform_top_m": 0.35}, "platform top"),
        ({"walking": False, "bin_distance_m": (0.43, 0.45), "platform_top_m": 0.45}, "only for walking"),
    ],
)
def test_a_task_the_robot_cannot_do_is_refused(changes: dict[str, Any], complaint: str) -> None:
    with pytest.raises(ValueError, match=complaint):
        TaskConfig("bad", **changes)


def test_a_task_whose_bin_cannot_be_placed_says_so_instead_of_hanging() -> None:
    # Straight behind the robot and close: always inside the room it needs to back off from the table.
    cramped = TaskConfig("cramped", bin_distance_m=(0.6, 0.6), bin_bearing_deg=(179.0, 181.0))
    with pytest.raises(ValueError, match="cramped"):
        layout(0, cramped)


def _blind(sim: MujocoSim, monkeypatch: pytest.MonkeyPatch) -> MujocoSim:
    def observe() -> Observation:
        return Observation(
            episode_id=sim.episode_id,
            step=sim.step,
            sim_time=float(sim.data.time),
            head_rgb_jpeg=b"",
            joint_state=[float(v) for v in sim.data.qpos[sim.robot.actuated_qadr]],
        )

    monkeypatch.setattr(sim, "observe", observe)
    return sim


@pytest.mark.parametrize("name", sorted(TASKS))
def test_the_scripted_policy_completes_every_preset(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _blind(MujocoSim(TASKS[name]), monkeypatch)
    sim.reset(1)
    assert sim.ground_truth().box_xyz[2] == pytest.approx(0.72 + TASKS[name].box_half[2], abs=0.01)
    policy = OraclePolicy(sim)
    feedback = "start"
    for _ in range(20):
        call = policy.act(sim.observe(), "", feedback)
        if call is None:
            break
        result = sim.execute(call)
        feedback = f"{call.skill}: {'ok' if result.ok else result.error}"
    verdict = validate(sim.ground_truth())
    assert verdict.accepted, verdict.reason


def test_a_recording_replays_only_on_the_task_it_was_recorded_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    near = _blind(MujocoSim(TASKS["near_bin"]), monkeypatch)
    manifest = run_episode(
        near,
        OraclePolicy(near),
        run_dir=tmp_path,
        events=EventLog(tmp_path, "t"),
        seed=1,
        split="dev",
        round_=0,
        instruction="hug",
    )
    assert "near_bin" in manifest.sim_version
    path = tmp_path / "episodes" / manifest.episode_id / "manifest.json"
    assert replay(near, path)
    assert not replay(MujocoSim(TASKS["across_room"]), path)
