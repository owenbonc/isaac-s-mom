"""Workstream 1: the real MuJoCo scene with the walking G1 (SPEC §6 Gate A, rule 7).

The physics tests run everywhere with a blank camera; only the tests that look at frames need a GL context.
"""

import io
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from isaacs_mom.contracts import Observation, Perturbation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.preview import OraclePolicy, run_scripted
from isaacs_mom.validator import replay, validate

SIM = MujocoSim()
SEED = 1  # a layout the scripted policy completes; the box starts 9 cm left of centre

try:
    SIM.render_third_person()
    NO_GL = ""
except Exception as e:  # a missing GL context surfaces as several exception types
    NO_GL = f"MuJoCo cannot render here: {e}"
needs_gl = pytest.mark.skipif(bool(NO_GL), reason=NO_GL)


@pytest.fixture
def sim() -> MujocoSim:
    SIM.reset(SEED)
    return SIM


@pytest.fixture
def blind_sim(sim: MujocoSim, monkeypatch: pytest.MonkeyPatch) -> MujocoSim:
    """The sim with a blank head camera: everything else, physics included, is real."""

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


@needs_gl
def test_same_seed_same_frame_and_moving_the_box_changes_it(sim: MujocoSim) -> None:
    first = sim.observe().head_rgb_jpeg
    assert Image.open(io.BytesIO(first)).size == (320, 240)
    sim.reset(SEED)
    assert sim.observe().head_rgb_jpeg == first
    sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.observe().head_rgb_jpeg != first


def test_same_seed_same_scene(sim: MujocoSim) -> None:
    first = sim.ground_truth()
    sim.reset(SEED)
    assert sim.ground_truth() == first


def test_hug_and_lift_raise_the_box_with_physics(sim: MujocoSim) -> None:
    start = sim.ground_truth().box_xyz
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    sim.execute(SkillCall(skill="hug"))
    result = sim.execute(SkillCall(skill="lift"))
    assert result.ok and len(result.executed_actions) == len(result.timestamps) > 0
    assert np.all(np.diff(result.timestamps) > 0)
    assert sim.ground_truth().box_xyz[2] > start[2] + 0.05  # clear of the table, though it slips a little in the hug
    assert not sim.ground_truth().robot_fallen


def test_scripted_episode_puts_the_box_in_the_bin(blind_sim: MujocoSim) -> None:
    policy = OraclePolicy(blind_sim)
    for _ in range(20):
        call = policy.act(blind_sim.observe(), "", "")
        if call is None:
            break
        assert blind_sim.execute(call).ok
    verdict = validate(blind_sim.ground_truth())
    assert verdict.accepted, verdict.reason


def test_out_of_reach_skills_are_refused_not_executed(blind_sim: MujocoSim) -> None:
    before = blind_sim.observe().joint_state
    result = blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 20}))
    assert not result.ok and result.error and not result.executed_actions
    assert blind_sim.observe().joint_state == before
    assert not blind_sim.ground_truth().joint_limit_violation


def test_a_recorded_episode_replays_exactly(blind_sim: MujocoSim, tmp_path: Path) -> None:
    manifest = run_episode(
        blind_sim,
        OraclePolicy(blind_sim),
        run_dir=tmp_path,
        events=EventLog(tmp_path, "t"),
        seed=SEED,
        split="dev",
        round_=0,
        instruction="hug",
        perturb_at=1,
    )
    assert manifest.validation and manifest.validation.accepted, manifest.validation
    assert replay(blind_sim, tmp_path / "episodes" / manifest.episode_id / "manifest.json")


def _put_box(sim: MujocoSim, xyz: tuple[float, float, float], yaw: float = 0.0, tilt: float = 0.0) -> None:
    """Test-only: set the box down anywhere, turned by `yaw` and then tipped by `tilt` about its own y axis."""
    qyaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    qtilt = np.array([np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0])
    w1, x1, y1, z1 = qyaw
    w2, x2, y2, z2 = qtilt
    quat = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    adr = sim.model.jnt_qposadr[sim.model.joint("box_free").id]
    dof = sim.model.jnt_dofadr[sim.model.joint("box_free").id]
    sim.data.qpos[adr : adr + 7] = [*xyz, *quat]
    sim.data.qvel[dof : dof + 6] = 0.0


def _wait(sim: MujocoSim, seconds: float) -> None:
    for _ in range(round(seconds / 0.2)):
        sim.execute(SkillCall(skill="reobserve"))


def _in_bin_frame(sim: MujocoSim, local_x: float) -> tuple[float, float, float]:
    """World xy of a point `local_x` out from the bin's centre along its own x axis."""
    gt = sim.ground_truth()
    return gt.bin_xy[0] + local_x * np.cos(gt.bin_yaw), gt.bin_xy[1] + local_x * np.sin(gt.bin_yaw), gt.bin_yaw


def test_a_box_left_on_the_table_fails(sim: MujocoSim) -> None:
    box_y = sim.ground_truth().box_xyz[1]
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": box_y * 100}))  # lined up, never hugged
    _wait(sim, 1.6)
    gt = sim.ground_truth()
    assert gt.box_at_rest and not gt.box_on_bin_floor
    assert not validate(gt).accepted


def test_a_box_on_the_floor_beside_the_bin_fails(sim: MujocoSim) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.45)
    _put_box(sim, (x, y, 0.13), yaw)
    _wait(sim, 2.0)
    gt = sim.ground_truth()
    assert gt.box_at_rest and not gt.box_on_bin_floor
    assert not validate(gt).accepted


def test_a_box_leaning_on_the_bins_outside_fails(sim: MujocoSim) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.24 + 0.17)
    _put_box(sim, (x, y, 0.16), yaw, tilt=np.deg2rad(-15))
    _wait(sim, 2.0)
    gt = sim.ground_truth()
    assert np.hypot(gt.box_xyz[0] - gt.bin_xy[0], gt.box_xyz[1] - gt.bin_xy[1]) < 0.45  # still against the bin
    assert not gt.box_on_bin_floor
    assert not validate(gt).accepted


def test_a_box_dropped_in_the_bin_is_accepted_only_once_at_rest(sim: MujocoSim) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.0)
    _put_box(sim, (x, y, 0.75), yaw)
    sim.execute(SkillCall(skill="reobserve"))
    assert not sim.ground_truth().box_at_rest
    _wait(sim, 2.0)
    gt = sim.ground_truth()
    assert gt.box_on_bin_floor and gt.box_at_rest
    assert validate(gt).accepted


def test_a_box_pushed_into_the_table_is_flagged(sim: MujocoSim) -> None:
    start = sim.ground_truth().box_xyz
    _put_box(sim, (start[0], start[1], start[2] - 0.05))
    sim.execute(SkillCall(skill="reobserve"))
    gt = sim.ground_truth()
    assert gt.penetration
    assert validate(gt).reason == "objects penetrated each other"


def test_a_shoved_robot_is_flagged_as_fallen(sim: MujocoSim) -> None:
    sim.data.qvel[1] = 3.0  # a hard sideways shove on the pelvis
    _wait(sim, 2.0)
    gt = sim.ground_truth()
    assert gt.robot_fallen
    assert validate(gt).reason == "robot fell"


def _state(sim: MujocoSim) -> tuple[object, ...]:
    return (
        sim.work_y,
        sim.closed,
        sim.lifted,
        sim.goal_xy.tolist(),
        sim.goal_yaw,
        sim.pose(),
        sim.data.time,
        sim.data.ctrl.tolist(),
    )


def _carry(sim: MujocoSim) -> None:
    box_y = sim.ground_truth().box_xyz[1]
    for call in [
        SkillCall(skill="reach_left", params={"distance_cm": box_y * 100}),
        SkillCall(skill="hug"),
        SkillCall(skill="lift"),
        SkillCall(skill="walk_back", params={"distance_cm": 30}),
    ]:
        assert sim.execute(call).ok, call


def _face_the_bin(sim: MujocoSim) -> float:
    """Turn to the bin; its distance from the pelvis."""
    x, y, heading = sim.pose()
    bin_xy = sim.ground_truth().bin_xy
    turn = math.degrees(math.remainder(math.atan2(bin_xy[1] - y, bin_xy[0] - x) - heading, math.tau))
    assert sim.execute(SkillCall(skill="turn_left" if turn > 0 else "turn_right", params={"angle_deg": abs(turn)})).ok
    x, y, _ = sim.pose()
    return math.hypot(bin_xy[0] - x, bin_xy[1] - y)


def test_walking_into_the_bin_is_refused_and_changes_nothing(blind_sim: MujocoSim) -> None:
    _carry(blind_sim)
    dist = _face_the_bin(blind_sim)
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="walk_forward", params={"distance_cm": dist * 100}))
    assert not result.ok and result.error and "into the bin" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


def test_walking_into_the_table_is_refused_and_changes_nothing(blind_sim: MujocoSim) -> None:
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="walk_forward", params={"distance_cm": 30}))
    assert not result.ok and result.error and "into the table" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


def test_walk_forward_goes_the_requested_distance_on_its_own_feet(blind_sim: MujocoSim) -> None:
    blind_sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    blind_sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 90}))
    x, y, heading = blind_sim.pose()
    assert blind_sim.execute(SkillCall(skill="walk_forward", params={"distance_cm": 60})).ok
    x2, y2, heading2 = blind_sim.pose()
    assert math.hypot(x2 - x, y2 - y) == pytest.approx(0.60, abs=0.05)
    assert math.degrees(math.remainder(math.atan2(y2 - y, x2 - x) - heading, math.tau)) == pytest.approx(0, abs=5)
    assert abs(math.degrees(math.remainder(heading2 - heading, math.tau))) < 4
    assert not blind_sim.ground_truth().robot_fallen


def test_turning_turns_the_whole_body_in_place(blind_sim: MujocoSim) -> None:
    blind_sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    x, y, heading = blind_sim.pose()
    assert blind_sim.execute(SkillCall(skill="turn_right", params={"angle_deg": 135})).ok
    x2, y2, heading2 = blind_sim.pose()
    assert math.degrees(math.remainder(heading2 - heading, math.tau)) == pytest.approx(-135, abs=4)
    assert math.hypot(x2 - x, y2 - y) < 0.06


def test_a_hugged_box_survives_walking_and_turning(blind_sim: MujocoSim) -> None:
    _carry(blind_sim)
    assert blind_sim.execute(SkillCall(skill="turn_right", params={"angle_deg": 120})).ok
    assert blind_sim.execute(SkillCall(skill="walk_forward", params={"distance_cm": 80})).ok
    x, y, heading = blind_sim.pose()
    box = blind_sim.ground_truth().box_xyz
    ahead = math.cos(heading) * (box[0] - x) + math.sin(heading) * (box[1] - y)
    assert box[2] > 0.8 and 0.25 < ahead < 0.45  # still held up in front of the chest
    assert not blind_sim.ground_truth().robot_fallen


def test_a_robot_shoved_while_walking_is_flagged_as_fallen(blind_sim: MujocoSim) -> None:
    blind_sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    blind_sim.data.qvel[1] = 3.0  # a hard sideways shove on the pelvis
    blind_sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    _wait(blind_sim, 1.0)
    gt = blind_sim.ground_truth()
    assert gt.robot_fallen
    assert validate(gt).reason == "robot fell"


def test_the_scripted_policy_finds_a_bin_moved_mid_carry(blind_sim: MujocoSim) -> None:
    verdict, _ = run_scripted(blind_sim, SEED, move_bin_at=5)
    assert verdict.accepted, verdict.reason


def test_an_out_of_reach_place_is_refused_and_changes_nothing(
    blind_sim: MujocoSim, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = blind_sim.ground_truth().box_xyz
    blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    blind_sim.execute(SkillCall(skill="hug"))
    blind_sim.execute(SkillCall(skill="lift"))
    monkeypatch.setattr("isaacs_mom.sim.mujoco.sim.PLACE_X", 0.8)  # further than the G1's arms reach
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="place"))
    assert not result.ok and result.error and "can't reach" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


def test_moving_the_box_is_ignored_once_it_is_hugged(sim: MujocoSim) -> None:
    start = sim.ground_truth().box_xyz
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    sim.execute(SkillCall(skill="hug"))
    held = sim.ground_truth().box_xyz
    sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth().box_xyz == held
