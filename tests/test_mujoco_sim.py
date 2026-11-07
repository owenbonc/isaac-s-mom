# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""Workstream 1: the real MuJoCo scene with the walking G1 (SPEC §6 Gate A, rule 7).

The physics tests run everywhere with a blank camera; only the tests that look at frames need a GL context.
"""

import io
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from isaacs_mom.contracts import Observation, Perturbation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.preview import OraclePolicy, run_scripted
from isaacs_mom.sim.mujoco.scene import bin_pose
from isaacs_mom.validator import replay, validate

SIM = MujocoSim()
SEED = 1  # a layout the scripted policy completes; the box starts 9 cm left of centre

try:
    mujoco.Renderer(SIM.model, 240, 320).close()
    NO_GL = ""
# No GL backend (ImportError), no CGL, EGL or OSMesa context (RuntimeError), or no usable one (FatalError).
except (ImportError, RuntimeError, mujoco.FatalError) as e:
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
    assert sim.perturb(Perturbation(kind="move_box", seed=99))
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


def test_a_reach_past_the_limit_is_refused_though_the_arms_could_make_it(blind_sim: MujocoSim) -> None:
    # Closed hands, so 12 cm and 13 cm are both within the arms' reach and only the limit can refuse.
    _put_box(blind_sim, (0.0, -1.0, 0.13))
    blind_sim.execute(SkillCall(skill="hug"))
    assert blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 12})).ok
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 1}))  # 13 cm: IK reaches it
    assert not result.ok and result.error and "reach limit" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


@pytest.mark.parametrize(
    ("skill", "target", "unreachable"),
    [("reach_left", "BOX_X", 0.8), ("hug", "_hug_z", 2.0), ("lift", "LIFT_DZ", 1.0)],
)
def test_a_skill_whose_target_is_out_of_reach_changes_nothing(
    blind_sim: MujocoSim, monkeypatch: pytest.MonkeyPatch, skill: str, target: str, unreachable: float
) -> None:
    start = blind_sim.ground_truth().box_xyz
    if skill != "reach_left":
        blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    if skill == "lift":
        blind_sim.execute(SkillCall(skill="hug"))
    if target.startswith("_"):  # the hug height follows the task's box, so it lives on the sim
        monkeypatch.setattr(blind_sim, target, unreachable)
    else:
        monkeypatch.setattr(f"isaacs_mom.sim.mujoco.sim.{target}", unreachable)
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall.model_validate({"skill": skill}))
    assert not result.ok and result.error and "can't reach" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


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


def test_a_box_set_down_from_5_mm_is_at_rest_only_a_full_second_later(sim: MujocoSim) -> None:
    # Line the open hands up with the box, so the stepping robot's forearms don't brush it.
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": sim.ground_truth().box_xyz[1] * 100}))
    _wait(sim, 1.6)
    x, y, z = sim.ground_truth().box_xyz
    _put_box(sim, (x, y, z + 0.005))  # lands at 0.2 m/s and is still within 0.2 s
    _wait(sim, 0.6)
    assert not sim.ground_truth().box_at_rest
    _wait(sim, 0.8)
    assert sim.ground_truth().box_at_rest


def test_a_box_turning_in_place_is_not_at_rest(sim: MujocoSim) -> None:
    # On the floor well clear of the robot: a spinning box's corners would reach the forearms at the table.
    x, y, z = 0.0, -1.0, 0.125
    _put_box(sim, (x, y, z + 0.002))
    dof = sim.model.jnt_dofadr[sim.model.joint("box_free").id]
    sim.data.qvel[dof + 5] = 1.0  # turning about its vertical axis, its centre still
    box = sim.model.body("box").id
    sim.data.xfrc_applied[box, 2] = sim.model.body_mass[box] * 9.81  # weightless, so friction can't stop it
    _wait(sim, 1.4)
    assert np.hypot(sim.ground_truth().box_xyz[0] - x, sim.ground_truth().box_xyz[1] - y) < 1e-3
    assert not sim.ground_truth().box_at_rest


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
    assert "penetration" in validate(gt).reason.split("; ")


def test_a_shoved_robot_is_flagged_as_fallen(sim: MujocoSim) -> None:
    sim.data.qvel[1] = 3.0  # a hard sideways shove on the pelvis
    _wait(sim, 2.0)
    gt = sim.ground_truth()
    assert gt.robot_fallen
    assert validate(gt).reason.split("; ")[0] == "robot fell"


def test_joints_count_as_past_their_range_only_beyond_the_tolerance(sim: MujocoSim) -> None:
    joint = sim.model.joint("left_shoulder_roll_joint").id
    adr, low = sim.model.jnt_qposadr[joint], sim.model.jnt_range[joint][0]
    assert sim.joints_past_limits() == []
    sim.data.qpos[adr] = low - 0.01  # inside the 0.02 rad tolerance
    assert sim.joints_past_limits() == []
    sim.data.qpos[adr] = low - 0.05
    assert sim.joints_past_limits() == ["left_shoulder_roll_joint"]


def test_an_arm_forced_past_its_joint_range_is_flagged(sim: MujocoSim) -> None:
    joint = sim.model.joint("left_shoulder_roll_joint").id
    sim.data.qpos[sim.model.jnt_qposadr[joint]] = sim.model.jnt_range[joint][0] - 0.3  # a knock 0.3 rad past it
    sim.execute(SkillCall(skill="reobserve"))
    gt = sim.ground_truth()
    assert gt.joint_limit_violation
    assert "joint limit violated" in validate(gt).reason.split("; ")


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
    assert "robot fell" in validate(gt).reason.split("; ")


def test_the_scripted_policy_finds_a_bin_moved_mid_carry(blind_sim: MujocoSim) -> None:
    verdict, _ = run_scripted(blind_sim, SEED, move_bin_at=5)
    assert verdict.accepted, verdict.reason


@pytest.mark.parametrize(
    ("waypoint", "unreachable"),
    [("PLACE_X", 0.8), ("PLACE_Z", 0.4), ("_release_half", 0.5)],  # the first, second, then only the last out of reach
)
def test_a_place_with_any_waypoint_out_of_reach_changes_nothing(
    blind_sim: MujocoSim, monkeypatch: pytest.MonkeyPatch, waypoint: str, unreachable: float
) -> None:
    start = blind_sim.ground_truth().box_xyz
    blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    blind_sim.execute(SkillCall(skill="hug"))
    blind_sim.execute(SkillCall(skill="lift"))
    if waypoint.startswith("_"):  # the release width follows the task's box, so it lives on the sim
        monkeypatch.setattr(blind_sim, waypoint, unreachable)
    else:
        monkeypatch.setattr(f"isaacs_mom.sim.mujoco.sim.{waypoint}", unreachable)
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="place"))
    assert not result.ok and result.error and "can't reach" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


def test_moving_the_box_is_refused_once_it_is_hugged(sim: MujocoSim) -> None:
    start = sim.ground_truth().box_xyz
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    sim.execute(SkillCall(skill="hug"))
    held = sim.ground_truth()
    assert not sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth() == held


def test_a_walk_that_never_arrives_reports_how_far_short(sim: MujocoSim, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim.walker, "command_to", lambda *args, **kwargs: np.zeros(3))  # legs that won't go anywhere
    sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30}))
    result = sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 50}))
    assert not result.ok and result.error and "short" in result.error
    assert result.executed_actions  # it did try: the stepping it executed is still recorded (rule 4)


def test_a_moved_box_never_lands_in_the_robots_arms(sim: MujocoSim) -> None:
    box_geom = sim.model.geom("box").id
    pelvis = sim.model.body("pelvis").id
    robot = {g for g in range(sim.model.ngeom) if sim.model.body_rootid[sim.model.geom_bodyid[g]] == pelvis}
    for reach in ("reach_left", "reach_right"):  # an arm swung in over the table, where a careless drop would hit it
        for seed in range(8):
            sim.reset(SEED)
            sim.execute(SkillCall(skill=reach, params={"distance_cm": 10}))
            if not sim.perturb(Perturbation(kind="move_box", seed=seed)):
                continue
            touching = {
                g
                for c in sim.data.contact[: sim.data.ncon]
                for g in (c.geom1, c.geom2)
                if box_geom in (c.geom1, c.geom2)
            }
            assert not touching & robot, f"{reach} move_box seed {seed}"


@pytest.mark.parametrize("height", [0.13, 0.45])  # on the floor, in the bin
def test_moving_the_box_is_refused_once_it_is_off_the_table(sim: MujocoSim, height: float) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.45 if height < 0.3 else 0.0)
    _put_box(sim, (x, y, height), yaw)
    _wait(sim, 1.0)
    before = sim.ground_truth()
    assert not sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth() == before


def test_a_box_out_of_the_closed_hands_still_moves(sim: MujocoSim) -> None:
    start = sim.ground_truth().box_xyz
    _put_box(sim, (0.0, -1.0, 0.13))  # out of the way while the hands close on nothing
    sim.execute(SkillCall(skill="hug"))
    for _ in range(2):
        sim.execute(SkillCall(skill="turn_right", params={"angle_deg": 45}))
    _put_box(sim, start)
    _wait(sim, 0.4)
    assert sim.closed and sim.ground_truth().box_xyz[2] > 0.8
    assert sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth().box_xyz[:2] != pytest.approx(start[:2], abs=0.01)


def test_a_moved_box_is_at_rest_only_a_full_second_later(sim: MujocoSim) -> None:
    def line_up() -> None:  # so the stepping robot's forearms don't brush the box while it waits
        dy = sim.ground_truth().box_xyz[1] - sim.work_y
        sim.execute(SkillCall(skill="reach_left" if dy > 0 else "reach_right", params={"distance_cm": abs(dy) * 100}))

    line_up()
    _wait(sim, 1.6)
    assert sim.ground_truth().box_at_rest
    assert sim.perturb(Perturbation(kind="move_box", seed=99))
    assert not sim.ground_truth().box_at_rest
    line_up()
    _wait(sim, 1.2)
    assert sim.ground_truth().box_at_rest


def test_a_placed_box_stays_in_the_bin_and_the_refused_perturbation_is_not_recorded(
    blind_sim: MujocoSim, tmp_path: Path
) -> None:
    events = EventLog(tmp_path, "t")
    common = {"run_dir": tmp_path, "events": events, "split": "dev", "round_": 0, "instruction": "hug"}
    steps = len(run_episode(blind_sim, OraclePolicy(blind_sim), seed=3, **common).steps)
    # Moved after the last step, the box is in the bin by then.
    manifest = run_episode(blind_sim, OraclePolicy(blind_sim), seed=3, perturb_at=steps, **common)
    assert manifest.validation and manifest.validation.accepted, manifest.validation
    assert manifest.perturbations == []
    logged = [json.loads(line) for line in events.path.read_text().splitlines()]
    mine = [e for e in logged if e["data"].get("episode_id") == manifest.episode_id]
    assert not [e for e in mine if e["type"] == "perturbation"]
    assert [e["data"]["step"] for e in mine if e["type"] == "error"] == [steps]
    assert replay(blind_sim, tmp_path / "episodes" / manifest.episode_id / "manifest.json")


def test_moving_the_bin_is_refused_only_where_it_would_land_on_the_box(sim: MujocoSim) -> None:
    def pose_for(seed: int) -> tuple[tuple[float, float], float]:
        return bin_pose(np.random.default_rng(seed), sim.task, around=sim.pose()[:2])

    (x, y), yaw = pose_for(0)
    _put_box(sim, (x, y, 0.13), yaw)  # on the floor exactly where seed 0 would put the bin
    _wait(sim, 1.0)
    before = sim.ground_truth()
    assert not before.penetration
    assert not sim.perturb(Perturbation(kind="move_bin", seed=0))
    assert sim.ground_truth() == before
    clear = next(s for s in range(1, 100) if math.dist(pose_for(s)[0], before.box_xyz[:2]) > 0.8)
    expected = pose_for(clear)
    assert sim.perturb(Perturbation(kind="move_bin", seed=clear))
    after = sim.ground_truth()
    assert after.bin_xy == pytest.approx(expected[0]) and after.bin_yaw == pytest.approx(expected[1])
    assert after.box_xyz == pytest.approx(before.box_xyz, abs=1e-6) and not after.penetration


def test_moving_the_bin_is_refused_while_the_box_is_in_it(sim: MujocoSim) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.0)
    _put_box(sim, (x, y, 0.45), yaw)
    _wait(sim, 1.6)
    before = sim.ground_truth()
    assert before.box_on_bin_floor
    assert not sim.perturb(Perturbation(kind="move_bin", seed=99))
    assert sim.ground_truth() == before


def test_ground_truth_follows_the_robots_floor_pose_and_heading(blind_sim: MujocoSim) -> None:
    start = blind_sim.ground_truth()
    assert start.robot_xy == pytest.approx((0.0, 0.0), abs=0.03) and start.robot_yaw == pytest.approx(0.0, abs=0.05)
    assert blind_sim.execute(SkillCall(skill="walk_back", params={"distance_cm": 30})).ok
    assert blind_sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 90})).ok
    moved = blind_sim.ground_truth()
    assert moved.robot_xy == pytest.approx((-0.30, 0.0), abs=0.05)
    assert math.degrees(moved.robot_yaw) == pytest.approx(90, abs=5)
