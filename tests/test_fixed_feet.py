# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""Workstream 1: the `fixed_feet` task (SPEC §3.1, §6 Gate A, rule 7), where the G1 stands and twists at the waist.

The physics tests run everywhere with a blank camera; only the tests that look at frames need a GL context.
"""

import io
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from isaacs_mom.contracts import Observation, Perturbation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.preview import OraclePolicy
from isaacs_mom.sim.mujoco.scene import bin_pose
from isaacs_mom.sim.mujoco.task import TASKS
from isaacs_mom.validator import replay, validate

SIM = MujocoSim(TASKS["fixed_feet"])
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
    assert blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 12})).ok
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": 1}))  # 13 cm: IK reaches it
    assert not result.ok and result.error and "reach limit" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


@pytest.mark.parametrize(
    ("skill", "target", "unreachable"),
    [("reach_left", "BOX_X", 0.8), ("hug", "_hug_z", 2.0), ("lift", "_lift_m", 1.0)],
)
def test_a_skill_whose_target_is_out_of_reach_changes_nothing(
    blind_sim: MujocoSim, monkeypatch: pytest.MonkeyPatch, skill: str, target: str, unreachable: float
) -> None:
    start = blind_sim.ground_truth().box_xyz
    if skill != "reach_left":
        blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    if skill == "lift":
        blind_sim.execute(SkillCall(skill="hug"))
    if target.startswith("_"):  # set from the task, so it lives on the sim
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
    _wait(sim, 1.6)
    x, y, z = sim.ground_truth().box_xyz
    _put_box(sim, (x, y, z + 0.005))  # lands at 0.2 m/s and is still within 0.2 s
    _wait(sim, 0.6)
    assert not sim.ground_truth().box_at_rest
    _wait(sim, 0.8)
    assert sim.ground_truth().box_at_rest


def test_a_box_turning_in_place_is_not_at_rest(sim: MujocoSim) -> None:
    x, y, z = sim.ground_truth().box_xyz
    sim.execute(SkillCall(skill="reach_left", params={"distance_cm": y * 100}))  # open hands clear either side
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


@pytest.mark.parametrize("depth", [0.035, 0.045, 0.05])  # read after a tick, at ~75% of the depth
def test_a_box_pushed_into_the_table_is_flagged(sim: MujocoSim, depth: float) -> None:
    start = sim.ground_truth().box_xyz
    _put_box(sim, (start[0], start[1], start[2] - depth))
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


def test_an_arm_forced_past_its_joint_range_is_flagged(sim: MujocoSim) -> None:
    joint = sim.model.joint("left_shoulder_roll_joint").id
    sim.data.qpos[sim.model.jnt_qposadr[joint]] = sim.model.jnt_range[joint][0] - 0.3  # a knock 0.3 rad past it
    sim.execute(SkillCall(skill="reobserve"))
    gt = sim.ground_truth()
    assert gt.joint_limit_violation
    assert "joint limit violated" in validate(gt).reason.split("; ")


def _state(sim: MujocoSim) -> tuple[object, ...]:
    return sim.work_y, sim.waist, sim.closed, sim.lifted, sim.data.time, sim.data.ctrl.tolist()


def _hug_and_lift(sim: MujocoSim) -> None:
    box_y = sim.ground_truth().box_xyz[1]
    for call in [
        SkillCall(skill="reach_left", params={"distance_cm": box_y * 100}),
        SkillCall(skill="hug"),
        SkillCall(skill="lift"),
    ]:
        assert sim.execute(call).ok, call


@pytest.mark.parametrize("skill", ["turn_left", "turn_right"])
def test_a_twist_that_would_sweep_the_arms_into_the_box_is_refused_and_changes_nothing(
    blind_sim: MujocoSim, skill: str
) -> None:
    before, qpos, qvel = _state(blind_sim), blind_sim.data.qpos.tobytes(), blind_sim.data.qvel.tobytes()
    result = blind_sim.execute(SkillCall.model_validate({"skill": skill, "params": {"angle_deg": 30}}))
    assert not result.ok and result.error and "sweep into the box" in result.error and not result.executed_actions
    assert (_state(blind_sim), blind_sim.data.qpos.tobytes(), blind_sim.data.qvel.tobytes()) == (before, qpos, qvel)


def test_a_turn_past_the_limit_is_refused_and_changes_nothing(blind_sim: MujocoSim) -> None:
    start = blind_sim.ground_truth().box_xyz
    blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    blind_sim.execute(SkillCall(skill="hug"))
    blind_sim.execute(SkillCall(skill="lift"))
    for _ in range(3):
        assert blind_sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 45})).ok
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 45}))
    assert not result.ok and result.error and "twist limit" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


@pytest.mark.parametrize(
    ("waypoint", "unreachable"),
    [
        ("_place_x", 0.8),
        ("_place_z", 0.4),
        ("_release_half", 0.5),
    ],  # the first, second, then only the last out of reach
)
def test_a_place_with_any_waypoint_out_of_reach_changes_nothing(
    blind_sim: MujocoSim, monkeypatch: pytest.MonkeyPatch, waypoint: str, unreachable: float
) -> None:
    start = blind_sim.ground_truth().box_xyz
    blind_sim.execute(SkillCall(skill="reach_left", params={"distance_cm": start[1] * 100}))
    blind_sim.execute(SkillCall(skill="hug"))
    blind_sim.execute(SkillCall(skill="lift"))
    monkeypatch.setattr(blind_sim, waypoint, unreachable)  # set from the task, so it lives on the sim
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
    _wait(sim, 1.6)
    assert sim.perturb(Perturbation(kind="move_box", seed=99))
    assert not sim.ground_truth().box_at_rest
    _wait(sim, 1.2)
    assert sim.ground_truth().box_at_rest


def test_a_placed_box_stays_in_the_bin_and_the_refused_perturbation_is_not_recorded(
    blind_sim: MujocoSim, tmp_path: Path
) -> None:
    events = EventLog(tmp_path, "t")
    manifest = run_episode(
        blind_sim,
        OraclePolicy(blind_sim),
        run_dir=tmp_path,
        events=events,
        seed=3,
        split="dev",
        round_=0,
        instruction="hug",
        perturb_at=7,  # seed 3 places on step 6, so the box is in the bin by then
    )
    assert manifest.validation and manifest.validation.accepted, manifest.validation
    assert manifest.perturbations == []
    logged = [json.loads(line) for line in events.path.read_text().splitlines()]
    assert not [e for e in logged if e["type"] == "perturbation"]
    assert [e["data"]["step"] for e in logged if e["type"] == "error"] == [7]
    assert replay(blind_sim, tmp_path / "episodes" / manifest.episode_id / "manifest.json")


def test_moving_the_bin_is_refused_only_where_it_would_land_on_the_box(sim: MujocoSim) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.375)
    _put_box(sim, (x, y, 0.13), yaw)  # on the floor, 1 cm outside the bin's far wall
    _wait(sim, 1.0)
    before = sim.ground_truth()
    assert not before.penetration
    assert not sim.perturb(Perturbation(kind="move_bin", seed=0))  # a pose that covers the box
    assert sim.ground_truth() == before
    assert sim.perturb(Perturbation(kind="move_bin", seed=2))  # a pose clear of it
    (bin_x, bin_y), bin_yaw = bin_pose(np.random.default_rng(2), sim.task)  # around where the feet stand
    after = sim.ground_truth()
    assert after.bin_xy == pytest.approx((bin_x, bin_y)) and after.bin_yaw == pytest.approx(bin_yaw)
    assert after.box_xyz == pytest.approx(before.box_xyz, abs=1e-6) and not after.penetration


def test_moving_the_bin_is_refused_while_the_box_is_in_it(sim: MujocoSim) -> None:
    x, y, yaw = _in_bin_frame(sim, 0.0)
    _put_box(sim, (x, y, 0.45), yaw)
    _wait(sim, 1.6)
    before = sim.ground_truth()
    assert before.box_on_bin_floor
    assert not sim.perturb(Perturbation(kind="move_bin", seed=99))
    assert sim.ground_truth() == before


def test_a_box_landing_on_a_corner_is_contact_not_penetration(blind_sim: MujocoSim, tmp_path: Path) -> None:
    # Seed 23 with the box moved first drops it corner-first, 2.1 cm into the bin floor for one step.
    manifest = run_episode(
        blind_sim,
        OraclePolicy(blind_sim),
        run_dir=tmp_path,
        events=EventLog(tmp_path, "t"),
        seed=23,
        split="dev",
        round_=0,
        instruction="hug",
        perturb_at=1,
    )
    assert manifest.validation and manifest.validation.accepted, manifest.validation


def test_a_box_with_an_open_hand_against_it_still_moves(sim: MujocoSim) -> None:
    box, pads = sim.model.geom("box").id, {sim.model.geom(f"{s}_hand_pad").id for s in ("left", "right")}
    start = sim.ground_truth().box_xyz

    def touching() -> set[int]:
        return {g for c in sim.data.contact[: sim.data.ncon] for g in (c.geom1, c.geom2) if box in (c.geom1, c.geom2)}

    for y in np.arange(start[1], 0.3, 0.005):  # slide it left until the open left pad touches it
        _put_box(sim, (start[0], float(y), start[2]))
        mujoco.mj_forward(sim.model, sim.data)
        if touching() & pads:
            break
    assert touching() & pads and not sim.closed
    before = sim.ground_truth().box_xyz
    assert sim.perturb(Perturbation(kind="move_box", seed=99))
    assert sim.ground_truth().box_xyz[:2] != pytest.approx(before[:2], abs=0.01)


def test_a_moved_box_never_lands_in_the_robots_arms(sim: MujocoSim) -> None:
    box_geom = sim.model.geom("box").id
    pelvis = sim.model.body("pelvis").id
    robot = {g for g in range(sim.model.ngeom) if sim.model.body_rootid[sim.model.geom_bodyid[g]] == pelvis}
    for reach in ("reach_left", "reach_right"):  # an open arm swung in over the table
        for seed in range(8):
            sim.reset(SEED)
            sim.execute(SkillCall(skill=reach, params={"distance_cm": 10}))
            if not sim.perturb(Perturbation(kind="move_box", seed=seed)):
                continue
            pairs = [(c.geom1, c.geom2) for c in sim.data.contact[: sim.data.ncon]]
            touching = {g for pair in pairs if box_geom in pair for g in pair}
            assert not touching & robot, f"{reach} move_box seed {seed}"


def test_moving_the_bin_onto_the_robot_is_refused(sim: MujocoSim, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("isaacs_mom.sim.mujoco.sim.bin_pose", lambda *_, **__: ((0.0, 0.0), 0.0))  # where it stands
    before = sim.ground_truth()
    assert not sim.perturb(Perturbation(kind="move_bin", seed=0))
    assert sim.ground_truth() == before


@pytest.mark.parametrize("skill", ["walk_forward", "walk_back"])
def test_walking_is_refused_with_fixed_feet_and_changes_nothing(blind_sim: MujocoSim, skill: str) -> None:
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall.model_validate({"skill": skill, "params": {"distance_cm": 30}}))
    assert not result.ok and result.error and "feet stay put" in result.error and not result.executed_actions
    assert _state(blind_sim) == before


def test_a_twist_past_45_degrees_at_once_is_refused_and_changes_nothing(blind_sim: MujocoSim) -> None:
    _hug_and_lift(blind_sim)
    before = _state(blind_sim)
    result = blind_sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 46}))
    assert not result.ok and result.error and "45 deg" in result.error and not result.executed_actions
    assert _state(blind_sim) == before
    assert blind_sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 45})).ok


def test_the_fixed_feet_task_names_itself_and_keeps_its_scene(sim: MujocoSim) -> None:
    assert sim.sim_version.endswith("task=fixed_feet")
    assert sim.robot_config == "unitree_g1_29dof_free_standing"
    assert sim.walker is None
    assert sim._hand_point() == pytest.approx((0.34, 0.0, 0.90))  # today's ready pose for the 30 cm box
    assert sim._hand_point(closed=True, lifted=True)[2] == pytest.approx(0.97)  # hug at 0.85, lift 12 cm


def test_the_twisted_drop_aims_on_the_leaning_pose(blind_sim: MujocoSim) -> None:
    # Seed 27 twists 117 degrees with the box held out, and the robot leans ~10 degrees: aimed as if upright, the hands
    # fall 1-2 cm short of the drop and `place` is refused.
    blind_sim.reset(27)
    policy, feedback = OraclePolicy(blind_sim), "start"
    for _ in range(20):
        call = policy.act(blind_sim.observe(), "", feedback)
        if call is None:
            break
        result = blind_sim.execute(call)
        assert result.ok, result.error
        feedback = f"{call.skill}: ok"
    verdict = validate(blind_sim.ground_truth())
    assert verdict.accepted, verdict.reason


def test_ground_truth_heading_follows_the_waist_twist(sim: MujocoSim) -> None:
    _hug_and_lift(sim)  # a 30 degree twist with the box still on the table would sweep the arms into it
    before = sim.ground_truth()
    assert before.robot_yaw == pytest.approx(0.0, abs=np.deg2rad(3))
    sim.execute(SkillCall(skill="turn_left", params={"angle_deg": 30}))
    after = sim.ground_truth()
    assert after.robot_yaw - before.robot_yaw == pytest.approx(np.deg2rad(30), abs=np.deg2rad(3))
    assert np.hypot(after.robot_xy[0] - before.robot_xy[0], after.robot_xy[1] - before.robot_xy[1]) < 0.02
