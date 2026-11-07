# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""MujocoSim: the TruthSim over the free-standing G1 scene (SPEC §3.1, §4.1).

The arms work around a point in front of the torso. `reach_*` slides that point sideways, `hug` closes the hands on
it, `lift` raises them, `turn_*` twists the waist, and `place` reaches out, lowers and lets go. A skill that would
leave the reachable range is refused and reported, never executed (rule 7).
"""

import io
import uuid
from collections import deque
from collections.abc import Callable

import mujoco
import numpy as np
from PIL import Image

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall
from isaacs_mom.sim.mujoco.control import IK_TOLERANCE, Robot, move
from isaacs_mom.sim.mujoco.scene import (
    BIN_INNER_HALF,
    BOX_HALF,
    BOX_X,
    HEAD_SIZE,
    TABLE_TOP_Z,
    THIRD_SIZE,
    bin_pose,
    box_xy,
    build_model,
    layout,
    stand_qpos,
)

READY_Z = 0.90  # open hands wait beside the box, below its top
HUG_Z = 0.85
LIFT_DZ = 0.12
OPEN_HALF = 0.30  # hand offset from the working point when open: clear of a box up to 12 cm off-centre
CLOSED_HALF = 0.135  # squeezes the 30 cm box by 1.5 cm a side
RELEASE_HALF = 0.22  # 4 cm clear of the box, and still within reach with the arms stretched out
PLACE_X = 0.38  # reach out over the bin centre before letting go
PLACE_Z = 0.86
REACH_LIMIT = 0.12
TURN_LIMIT = np.deg2rad(150)
DEFAULT_REACH_CM = 5.0
DEFAULT_TURN_DEG = 30.0
MAX_TURN_STEP_DEG = 45.0  # SkillCall clamps angle_deg to this
TURN_SPEED = np.deg2rad(60)  # rad/s
SETTLE_S = 2.5  # the drop and bounce take ~0.5 s, then the 1 s at-rest window
FALLEN_Z = 0.6
PENETRATION_TOL = 0.02  # a dropped box dips ~1 cm into the bin floor for a step; tunnelling goes far deeper
AT_REST_SPEED = 0.02  # m/s
AT_REST_WINDOW = 1.0  # s of sim time
JOINT_LIMIT_TOL = 0.02  # rad


def _jpeg(pixels: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="JPEG", quality=85)
    return out.getvalue()


class MujocoSim:
    sim_version = f"mujoco-{mujoco.__version__}"
    robot_config = "unitree_g1_29dof_free_standing"

    def __init__(self) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.robot = Robot(self.model)
        self._head = mujoco.Renderer(self.model, HEAD_SIZE[1], HEAD_SIZE[0])
        self._third = mujoco.Renderer(self.model, THIRD_SIZE[1], THIRD_SIZE[0])
        self._box = self.model.body("box").id
        self._box_qadr = self.model.jnt_qposadr[self.model.joint("box_free").id]
        self._box_dadr = self.model.jnt_dofadr[self.model.joint("box_free").id]
        self._bin = self.model.body("bin").mocapid[0]
        self._box_geom = self.model.geom("box").id
        self._bin_floor = self.model.geom("bin_floor").id
        self.on_tick: Callable[[], None] | None = None  # e.g. to film the motion for the demo video
        self._scenery = {
            g
            for g in range(self.model.ngeom)
            if self.model.body(self.model.geom_bodyid[g]).name in ("box", "table", "bin")
        }
        self._limited = [
            (self.model.jnt_qposadr[j], *self.model.jnt_range[j])
            for j in range(self.model.njnt)
            if self.model.jnt_limited[j] and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.reset(0)

    def reset(self, seed: int) -> str:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = stand_qpos(m)
        scene = layout(seed)
        self._put_box(scene.box_xy)
        self._put_bin(scene.bin_xy, scene.bin_yaw)
        d.ctrl[:] = d.qpos[self.robot.actuated_qadr]
        mujoco.mj_forward(m, d)
        self.work_y = 0.0
        self.waist = 0.0
        self.closed = False
        self.lifted = False
        self._speeds: deque[tuple[float, float]] = deque()
        self._fallen = self._limit_violated = self._penetrated = False
        self._actions: list[list[float]] = []
        self._times: list[float] = []
        move(m, d, d.ctrl.copy(), 0.5, self._tick)
        self._move_hands((0.18, self.work_y, 0.95), OPEN_HALF, 1.0)
        self._move_hands(self._hand_point(), OPEN_HALF, 0.8)
        self.episode_id = f"ep-{seed}-{uuid.uuid4().hex[:6]}"
        self.step = 0
        return self.episode_id

    def observe(self) -> Observation:
        self._head.update_scene(self.data, camera="head")
        return Observation(
            episode_id=self.episode_id,
            step=self.step,
            sim_time=float(self.data.time),
            head_rgb_jpeg=_jpeg(self._head.render()),
            joint_state=[float(v) for v in self.data.qpos[self.robot.actuated_qadr]],
        )

    def execute(self, call: SkillCall) -> ExecResult:
        self._actions, self._times = [], []
        error = self._run(call)
        self.step += 1
        return ExecResult(ok=error is None, error=error, executed_actions=self._actions, timestamps=self._times)

    def ground_truth(self) -> GroundTruth:
        d = self.data
        recent = [s for t, s in self._speeds if t >= d.time - AT_REST_WINDOW]
        settled_long_enough = bool(self._speeds) and self._speeds[0][0] <= d.time - AT_REST_WINDOW
        bin_yaw = 2 * float(np.arctan2(d.mocap_quat[self._bin][3], d.mocap_quat[self._bin][0]))
        return GroundTruth(
            box_xyz=tuple(float(v) for v in d.xpos[self._box]),  # type: ignore[arg-type]
            bin_xy=(float(d.mocap_pos[self._bin][0]), float(d.mocap_pos[self._bin][1])),
            bin_yaw=bin_yaw,
            bin_inner_half=BIN_INNER_HALF,
            box_on_bin_floor=any({c.geom1, c.geom2} == {self._box_geom, self._bin_floor} for c in d.contact[: d.ncon]),
            box_at_rest=settled_long_enough and max(recent) < AT_REST_SPEED,
            robot_fallen=self._fallen,
            joint_limit_violation=self._limit_violated,
            penetration=self._penetrated,
        )

    def perturb(self, perturbation: Perturbation) -> None:
        rng = np.random.default_rng(perturbation.seed)
        if perturbation.kind == "move_box":
            self._put_box(box_xy(rng))
        else:
            self._put_bin(*bin_pose(rng))
        mujoco.mj_forward(self.model, self.data)

    def render_third_person(self) -> bytes:
        self._third.update_scene(self.data, camera="third_person")
        return _jpeg(self._third.render())

    def _run(self, call: SkillCall) -> str | None:
        half = CLOSED_HALF if self.closed else OPEN_HALF
        match call.skill:
            case "reobserve":
                move(self.model, self.data, self.data.ctrl.copy(), 0.2, self._tick)
            case "reach_left" | "reach_right":
                sign = 1.0 if call.skill == "reach_left" else -1.0
                work_y = self.work_y + sign * call.params.get("distance_cm", DEFAULT_REACH_CM) / 100
                if abs(work_y) > REACH_LIMIT + 1e-9:
                    return f"{call.skill}: past the {REACH_LIMIT * 100:.0f} cm reach limit"
                previous, self.work_y = self.work_y, work_y
                error = self._move_hands(self._hand_point(), half, 0.6)
                if error:
                    self.work_y = previous
                return error
            case "turn_left" | "turn_right":
                sign = 1.0 if call.skill == "turn_left" else -1.0
                waist = self.waist + sign * np.deg2rad(call.params.get("angle_deg", DEFAULT_TURN_DEG))
                if abs(waist) > TURN_LIMIT:
                    return f"{call.skill}: past the {np.degrees(TURN_LIMIT):.0f} deg twist limit"
                goal = self.data.ctrl.copy()
                goal[self.robot.waist_yaw_act] = waist
                move(self.model, self.data, goal, max(0.4, abs(waist - self.waist) / TURN_SPEED), self._tick)
                self.waist = waist
            case "hug":
                self.closed = True
                return self._move_hands(self._hand_point(), CLOSED_HALF, 0.8)
            case "lift":
                self.lifted = True
                return self._move_hands(self._hand_point(), half, 1.0)
            case "place":
                height = self._hand_point()[2]
                goals, ctrl = [], self.data.ctrl.copy()
                for point, side_half in [
                    ((PLACE_X, self.work_y, height), half),
                    ((PLACE_X, self.work_y, PLACE_Z), half),
                    ((PLACE_X - 0.03, self.work_y, PLACE_Z + 0.04), RELEASE_HALF),
                ]:
                    ctrl, error = self._plan_hands(point, side_half, ctrl)
                    if error:
                        return error
                    goals.append(ctrl)
                for goal, duration in zip(goals, (0.6, 0.8, 0.5), strict=True):
                    move(self.model, self.data, goal, duration, self._tick)
                self.closed = self.lifted = False
                move(self.model, self.data, self.data.ctrl.copy(), SETTLE_S, self._tick)
        return None

    def _hand_point(self) -> tuple[float, float, float]:
        z = (HUG_Z if self.closed else READY_Z) + (LIFT_DZ if self.lifted else 0.0)
        return BOX_X, self.work_y, z

    def _plan_hands(
        self, point: tuple[float, float, float], half: float, ctrl: np.ndarray
    ) -> tuple[np.ndarray, str | None]:
        """Targets, from `ctrl`, putting both hands at `point` ± `half` sideways in the twisted torso's frame."""
        goal = ctrl.copy()
        for side, sign in (("left", 1.0), ("right", -1.0)):
            target = self.robot.to_world(self.data, np.array([point[0], point[1] + sign * half, point[2]]), self.waist)
            q, miss = self.robot.solve_arm(self.data, goal, side, target)
            if miss > IK_TOLERANCE:
                return ctrl, f"{side} hand can't reach there ({miss * 100:.0f} cm short)"
            goal[self.robot.arm_act[side]] = q
        return goal, None

    def _move_hands(self, point: tuple[float, float, float], half: float, duration: float) -> str | None:
        goal, error = self._plan_hands(point, half, self.data.ctrl.copy())
        if error is None:
            move(self.model, self.data, goal, duration, self._tick)
        return error

    def _tick(self) -> None:
        d = self.data
        self._actions.append([float(v) for v in d.ctrl])
        self._times.append(float(d.time))
        self._speeds.append((float(d.time), float(np.linalg.norm(d.qvel[self._box_dadr : self._box_dadr + 3]))))
        while self._speeds and self._speeds[0][0] < d.time - 2 * AT_REST_WINDOW:
            self._speeds.popleft()
        if self.on_tick:
            self.on_tick()
        self._fallen |= d.xpos[self.robot.pelvis][2] < FALLEN_Z
        self._limit_violated |= any(
            d.qpos[adr] < lo - JOINT_LIMIT_TOL or d.qpos[adr] > hi + JOINT_LIMIT_TOL for adr, lo, hi in self._limited
        )
        self._penetrated |= any(
            c.dist < -PENETRATION_TOL and (c.geom1 in self._scenery or c.geom2 in self._scenery)
            for c in d.contact[: d.ncon]
        )

    def _put_box(self, xy: tuple[float, float]) -> None:
        a = self._box_qadr
        self.data.qpos[a : a + 7] = [*xy, TABLE_TOP_Z + BOX_HALF[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._box_dadr : self._box_dadr + 6] = 0.0

    def _put_bin(self, xy: tuple[float, float], yaw: float) -> None:
        self.data.mocap_pos[self._bin] = [*xy, 0.0]
        self.data.mocap_quat[self._bin] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
