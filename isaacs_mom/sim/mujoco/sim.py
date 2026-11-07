# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""MujocoSim: the TruthSim over the walking G1 scene (SPEC §3.1, §4.1).

The legs run Unitree's walking policy for the whole episode, holding the robot's floor pose while the arms work. The
arms work around a point in front of the torso: `reach_*` slides it sideways, `hug` closes the hands on it, `lift`
raises them and `place` lowers and lets go. `walk_*` and `turn_*` move the floor pose. A skill that would leave the
arms' reach or walk the robot into the table or the bin is refused and reported, never executed (rule 7).
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
    TABLE_CENTER,
    TABLE_HALF,
    TABLE_TOP_Z,
    THIRD_SIZE,
    bin_pose,
    box_xy,
    build_model,
    layout,
    stand_qpos,
)
from isaacs_mom.sim.mujoco.walk import Walker, wrap

READY_Z = 0.90  # open hands wait beside the box, below its top
HUG_Z = 0.85
LIFT_DZ = 0.07  # clears the table; any higher and the box hides the bin from the head camera
OPEN_HALF = 0.30  # hand offset from the working point when open: clear of a box up to 12 cm off-centre
CLOSED_HALF = 0.135  # squeezes the 30 cm box by 1.5 cm a side
RELEASE_HALF = 0.22  # 4 cm clear of the box, and still within reach with the arms stretched out
PLACE_X = 0.40  # the hands push the box this far ahead of the pelvis before letting go
PLACE_Z = 0.84  # the box drops ~40 cm to the bin floor from here
REACH_LIMIT = 0.12
DEFAULT_REACH_CM = 5.0
DEFAULT_WALK_CM = 30.0
DEFAULT_TURN_DEG = 30.0
WALK_SPEED = 0.3  # m/s
TURN_RATE = 0.7  # rad/s
ARRIVED = 0.03  # m, and ARRIVED_YAW rad, from the goal pose
ARRIVED_YAW = np.deg2rad(3)
WALK_SETTLE_S = 0.6
TABLE_KEEP_OUT = 0.15  # the pelvis stays this far from the table top's edge
BIN_KEEP_OUT = PLACE_X  # and this far from the bin's centre, from where the box lands in its middle
SETTLE_S = 3.0  # a box that lands on a wall can tumble for ~1.5 s, then the 1 s at-rest window
FALLEN_Z = 0.6
PENETRATION_TOL = 0.02  # a dropped box dips ~1 cm into the bin floor for a step; tunnelling goes far deeper
AT_REST_SPEED = 0.02  # m/s
AT_REST_WINDOW = 1.0  # s of sim time
JOINT_LIMIT_TOL = 0.02  # rad


def _jpeg(pixels: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="JPEG", quality=85)
    return out.getvalue()


def _clearance(xy: np.ndarray, centre: np.ndarray, half: np.ndarray, yaw: float) -> float:
    """Floor distance from `xy` to a rectangle (0 inside it)."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = xy - centre
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


class MujocoSim:
    sim_version = f"mujoco-{mujoco.__version__}"
    robot_config = "unitree_g1_29dof_walking"

    def __init__(self) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self._stand = stand_qpos(self.model)
        self.robot = Robot(self.model, self._stand)
        self.walker = Walker(self.model)
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
        d.qpos[:] = self._stand
        scene = layout(seed)
        self._put_box(scene.box_xy)
        self._put_bin(scene.bin_xy, scene.bin_yaw)
        d.ctrl[:] = d.qpos[self.robot.actuated_qadr]
        mujoco.mj_forward(m, d)
        self.walker.reset()
        self.goal_xy, self.goal_yaw = np.zeros(2), 0.0
        self.work_y = 0.0
        self.closed = False
        self.lifted = False
        self._grip: dict[str, np.ndarray] | None = None
        self._speeds: deque[tuple[float, float]] = deque()
        self._fallen = self._limit_violated = self._penetrated = False
        self._actions: list[list[float]] = []
        self._times: list[float] = []
        self._hold(0.5)
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
        return GroundTruth(
            box_xyz=tuple(float(v) for v in d.xpos[self._box]),  # type: ignore[arg-type]
            bin_xy=(float(d.mocap_pos[self._bin][0]), float(d.mocap_pos[self._bin][1])),
            bin_yaw=self._bin_yaw(),
            bin_inner_half=BIN_INNER_HALF,
            box_on_bin_floor=any({c.geom1, c.geom2} == {self._box_geom, self._bin_floor} for c in d.contact[: d.ncon]),
            box_at_rest=settled_long_enough and max(recent, default=0.0) < AT_REST_SPEED,
            robot_fallen=self._fallen,
            joint_limit_violation=self._limit_violated,
            penetration=self._penetrated,
        )

    def perturb(self, perturbation: Perturbation) -> None:
        rng = np.random.default_rng(perturbation.seed)
        if perturbation.kind == "move_box":
            self._put_box(box_xy(rng))
        else:
            self._put_bin(*bin_pose(rng, around=self.pose()[:2]))
        mujoco.mj_forward(self.model, self.data)

    def render_third_person(self) -> bytes:
        self._third.update_scene(self.data, camera="third_person")
        return _jpeg(self._third.render())

    def pose(self) -> tuple[float, float, float]:
        """The robot's floor pose (x, y, heading), for previews and tests; the agents never see it."""
        return float(self.data.qpos[0]), float(self.data.qpos[1]), self.walker.yaw(self.data)

    def _run(self, call: SkillCall) -> str | None:
        half = CLOSED_HALF if self.closed else OPEN_HALF
        match call.skill:
            case "reobserve":
                self._hold(0.2)
            case "reach_left" | "reach_right":
                sign = 1.0 if call.skill == "reach_left" else -1.0
                work_y = self.work_y + sign * call.params.get("distance_cm", DEFAULT_REACH_CM) / 100
                if abs(work_y) > REACH_LIMIT + 1e-9:
                    return f"{call.skill}: past the {REACH_LIMIT * 100:.0f} cm reach limit"
                error = self._move_hands(self._hand_point(work_y=work_y), half, 0.6)
                if error is None:
                    self.work_y = work_y
                return error
            case "walk_forward" | "walk_back":
                sign = 1.0 if call.skill == "walk_forward" else -1.0
                step = sign * call.params.get("distance_cm", DEFAULT_WALK_CM) / 100
                heading = np.array([np.cos(self.goal_yaw), np.sin(self.goal_yaw)])
                goal = self.goal_xy + step * heading
                error = self._blocked(goal)
                if error:
                    return f"{call.skill}: {error}"
                self._walk_to(goal, self.goal_yaw, abs(step) / WALK_SPEED)
            case "turn_left" | "turn_right":
                sign = 1.0 if call.skill == "turn_left" else -1.0
                angle = np.deg2rad(call.params.get("angle_deg", DEFAULT_TURN_DEG))
                self._walk_to(self.goal_xy, wrap(self.goal_yaw + sign * angle), angle / TURN_RATE)
            case "hug":
                plan, error = self._plan_hands(self._hand_point(closed=True), CLOSED_HALF, self.data.ctrl, level=False)
                if error:
                    return error
                goal, self._grip = plan
                move(self.model, self.data, goal, 0.8, self._tick, self._drive)
                self.closed = True
            case "lift":
                error = self._move_hands(self._hand_point(lifted=True), half, 1.0)
                if error is None:
                    self.lifted = True
                return error
            case "place":
                # Out over the bin with the box level, then down as far as the arms go, so it drops in gently.
                height = self._hand_point()[2]
                error = self._move_through(
                    [
                        ((PLACE_X, self.work_y, height), half, 0.6, True),
                        ((PLACE_X, self.work_y, PLACE_Z), half, 0.6, False),
                        ((PLACE_X - 0.03, self.work_y, PLACE_Z + 0.04), RELEASE_HALF, 0.5, False),
                    ]
                )
                if error:
                    return error
                self.closed = self.lifted = False
                self._grip = None
                self._hold(SETTLE_S)
        return None

    def _hand_point(
        self, *, work_y: float | None = None, closed: bool | None = None, lifted: bool | None = None
    ) -> tuple[float, float, float]:
        """Where the hands go for the current state, or for the state a skill is about to commit."""
        closed = self.closed if closed is None else closed
        lifted = self.lifted if lifted is None else lifted
        z = (HUG_Z if closed else READY_Z) + (LIFT_DZ if lifted else 0.0)
        return BOX_X, self.work_y if work_y is None else work_y, z

    def _plan_hands(
        self, point: tuple[float, float, float], half: float, ctrl: np.ndarray, level: bool
    ) -> tuple[tuple[np.ndarray, dict[str, np.ndarray]], str | None]:
        """Targets, from `ctrl`, putting both hands at `point` ± `half` sideways in the robot's frame, and the hands'
        orientations there. With `level`, hands holding the box keep their grip orientation, so it doesn't tip."""
        goal, grip = ctrl.copy(), {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            target = np.array([point[0], point[1] + sign * half, point[2]])
            rotation = self._grip[side] if level and self._grip else None
            q, miss, grip[side] = self.robot.solve_arm(goal, side, target, rotation)
            if miss > IK_TOLERANCE:
                return (ctrl, grip), f"{side} hand can't reach there ({miss * 100:.0f} cm short)"
            goal[self.robot.arm_act[side]] = q
        return (goal, grip), None

    def _move_hands(self, point: tuple[float, float, float], half: float, duration: float) -> str | None:
        return self._move_through([(point, half, duration, True)])

    def _move_through(self, waypoints: list[tuple[tuple[float, float, float], float, float, bool]]) -> str | None:
        """Plan every (point, half, duration, level) waypoint before moving, so a refusal leaves the robot as it was."""
        goals, ctrl = [], self.data.ctrl
        for point, half, _, level in waypoints:
            (ctrl, _), error = self._plan_hands(point, half, ctrl, level)
            if error:
                return error
            goals.append(ctrl)
        for goal, (_, _, duration, _) in zip(goals, waypoints, strict=True):
            move(self.model, self.data, goal, duration, self._tick, self._drive)
        return None

    def _blocked(self, goal: np.ndarray) -> str | None:
        """Why walking straight from the current goal to `goal` would hit the table or the bin, if it would."""
        bin_xy = self.data.mocap_pos[self._bin][:2]
        gaps = {
            "the table": lambda p: _clearance(p, np.array(TABLE_CENTER), np.array(TABLE_HALF), 0.0) - TABLE_KEEP_OUT,
            "the bin": lambda p: float(np.linalg.norm(p - bin_xy)) - BIN_KEEP_OUT,
        }
        for name, gap in gaps.items():
            start = gap(self.goal_xy)
            for a in np.linspace(0.0, 1.0, 21)[1:]:
                g = gap(self.goal_xy + a * (goal - self.goal_xy))
                if g < 0 and g < start - 1e-6:
                    return f"would walk into {name}"
        return None

    def _walk_to(self, xy: np.ndarray, yaw: float, expected_s: float) -> None:
        self.goal_xy, self.goal_yaw = xy, yaw
        deadline = self.data.time + expected_s + 2.0
        while self.data.time < deadline:
            x, y, heading = self.pose()
            if np.hypot(x - xy[0], y - xy[1]) < ARRIVED and abs(wrap(yaw - heading)) < ARRIVED_YAW:
                break
            self._hold(0.1)
        self._hold(WALK_SETTLE_S)

    def _hold(self, duration: float) -> None:
        move(self.model, self.data, self.data.ctrl.copy(), duration, self._tick, self._drive)

    def _drive(self) -> None:
        command = self.walker.command_to(self.data, self.goal_xy, self.goal_yaw, WALK_SPEED, TURN_RATE)
        self.walker.act(self.data, command)

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

    def _bin_yaw(self) -> float:
        return 2 * float(np.arctan2(self.data.mocap_quat[self._bin][3], self.data.mocap_quat[self._bin][0]))

    def _put_box(self, xy: tuple[float, float]) -> None:
        a = self._box_qadr
        self.data.qpos[a : a + 7] = [*xy, TABLE_TOP_Z + BOX_HALF[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._box_dadr : self._box_dadr + 6] = 0.0

    def _put_bin(self, xy: tuple[float, float], yaw: float) -> None:
        self.data.mocap_pos[self._bin] = [*xy, 0.0]
        self.data.mocap_quat[self._bin] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
