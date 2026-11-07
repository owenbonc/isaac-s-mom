# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""MujocoSim: the TruthSim over the walking G1 scene (SPEC §3.1, §4.1).

The legs run Unitree's walking policy for the whole episode, holding the robot's floor pose while the arms work. The
arms work around a point in front of the torso: `reach_*` slides it sideways, `hug` closes the hands on it, `lift`
raises them and `place` centres the box, pushes it out, lowers and lets go. `walk_*` and `turn_*` move the floor
pose. A skill that would leave the arms' reach or walk the robot into the table or the bin is refused and reported,
never executed (rule 7).
"""

import copy
import io
import uuid
from collections import deque
from collections.abc import Callable

import mujoco
import numpy as np
from PIL import Image

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall
from isaacs_mom.sim.mujoco.control import IK_TOLERANCE_M, Robot, move
from isaacs_mom.sim.mujoco.scene import (
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
from isaacs_mom.sim.mujoco.task import BIN_INNER_HALF, DEFAULT_TASK, TaskConfig
from isaacs_mom.sim.mujoco.walk import Walker, wrap

# The hands work relative to the box: they hug at its mid-height, wait 5 cm above that, close 1.5 cm inside its sides,
# open 18 cm outside them (clear of a box 12 cm off-centre as the robot sways) and let go 7 cm outside them.
READY_ABOVE_HUG = 0.05
SQUEEZE = 0.015
OPEN_CLEARANCE = 0.18
RELEASE_CLEARANCE = 0.07
LIFT_DZ = 0.07  # clears the table; any higher and the box hides the bin from the head camera
PLACE_X = 0.40  # the hands push the box this far ahead of the pelvis before letting go
PLACE_Z = 0.84  # the box drops ~40 cm to the bin floor from here
REACH_LIMIT = 0.12
STEPPING_PELVIS_Z = 0.774  # the pelvis's mean height while the legs step in place
BOX_MOVE_TRIES = 20
DEFAULT_REACH_CM = 5.0
DEFAULT_WALK_CM = 30.0
DEFAULT_TURN_DEG = 30.0
WALK_SPEED_M_S = 0.3
TURN_RATE_RAD_S = 0.7
ARRIVED_M = 0.03  # from the goal pose
ARRIVED_YAW_RAD = np.deg2rad(3)
# The stepping gait sways a few centimetres about its stance; a walk this far off its goal didn't get there.
MISSED_M = 0.10
MISSED_YAW_RAD = np.deg2rad(10)
WALK_SETTLE_S = 0.6
TABLE_KEEP_OUT_M = 0.15  # the pelvis stays this far from the table top's edge
BIN_KEEP_OUT_M = 0.42  # and this far from the bin's centre, so the toes stay clear of its stand
SETTLE_STEP_S = 0.5
MAX_SETTLE_S = 6.0  # a box that lands leaning on a wall can take several seconds to tip flat
FALLEN_Z = 0.6
PENETRATION_TOL_M = 0.03  # a box landing on a corner dips up to ~2 cm into the bin floor for a step
AT_REST_SPEED_M_S = 0.02
AT_REST_SPIN_RAD_S = 0.2
AT_REST_WINDOW_S = 1.0
JOINT_LIMIT_TOL_RAD = 0.02


def _jpeg(pixels: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="JPEG", quality=85)
    return out.getvalue()


def _turn(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _clearance(xy: np.ndarray, centre: np.ndarray, half: np.ndarray, yaw: float) -> float:
    """Floor distance from `xy` to a rectangle (0 inside it)."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = xy - centre
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


class MujocoSim:
    robot_config = "unitree_g1_29dof_walking"

    def __init__(self, task: TaskConfig = DEFAULT_TASK) -> None:
        self.task = task
        # The task is part of the version, so a manifest only replays on the scene it was recorded in.
        self.sim_version = f"mujoco-{mujoco.__version__} task={task.name}"
        width = task.box_half[1]
        self._open_half, self._closed_half = width + OPEN_CLEARANCE, width - SQUEEZE
        self._release_half = width + RELEASE_CLEARANCE
        self._hug_z = TABLE_TOP_Z + task.box_half[2]
        self.model = build_model(task)
        self.data = mujoco.MjData(self.model)
        self._stand = stand_qpos(self.model)
        stepping = self._stand.copy()
        stepping[2] = STEPPING_PELVIS_Z
        self.robot = Robot(self.model, stepping)
        self.walker = Walker(self.model)
        # Made on first use, so the physics runs (and is tested) where no GL context exists.
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._box = self.model.body("box").id
        self._box_qadr = self.model.jnt_qposadr[self.model.joint("box_free").id]
        self._box_dadr = self.model.jnt_dofadr[self.model.joint("box_free").id]
        self._bin = self.model.body("bin").mocapid[0]
        self._box_geom = self.model.geom("box").id
        self._table_top = self.model.geom("table_top").id
        self._hand_pads = {self.model.geom(f"{side}_hand_pad").id for side in ("left", "right")}
        self._bin_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.model.body("bin").id}
        self._bin_floor = self.model.geom("bin_floor").id
        self.on_tick: Callable[[], None] | None = None  # e.g. to film the motion for the demo video
        pelvis = self.model.body("pelvis").id
        self._robot_geoms = {
            g for g in range(self.model.ngeom) if self.model.body_rootid[self.model.geom_bodyid[g]] == pelvis
        }
        self._scenery = {
            g
            for g in range(self.model.ngeom)
            if self.model.body(self.model.geom_bodyid[g]).name in ("box", "table", "bin")
        }
        self._limited = [
            (self.model.joint(j).name, self.model.jnt_qposadr[j], *self.model.jnt_range[j])
            for j in range(self.model.njnt)
            if self.model.jnt_limited[j] and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.reset(0)

    def reset(self, seed: int) -> str:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self._stand
        scene = layout(seed, self.task)
        self._put_box(scene.box_xy)
        self._put_bin(d, scene.bin_xy, scene.bin_yaw)
        d.ctrl[:] = d.qpos[self.robot.actuated_qadr]
        mujoco.mj_forward(m, d)
        self.walker.reset()
        self.goal_xy, self.goal_yaw = np.zeros(2), 0.0
        self.work_y = 0.0
        self.closed = False
        self.lifted = False
        self._grip: dict[str, np.ndarray] | None = None
        self._speeds: deque[tuple[float, float, float]] = deque()  # sim time, box speed, box spin
        self._fallen = self._limit_violated = self._penetrated = False
        self._actions: list[list[float]] = []
        self._times: list[float] = []
        self._hold(0.5)
        self._move_hands((0.18, self.work_y, 0.95), self._open_half, 1.0)
        self._move_hands(self._hand_point(), self._open_half, 0.8)
        self.episode_id = f"ep-{seed}-{uuid.uuid4().hex[:6]}"
        self.step = 0
        return self.episode_id

    def observe(self) -> Observation:
        return Observation(
            episode_id=self.episode_id,
            step=self.step,
            sim_time=float(self.data.time),
            head_rgb_jpeg=self._render("head", HEAD_SIZE),
            joint_state=[float(v) for v in self.data.qpos[self.robot.actuated_qadr]],
        )

    def execute(self, call: SkillCall) -> ExecResult:
        self._actions, self._times = [], []
        error = self._run(call)
        self.step += 1
        return ExecResult(ok=error is None, error=error, executed_actions=self._actions, timestamps=self._times)

    def ground_truth(self) -> GroundTruth:
        d = self.data
        return GroundTruth(
            box_xyz=(float(d.xpos[self._box][0]), float(d.xpos[self._box][1]), float(d.xpos[self._box][2])),
            bin_xy=(float(d.mocap_pos[self._bin][0]), float(d.mocap_pos[self._bin][1])),
            bin_yaw=self._bin_yaw(),
            bin_inner_half=BIN_INNER_HALF,
            box_on_bin_floor=any({c.geom1, c.geom2} == {self._box_geom, self._bin_floor} for c in d.contact[: d.ncon]),
            box_at_rest=self._box_at_rest(),
            robot_fallen=self._fallen,
            joint_limit_violation=self._limit_violated,
            penetration=self._penetrated,
        )

    def perturb(self, perturbation: Perturbation) -> bool:
        rng = np.random.default_rng(perturbation.seed)
        if perturbation.kind == "move_box":
            touching = self._touching(self.data, {self._box_geom})
            # A box in the hands, the bin or on the floor stays put: moving it would undo the robot's work.
            if self._table_top not in touching or touching & self._hand_pads:
                return False
            # Somewhere new on the table clear of the robot's arms; the same seed always picks the same spot.
            start = self.data.qpos[self._box_qadr : self._box_qadr + 7].copy()
            for _ in range(BOX_MOVE_TRIES):
                self._put_box(box_xy(rng, self.task))
                mujoco.mj_forward(self.model, self.data)
                if not self._touching(self.data, {self._box_geom}) & self._robot_geoms:
                    self._speeds.clear()  # it has not been seen at rest where it now is
                    return True
            self.data.qpos[self._box_qadr : self._box_qadr + 7] = start
            mujoco.mj_forward(self.model, self.data)
            return False
        if self._bin_floor in self._touching(self.data, {self._box_geom}):  # moving it would tip the box out
            return False
        xy, yaw = bin_pose(rng, self.task, around=self.pose()[:2])
        trial = copy.copy(self.data)
        self._put_bin(trial, xy, yaw)
        mujoco.mj_fwdPosition(self.model, trial)
        if self._touching(trial, self._bin_geoms):  # it would land on the box or the robot
            return False
        self._put_bin(self.data, xy, yaw)
        mujoco.mj_forward(self.model, self.data)
        return True

    def render_third_person(self) -> bytes:
        return self._render("third_person", THIRD_SIZE)

    def pose(self) -> tuple[float, float, float]:
        """The robot's floor pose (x, y, heading), for previews and tests; the agents never see it."""
        return float(self.data.qpos[0]), float(self.data.qpos[1]), self.walker.yaw(self.data)

    def _render(self, camera: str, size: tuple[int, int]) -> bytes:
        if camera not in self._renderers:
            self._renderers[camera] = mujoco.Renderer(self.model, size[1], size[0])
            # A fresh context's first frame differs slightly from every later one; spend it here, so frames repeat.
            self._renderers[camera].update_scene(self.data, camera=camera)
            self._renderers[camera].render()
        renderer = self._renderers[camera]
        renderer.update_scene(self.data, camera=camera)
        return _jpeg(renderer.render())

    def _run(self, call: SkillCall) -> str | None:
        half = self._closed_half if self.closed else self._open_half
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
                return self._walk_to(goal, self.goal_yaw, abs(step) / WALK_SPEED_M_S, call.skill)
            case "turn_left" | "turn_right":
                sign = 1.0 if call.skill == "turn_left" else -1.0
                angle = np.deg2rad(call.params.get("angle_deg", DEFAULT_TURN_DEG))
                return self._walk_to(
                    self.goal_xy, wrap(self.goal_yaw + sign * angle), angle / TURN_RATE_RAD_S, call.skill
                )
            case "hug":
                point = self._hand_point(closed=True)
                (goal, grip), error = self._plan_hands(point, self._closed_half, self.data.ctrl, level=False)
                if error:
                    return error
                move(self.model, self.data, goal, 0.8, self._tick, self._drive)
                self.closed, self._grip = True, grip
            case "lift":
                error = self._move_hands(self._hand_point(lifted=True), half, 1.0)
                if error is None:
                    self.lifted = True
                return error
            case "place":
                # Straight out ahead with the box level, then down as far as the arms go, so it drops in gently.
                height = self._hand_point()[2]
                error = self._move_through(
                    [
                        ((PLACE_X, 0.0, height), half, 0.6, True),
                        ((PLACE_X, 0.0, PLACE_Z), half, 0.6, False),
                        ((PLACE_X - 0.03, 0.0, PLACE_Z + 0.04), self._release_half, 0.5, False),
                    ]
                )
                if error:
                    return error
                self.closed = self.lifted = False
                self._grip = None
                self.work_y = 0.0
                waited = 0.0
                while waited < MAX_SETTLE_S and not self._box_at_rest():
                    self._hold(SETTLE_STEP_S)
                    waited += SETTLE_STEP_S
        return None

    def _hand_point(
        self, *, work_y: float | None = None, closed: bool | None = None, lifted: bool | None = None
    ) -> tuple[float, float, float]:
        """Where the hands go for the current state, or for the state a skill is about to commit."""
        closed = self.closed if closed is None else closed
        lifted = self.lifted if lifted is None else lifted
        z = self._hug_z + (0.0 if closed else READY_ABOVE_HUG) + (LIFT_DZ if lifted else 0.0)
        return BOX_X, self.work_y if work_y is None else work_y, z

    def _plan_hands(
        self, point: tuple[float, float, float], half: float, ctrl: np.ndarray, level: bool
    ) -> tuple[tuple[np.ndarray, dict[str, np.ndarray]], str | None]:
        """Targets, from `ctrl`, putting both hands at `point` ± `half` sideways in the frame of the stance the legs
        hold, and the hands' orientations there. With `level`, hands holding the box keep their grip orientation, so
        it doesn't tip."""
        x, y, heading = self.pose()
        # The pelvis sways a couple of centimetres off its stance; aim from where it is now.
        offset = self.goal_xy - (x, y)
        goal, grip = ctrl.copy(), {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            in_stance = np.array([point[0], point[1] + sign * half])
            world = offset + _turn(in_stance, self.goal_yaw)
            target = np.array([*_turn(world, -heading), point[2]])
            rotation = self._grip[side] if level and self._grip else None
            q, miss, grip[side] = self.robot.solve_arm(goal, side, target, rotation)
            if miss > IK_TOLERANCE_M:
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
            "the table": lambda p: _clearance(p, np.array(TABLE_CENTER), np.array(TABLE_HALF), 0.0) - TABLE_KEEP_OUT_M,
            "the bin": lambda p: float(np.linalg.norm(p - bin_xy)) - BIN_KEEP_OUT_M,
        }
        for name, gap in gaps.items():
            start = gap(self.goal_xy)
            for a in np.linspace(0.0, 1.0, 21)[1:]:
                g = gap(self.goal_xy + a * (goal - self.goal_xy))
                if g < 0 and g < start - 1e-6:
                    return f"would walk into {name}"
        return None

    def _walk_to(self, xy: np.ndarray, yaw: float, expected_s: float, skill: str) -> str | None:
        """Walk the stance to `xy`, `yaw`; reports how far short it stopped if it hasn't arrived by the deadline."""
        self.goal_xy, self.goal_yaw = xy, yaw
        deadline = self.data.time + expected_s + 2.0
        while self.data.time < deadline and not self._arrived():
            self._hold(0.1)
        self._hold(WALK_SETTLE_S)
        distance, angle = self._off_goal()
        if distance < MISSED_M and angle < MISSED_YAW_RAD:
            return None
        return f"{skill}: stopped {distance * 100:.0f} cm and {np.degrees(angle):.0f} deg short"

    def _off_goal(self) -> tuple[float, float]:
        x, y, heading = self.pose()
        return float(np.hypot(x - self.goal_xy[0], y - self.goal_xy[1])), abs(wrap(self.goal_yaw - heading))

    def _arrived(self) -> bool:
        distance, angle = self._off_goal()
        return distance < ARRIVED_M and angle < ARRIVED_YAW_RAD

    def _hold(self, duration: float) -> None:
        move(self.model, self.data, self.data.ctrl.copy(), duration, self._tick, self._drive)

    def _drive(self) -> None:
        command = self.walker.command_to(self.data, self.goal_xy, self.goal_yaw, WALK_SPEED_M_S, TURN_RATE_RAD_S)
        self.walker.act(self.data, command)

    def joints_past_limits(self) -> list[str]:
        """The hinge joints more than JOINT_LIMIT_TOL_RAD past their range right now."""
        q = self.data.qpos
        return [
            name
            for name, adr, lo, hi in self._limited
            if q[adr] < lo - JOINT_LIMIT_TOL_RAD or q[adr] > hi + JOINT_LIMIT_TOL_RAD
        ]

    def _box_at_rest(self) -> bool:
        """For the last AT_REST_WINDOW_S of sim time the box has moved slower than AT_REST_SPEED_M_S and turned
        slower than AT_REST_SPIN_RAD_S."""
        now = self.data.time
        if not self._speeds or self._speeds[0][0] > now - AT_REST_WINDOW_S:
            return False
        return all(
            speed < AT_REST_SPEED_M_S and spin < AT_REST_SPIN_RAD_S
            for t, speed, spin in self._speeds
            if t >= now - AT_REST_WINDOW_S
        )

    def _touching(self, data: mujoco.MjData, geoms: set[int]) -> set[int]:
        """The geoms in contact with any of `geoms`, other than those."""
        pairs = [(c.geom1, c.geom2) for c in data.contact[: data.ncon]]
        return ({b for a, b in pairs if a in geoms} | {a for a, b in pairs if b in geoms}) - geoms

    def _tick(self) -> None:
        d = self.data
        self._actions.append([float(v) for v in d.ctrl])
        self._times.append(float(d.time))
        box_vel = d.qvel[self._box_dadr : self._box_dadr + 6]
        self._speeds.append((float(d.time), float(np.linalg.norm(box_vel[:3])), float(np.linalg.norm(box_vel[3:]))))
        while self._speeds and self._speeds[0][0] < d.time - 2 * AT_REST_WINDOW_S:
            self._speeds.popleft()
        if self.on_tick:
            self.on_tick()
        self._fallen |= d.xpos[self.robot.pelvis][2] < FALLEN_Z
        self._limit_violated |= bool(self.joints_past_limits())
        self._penetrated |= any(
            c.dist < -PENETRATION_TOL_M and (c.geom1 in self._scenery or c.geom2 in self._scenery)
            for c in d.contact[: d.ncon]
        )

    def _bin_yaw(self) -> float:
        return 2 * float(np.arctan2(self.data.mocap_quat[self._bin][3], self.data.mocap_quat[self._bin][0]))

    def _put_box(self, xy: tuple[float, float]) -> None:
        a = self._box_qadr
        self.data.qpos[a : a + 7] = [*xy, TABLE_TOP_Z + self.task.box_half[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._box_dadr : self._box_dadr + 6] = 0.0

    def _put_bin(self, data: mujoco.MjData, xy: tuple[float, float], yaw: float) -> None:
        data.mocap_pos[self._bin] = [*xy, 0.0]
        data.mocap_quat[self._bin] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
