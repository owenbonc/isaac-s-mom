"""FakeSim: a top-down 2D stand-in for the MuJoCo scene so every workstream and CI can run the loop.

It mirrors MujocoSim's skills with trivial physics: the robot walks and turns exactly as told, the hug catches the box
only when the hands are lined up with it, and `place` drops it straight ahead of the torso, into the bin or onto the
floor.
"""

import io
import math
import uuid

import numpy as np
from PIL import Image, ImageDraw

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall

BOX_X = 0.34  # metres ahead of the robot, where the arms work
BOX_HALF = (0.125, 0.15)
TABLE_CENTER, TABLE_HALF = (0.40, 0.0), (0.2, 0.25)
BIN_INNER_HALF, BIN_WALL = 0.22, 0.02
BIN_OUTER_HALF = BIN_INNER_HALF + BIN_WALL
BIN_DIST = (1.0, 1.4)
BIN_BEARING = (math.radians(75), math.radians(285))
TABLE_CLEARANCE = 0.45
PLACE_X = 0.40
REACH_LIMIT = 0.12
HUG_TOLERANCE = 0.03
TABLE_KEEP_OUT = 0.15
BIN_KEEP_OUT = PLACE_X  # the closest approach drops the box in the middle
DEFAULT_REACH_CM = 5.0
DEFAULT_WALK_CM = 30.0
DEFAULT_TURN_DEG = 30.0
STEPS_PER_SKILL = 10
DT = 0.02
TABLE_Z, HELD_Z, BIN_Z, FLOOR_Z = 0.845, 0.915, 0.445, 0.125
SIZE = (320, 240)
VIEW = 1.6  # metres from the robot to the image's top and bottom edges


def _rot(angle: float, p: tuple[float, float] | np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def _table_clearance(xy: np.ndarray) -> float:
    gap = np.maximum(np.abs(xy - TABLE_CENTER) - TABLE_HALF, 0.0)
    return float(np.linalg.norm(gap))


def _box_xy(rng: np.random.Generator) -> np.ndarray:
    return np.array([BOX_X + rng.uniform(-0.02, 0.02), rng.uniform(-0.1, 0.1)])


def _bin_pose(rng: np.random.Generator, around: np.ndarray) -> tuple[np.ndarray, float]:
    while True:
        bearing = rng.uniform(*BIN_BEARING)
        xy = around + rng.uniform(*BIN_DIST) * np.array([math.cos(bearing), math.sin(bearing)])
        if _table_clearance(xy) > BIN_INNER_HALF + BIN_WALL + TABLE_CLEARANCE:
            return xy, bearing


class FakeSim:
    sim_version = "fake-3"
    robot_config = "fake-walker-2d"

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed: int) -> str:
        rng = np.random.default_rng(seed)
        self.episode_id = f"ep-{seed}-{uuid.uuid4().hex[:6]}"
        self.box = _box_xy(rng)
        self.bin, self.bin_yaw = _bin_pose(rng, np.zeros(2))
        self.box_z = TABLE_Z
        self.xy, self.heading = np.zeros(2), 0.0
        self.work_y = 0.0
        self.closed = self.lifted = self.held = False
        self.step = 0
        self.sim_time = 0.0
        return self.episode_id

    def observe(self) -> Observation:
        return Observation(
            episode_id=self.episode_id,
            step=self.step,
            sim_time=self.sim_time,
            head_rgb_jpeg=self._render(),
            joint_state=self._joints(),
        )

    def execute(self, call: SkillCall) -> ExecResult:
        error = self._apply(call)
        self.step += 1
        if error:
            return ExecResult(ok=False, error=error, executed_actions=[], timestamps=[])
        actions, timestamps = [], []
        for _ in range(STEPS_PER_SKILL):
            self.sim_time += DT
            actions.append([*self.xy, self.heading, *self._joints()])
            timestamps.append(self.sim_time)
        return ExecResult(ok=True, executed_actions=actions, timestamps=timestamps)

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            box_xyz=(float(self.box[0]), float(self.box[1]), self.box_z),
            bin_xy=(float(self.bin[0]), float(self.bin[1])),
            bin_yaw=self.bin_yaw,
            bin_inner_half=BIN_INNER_HALF,
            box_on_bin_floor=self.box_z == BIN_Z,
            box_at_rest=True,
            robot_fallen=False,
            joint_limit_violation=False,
            penetration=False,
        )

    def perturb(self, perturbation: Perturbation) -> bool:
        rng = np.random.default_rng(perturbation.seed)
        if perturbation.kind == "move_box":
            if self.box_z != TABLE_Z or self.held:
                return False
            self.box = _box_xy(rng)
            return True
        if self.box_z == BIN_Z:  # moving it would tip the box out
            return False
        bin_xy, bin_yaw = _bin_pose(rng, self.xy)
        # A box on the floor is in the bin's way; one on the table or in the hands is above it.
        if self.box_z < TABLE_Z and not self._clear_of_box(bin_xy, bin_yaw):
            return False
        self.bin, self.bin_yaw = bin_xy, bin_yaw
        return True

    def render_third_person(self) -> bytes:
        return self._render()

    def pose(self) -> tuple[float, float, float]:
        """The robot's floor pose (x, y, heading), for tests; the agents never see it."""
        return float(self.xy[0]), float(self.xy[1]), self.heading

    def _joints(self) -> list[float]:
        return [self.work_y, float(self.closed), float(self.lifted)]

    def _apply(self, call: SkillCall) -> str | None:
        match call.skill:
            case "reach_left" | "reach_right":
                sign = 1.0 if call.skill == "reach_left" else -1.0
                work_y = self.work_y + sign * call.params.get("distance_cm", DEFAULT_REACH_CM) / 100
                if abs(work_y) > REACH_LIMIT + 1e-9:
                    return f"{call.skill}: past the {REACH_LIMIT * 100:.0f} cm reach limit"
                self.work_y = work_y
            case "walk_forward" | "walk_back":
                sign = 1.0 if call.skill == "walk_forward" else -1.0
                step = sign * call.params.get("distance_cm", DEFAULT_WALK_CM) / 100
                goal = self.xy + _rot(self.heading, (step, 0.0))
                error = self._blocked(goal)
                if error:
                    return f"{call.skill}: {error}"
                self.xy = goal
            case "turn_left" | "turn_right":
                sign = 1.0 if call.skill == "turn_left" else -1.0
                self.heading += sign * math.radians(call.params.get("angle_deg", DEFAULT_TURN_DEG))
            case "hug":
                local = _rot(-self.heading, self.box - self.xy)
                lined_up = abs(local[0] - BOX_X) < HUG_TOLERANCE and abs(local[1] - self.work_y) < HUG_TOLERANCE
                self.held = lined_up and self.box_z == TABLE_Z
                self.closed = True
            case "lift":
                self.lifted = True
                if self.held:
                    self.box_z = HELD_Z
            case "place":
                if self.held and self.lifted:
                    self.box = self.xy + _rot(self.heading, (PLACE_X, 0.0))
                    self.box_z = BIN_Z if self._over_bin() else FLOOR_Z
                self.closed = self.lifted = self.held = False
                self.work_y = 0.0
            case "reobserve":
                pass
        if self.held:
            self.box = self._hands(BOX_X)
        return None

    def _blocked(self, goal: np.ndarray) -> str | None:
        gaps = {
            "the table": lambda p: _table_clearance(p) - TABLE_KEEP_OUT,
            "the bin": lambda p: float(np.linalg.norm(p - self.bin)) - BIN_KEEP_OUT,
        }
        for name, gap in gaps.items():
            start = gap(self.xy)
            for a in np.linspace(0.0, 1.0, 21)[1:]:
                g = gap(self.xy + a * (goal - self.xy))
                if g < 0 and g < start - 1e-6:
                    return f"would walk into {name}"
        return None

    def _hands(self, forward: float) -> np.ndarray:
        return self.xy + _rot(self.heading, (forward, self.work_y))

    def _clear_of_box(self, bin_xy: np.ndarray, bin_yaw: float) -> bool:
        """A bin at this pose would stand clear of the box, however the box is turned."""
        x, y = self.box - bin_xy
        outside = np.maximum(np.abs(_rot(-bin_yaw, (float(x), float(y)))) - BIN_OUTER_HALF, 0.0)
        return float(np.linalg.norm(outside)) > math.hypot(*BOX_HALF)

    def _over_bin(self) -> bool:
        """Every corner of the box, which is turned with the robot, is inside the bin's walls."""
        corners = [(sx * BOX_HALF[0], sy * BOX_HALF[1]) for sx in (-1, 1) for sy in (-1, 1)]
        for corner in corners:
            local = _rot(-self.bin_yaw, self.box + _rot(self.heading, corner) - self.bin)
            if np.any(np.abs(local) > BIN_INNER_HALF):
                return False
        return True

    def _render(self) -> bytes:
        image = Image.new("RGB", SIZE, (205, 205, 200))
        draw = ImageDraw.Draw(image)
        scale = SIZE[1] / (2 * VIEW)

        def px(xy: np.ndarray) -> tuple[float, float]:
            # Top-down around the robot, turned with it: ahead is up and its left is left.
            local = _rot(-self.heading, xy - self.xy)
            return SIZE[0] / 2 - local[1] * scale, SIZE[1] / 2 - local[0] * scale

        def square(centre: np.ndarray, half: tuple[float, float], yaw: float, fill: tuple[int, int, int]) -> None:
            corners = [(half[0], half[1]), (-half[0], half[1]), (-half[0], -half[1]), (half[0], -half[1])]
            draw.polygon([px(centre + _rot(yaw, c)) for c in corners], fill=fill)

        square(np.array(TABLE_CENTER), TABLE_HALF, 0.0, (160, 115, 75))
        square(self.bin, (BIN_INNER_HALF, BIN_INNER_HALF), self.bin_yaw, (65, 160, 90))
        box_colour = (205, 85, 45) if self.box_z != FLOOR_Z else (120, 60, 40)
        square(self.box, BOX_HALF, self.heading if self.held else 0.0, box_colour)
        half = 0.135 if self.closed else 0.30
        for sign in (1.0, -1.0):
            hu, hv = px(self._hands(BOX_X) + _rot(self.heading, (0.0, sign * half)))
            r = 6 if self.lifted else 4
            draw.ellipse((hu - r, hv - r, hu + r, hv + r), fill=(60, 60, 60))
        cu, cv = px(self.xy)
        draw.ellipse((cu - 10, cv - 10, cu + 10, cv + 10), fill=(230, 230, 235), outline=(90, 90, 90))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85)
        return out.getvalue()
