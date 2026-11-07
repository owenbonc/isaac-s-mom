"""FakeSim: a top-down 2D stand-in for the MuJoCo scene so every workstream and CI can run the loop.

It mirrors MujocoSim's skills with trivial physics: the hug catches the box only when the hands are lined up with it
and the waist faces the table, and `place` drops it straight ahead of the torso, into the bin or onto the floor.
"""

import io
import math
import uuid

import numpy as np
from PIL import Image, ImageDraw

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall

BOX_X = 0.34  # metres forward of the robot, where the arms work
BOX_HALF = (0.125, 0.15)
BIN_INNER_HALF = 0.22
PLACE_X = 0.38
REACH_LIMIT = 0.12
TURN_LIMIT = math.radians(150)
HUG_TOLERANCE = 0.03
DEFAULT_REACH_CM = 5.0
DEFAULT_TURN_DEG = 30.0
STEPS_PER_SKILL = 10
DT = 0.02
TABLE_Z, HELD_Z, BIN_Z, FLOOR_Z = 0.845, 0.965, 0.445, 0.125
SIZE = (320, 240)
VIEW = 0.6  # metres from the robot to the image edge


def _box_xy(rng: np.random.Generator) -> np.ndarray:
    return np.array([BOX_X + rng.uniform(-0.02, 0.02), rng.uniform(-0.1, 0.1)])


def _bin_pose(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    angle = rng.uniform(math.radians(97), math.radians(112))
    dist = rng.uniform(0.43, 0.45)
    return np.array([dist * math.cos(angle), dist * math.sin(angle)]), angle


def _rot(angle: float, p: tuple[float, float]) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


class FakeSim:
    sim_version = "fake-2"
    robot_config = "fake-hugger-2d"

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed: int) -> str:
        rng = np.random.default_rng(seed)
        self.episode_id = f"ep-{seed}-{uuid.uuid4().hex[:6]}"
        self.box = _box_xy(rng)
        self.bin, self.bin_yaw = _bin_pose(rng)
        self.box_z = TABLE_Z
        self.work_y = 0.0
        self.waist = 0.0
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
            joint_state=[self.work_y, self.waist, float(self.closed), float(self.lifted)],
        )

    def execute(self, call: SkillCall) -> ExecResult:
        error = self._apply(call)
        self.step += 1
        if error:
            return ExecResult(ok=False, error=error, executed_actions=[], timestamps=[])
        actions, timestamps = [], []
        for _ in range(STEPS_PER_SKILL):
            self.sim_time += DT
            actions.append([self.work_y, self.waist, float(self.closed), float(self.lifted)])
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

    def perturb(self, perturbation: Perturbation) -> None:
        rng = np.random.default_rng(perturbation.seed)
        if perturbation.kind == "move_box":
            if self.box_z == TABLE_Z and not self.held:
                self.box = _box_xy(rng)
        else:
            self.bin, self.bin_yaw = _bin_pose(rng)

    def render_third_person(self) -> bytes:
        return self._render()

    def _apply(self, call: SkillCall) -> str | None:
        match call.skill:
            case "reach_left" | "reach_right":
                sign = 1.0 if call.skill == "reach_left" else -1.0
                work_y = self.work_y + sign * call.params.get("distance_cm", DEFAULT_REACH_CM) / 100
                if abs(work_y) > REACH_LIMIT + 1e-9:
                    return f"{call.skill}: would reach {abs(work_y) * 100:.0f} cm off-centre"
                self.work_y = work_y
                if self.held:
                    self.box = self._hands(BOX_X)
            case "turn_left" | "turn_right":
                sign = 1.0 if call.skill == "turn_left" else -1.0
                waist = self.waist + sign * math.radians(call.params.get("angle_deg", DEFAULT_TURN_DEG))
                if abs(waist) > TURN_LIMIT:
                    return f"{call.skill}: would twist {math.degrees(abs(waist)):.0f} deg"
                self.waist = waist
                if self.held:
                    self.box = self._hands(BOX_X)
            case "hug":
                facing_table = abs(self.waist) < math.radians(5)
                lined_up = abs(self.work_y - self.box[1]) < HUG_TOLERANCE
                self.held = facing_table and lined_up and self.box_z == TABLE_Z
                self.closed = True
            case "lift":
                self.lifted = True
                if self.held:
                    self.box_z = HELD_Z
            case "place":
                if self.held and self.lifted:
                    self.box = _rot(self.waist, (PLACE_X, 0.0))
                    self.box_z = BIN_Z if self._over_bin() else FLOOR_Z
                self.closed = self.lifted = self.held = False
                self.work_y = 0.0
            case "reobserve":
                pass
        return None

    def _hands(self, forward: float) -> np.ndarray:
        return _rot(self.waist, (forward, self.work_y))

    def _over_bin(self) -> bool:
        """Every corner of the box, which is turned with the waist, is inside the bin's walls."""
        corners = [(sx * BOX_HALF[0], sy * BOX_HALF[1]) for sx in (-1, 1) for sy in (-1, 1)]
        for corner in corners:
            x, y = self.box + _rot(self.waist, corner) - self.bin
            if np.any(np.abs(_rot(-self.bin_yaw, (x, y))) > BIN_INNER_HALF):
                return False
        return True

    def _render(self) -> bytes:
        image = Image.new("RGB", SIZE, (205, 205, 200))
        draw = ImageDraw.Draw(image)
        scale = SIZE[1] / (2 * VIEW)

        def px(xy: np.ndarray) -> tuple[float, float]:
            # Top-down with the robot at the centre: forward (+x) is up, left (+y) is left.
            return SIZE[0] / 2 - xy[1] * scale, SIZE[1] / 2 - xy[0] * scale

        def square(centre: np.ndarray, half: tuple[float, float], yaw: float, fill: tuple[int, int, int]) -> None:
            corners = [(half[0], half[1]), (-half[0], half[1]), (-half[0], -half[1]), (half[0], -half[1])]
            draw.polygon([px(centre + _rot(yaw, c)) for c in corners], fill=fill)

        square(np.array([0.40, 0.0]), (0.2, 0.25), 0.0, (160, 115, 75))
        square(self.bin, (BIN_INNER_HALF, BIN_INNER_HALF), self.bin_yaw, (65, 160, 90))
        box_colour = (205, 85, 45) if self.box_z != FLOOR_Z else (120, 60, 40)
        square(self.box, BOX_HALF, self.waist if self.held else 0.0, box_colour)
        half = 0.135 if self.closed else 0.30
        for sign in (1.0, -1.0):
            hu, hv = px(_rot(self.waist, (BOX_X, self.work_y + sign * half)))
            r = 9 if self.lifted else 6
            draw.ellipse((hu - r, hv - r, hu + r, hv + r), fill=(60, 60, 60))
        cu, cv = px(np.zeros(2))
        draw.ellipse((cu - 14, cv - 14, cu + 14, cv + 14), fill=(230, 230, 235), outline=(90, 90, 90))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85)
        return out.getvalue()
