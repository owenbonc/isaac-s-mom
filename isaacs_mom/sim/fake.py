"""FakeSim: a top-down 2D stand-in for the MuJoCo scene so every workstream and CI can run the loop.

The hand moves on the table plane; push_forward drags the block along +x when the hand is lined up behind it.
"""

import io
import uuid

import numpy as np
from PIL import Image, ImageDraw

from isaacs_mom.contracts import (
    ExecResult,
    GroundTruth,
    Observation,
    Perturbation,
    SkillCall,
)

TABLE_X = (0.2, 0.8)  # metres, forward from the robot
TABLE_Y = (-0.3, 0.3)  # metres, left is negative
HAND_HOME = (0.25, 0.0)
BLOCK_HALF = 0.025
TARGET_RADIUS = 0.05
ALIGN_TOLERANCE = 0.03
DEFAULT_STEP_M = 0.05
STEPS_PER_SKILL = 10
DT = 0.02
SIZE = (320, 240)


def _layout(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Block and target positions as a pure function of the seed (SPEC §3.1)."""
    rng = np.random.default_rng(seed)
    block = np.array([rng.uniform(0.35, 0.45), rng.uniform(-0.15, 0.15)])
    target = np.array([block[0] + rng.uniform(0.12, 0.22), block[1] + rng.uniform(-0.02, 0.02)])
    return block, target


class FakeSim:
    sim_version = "fake-1"
    robot_config = "fake-hand-2d"

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed: int) -> str:
        self.episode_id = f"ep-{seed}-{uuid.uuid4().hex[:6]}"
        self.block, self.target = _layout(seed)
        self.hand = np.array(HAND_HOME)
        self.step = 0
        self.sim_time = 0.0
        return self.episode_id

    def observe(self) -> Observation:
        return Observation(
            episode_id=self.episode_id,
            step=self.step,
            sim_time=self.sim_time,
            head_rgb_jpeg=self._render(),
            joint_state=[float(v) for v in self.hand],
        )

    def execute(self, call: SkillCall) -> ExecResult:
        distance = call.params.get("distance_cm", DEFAULT_STEP_M * 100) / 100
        direction = {
            "approach_left": (0.0, -1.0),
            "approach_right": (0.0, 1.0),
            "push_forward": (1.0, 0.0),
            "retract": (-1.0, 0.0),
            "reobserve": (0.0, 0.0),
        }[call.skill]
        goal = self.hand + distance * np.array(direction)
        goal = np.clip(goal, [TABLE_X[0], TABLE_Y[0]], [TABLE_X[1], TABLE_Y[1]])
        actions: list[list[float]] = []
        timestamps: list[float] = []
        start = self.hand.copy()
        for i in range(1, STEPS_PER_SKILL + 1):
            self.hand = start + (goal - start) * i / STEPS_PER_SKILL
            self._push_block()
            self.sim_time += DT
            actions.append([float(v) for v in self.hand])
            timestamps.append(self.sim_time)
        self.step += 1
        return ExecResult(ok=True, executed_actions=actions, timestamps=timestamps)

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            block_xy=(float(self.block[0]), float(self.block[1])),
            target_xy=(float(self.target[0]), float(self.target[1])),
            target_radius=TARGET_RADIUS,
            block_at_rest=True,
            joint_limit_violation=False,
            table_penetration=False,
        )

    def perturb(self, perturbation: Perturbation) -> None:
        moved_block, moved_target = _layout(perturbation.seed)
        if perturbation.kind == "move_block":
            self.block = moved_block
        else:
            self.target = moved_target

    def render_third_person(self) -> bytes:
        return self._render()

    def _push_block(self) -> None:
        lined_up = abs(self.hand[1] - self.block[1]) < ALIGN_TOLERANCE
        if lined_up and self.hand[0] > self.block[0] - BLOCK_HALF and self.hand[0] < self.block[0] + BLOCK_HALF:
            self.block[0] = self.hand[0] + BLOCK_HALF

    def _render(self) -> bytes:
        image = Image.new("RGB", SIZE, (205, 180, 140))
        draw = ImageDraw.Draw(image)

        def px(xy: np.ndarray) -> tuple[float, float]:
            # Forward (+x) is up the image, left (-y) is left.
            u = (xy[1] - TABLE_Y[0]) / (TABLE_Y[1] - TABLE_Y[0]) * SIZE[0]
            v = SIZE[1] - (xy[0] - TABLE_X[0]) / (TABLE_X[1] - TABLE_X[0]) * SIZE[1]
            return u, v

        scale = SIZE[1] / (TABLE_X[1] - TABLE_X[0])
        tu, tv = px(self.target)
        r = TARGET_RADIUS * scale
        draw.ellipse((tu - r, tv - r, tu + r, tv + r), fill=(60, 170, 80))
        bu, bv = px(self.block)
        h = BLOCK_HALF * scale
        draw.rectangle((bu - h, bv - h, bu + h, bv + h), fill=(200, 40, 40))
        hu, hv = px(self.hand)
        draw.ellipse((hu - 6, hv - 6, hu + 6, hv + 6), fill=(90, 90, 90))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85)
        return out.getvalue()
