"""Shared types for every workstream (SPEC §4). A change here updates every caller in the same PR."""

import math
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

Skill = Literal[
    "reach_left",
    "reach_right",
    "walk_forward",
    "walk_back",
    "turn_left",
    "turn_right",
    "hug",
    "lift",
    "place",
    "reobserve",
]
SKILLS: tuple[Skill, ...] = get_args(Skill)
MAX_DISTANCE_CM = 150.0  # one walk across the room; the arms' reach is refused by the simulator well before this
MAX_ANGLE_DEG = 180.0
PARAM_LIMITS = {"distance_cm": MAX_DISTANCE_CM, "angle_deg": MAX_ANGLE_DEG}

Controller = Literal["student", "teacher"]
Split = Literal["dev", "heldout"]
EventType = Literal[
    "episode_started",
    "observation",
    "decision",
    "action_executed",
    "perturbation",
    "validation",
    "diagnosis",
    "finetune_started",
    "finetune_completed",
    "evaluation_result",
    "argon_call",
    "error",
]


class SkillTarget(BaseModel):
    """A skill and its parameters: what the student is trained to output (SPEC §4.3)."""

    skill: Skill
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _known_and_clamped(cls, params: dict[str, float]) -> dict[str, float]:
        unknown = set(params) - set(PARAM_LIMITS)
        if unknown:
            raise ValueError(f"unknown skill params: {sorted(unknown)}")
        # json.loads accepts NaN and Infinity, and clamping can't fix them: min/max pass NaN straight through.
        non_finite = sorted(k for k, v in params.items() if not math.isfinite(v))
        if non_finite:
            raise ValueError(f"skill params must be finite: {non_finite}")
        return {k: min(max(v, 0.0), PARAM_LIMITS[k]) for k, v in params.items()}


class SkillCall(SkillTarget):
    """A policy's decision. `reason` is shown on the dashboard and never trained on."""

    reason: str = ""


class Observation(BaseModel):
    """What the student and teacher may see: the head camera and the robot's own joints, never object poses."""

    episode_id: str
    step: int
    sim_time: float
    head_rgb_jpeg: bytes
    joint_state: list[float]


class ExecResult(BaseModel):
    ok: bool
    error: str | None = None
    executed_actions: list[list[float]]
    timestamps: list[float]


class Perturbation(BaseModel):
    kind: Literal["move_box", "move_bin"]
    seed: int


class GroundTruth(BaseModel):
    """Simulator facts for the validator and orchestrator only (SPEC §2 rule 1). The bin is square, `bin_inner_half`
    wide either side of its centre and turned by `bin_yaw`; `box_on_bin_floor` means touching the floor inside its
    walls. `robot_xy` is the pelvis on the floor and `robot_yaw` the torso heading (pelvis heading plus waist twist),
    the frame the head camera and the skills share."""

    box_xyz: tuple[float, float, float]
    bin_xy: tuple[float, float]
    bin_yaw: float
    bin_inner_half: float
    box_on_bin_floor: bool
    box_at_rest: bool
    robot_fallen: bool
    joint_limit_violation: bool
    penetration: bool
    robot_xy: tuple[float, float]
    robot_yaw: float


class Validation(BaseModel):
    task_success: bool
    physics_valid: bool
    reason: str

    @property
    def accepted(self) -> bool:
        return self.task_success and self.physics_valid


class Diagnosis(BaseModel):
    hypothesis: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    lesson: str


class StepRecord(BaseModel):
    step: int
    sim_time: float
    frame: str
    joint_state: list[float]
    skill_call: SkillCall
    controller: Controller
    checkpoint: str | None
    actions: str
    timestamps: list[float]
    ok: bool
    error: str | None = None


class PerturbationRecord(BaseModel):
    before_step: int
    perturbation: Perturbation


class EpisodeManifest(BaseModel):
    episode_id: str
    run_id: str
    seed: int
    split: Split
    round: int
    instruction: str
    sim_version: str
    robot_config: str
    # Who drove the episode from its first step, so one refused before any step is still attributed.
    driver: Controller | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    perturbations: list[PerturbationRecord] = Field(default_factory=list)
    takeover_step: int | None = None
    diagnosis: Diagnosis | None = None
    validation: Validation | None = None
    replay_valid: bool | None = None


class TrainingExample(BaseModel):
    image: str
    instruction: str
    feedback: str
    target: SkillTarget


class TrainingSetSummary(BaseModel):
    """The counts behind one training set, so every number shown carries its sample size (SPEC §2 rule 6)."""

    round: int
    episodes_scanned: int
    demonstrations: int
    accepted: int
    rejected: int
    unfinished: int
    rejected_by: dict[str, int]
    empty_dev_episodes: int
    examples: int


class ReplayFailure(BaseModel):
    episode_id: str
    reason: str


class ReplayReport(BaseModel):
    """`reproduced` out of `replayed` accepted episodes, with why each of the others failed."""

    replayed: int
    reproduced: int
    failures: list[ReplayFailure]


class EvaluationResult(BaseModel):
    checkpoint: str
    round: int
    successes: int
    n: int
    errored: int

    @property
    def rate(self) -> float:
        """Successes out of the episodes that didn't error (SPEC §3.5)."""
        scored = self.n - self.errored
        return self.successes / scored if scored else 0.0


class Event(BaseModel):
    t: float
    run_id: str
    type: EventType
    data: dict[str, object] = Field(default_factory=dict)
