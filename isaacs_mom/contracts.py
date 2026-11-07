"""Shared types for every workstream (SPEC §4). A change here updates every caller in the same PR."""

from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

Skill = Literal["approach_left", "approach_right", "push_forward", "retract", "reobserve"]
SKILLS: tuple[Skill, ...] = get_args(Skill)
MAX_DISTANCE_CM = 20.0

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
    def _only_distance(cls, params: dict[str, float]) -> dict[str, float]:
        unknown = set(params) - {"distance_cm"}
        if unknown:
            raise ValueError(f"unknown skill params: {sorted(unknown)}")
        if "distance_cm" in params:
            params["distance_cm"] = min(max(params["distance_cm"], 0.0), MAX_DISTANCE_CM)
        return params


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
    kind: Literal["move_block", "move_target"]
    seed: int


class GroundTruth(BaseModel):
    block_xy: tuple[float, float]
    target_xy: tuple[float, float]
    target_radius: float
    block_at_rest: bool
    joint_limit_violation: bool
    table_penetration: bool


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


class ReplayReport(BaseModel):
    checked: int
    reproduced: int
    not_reproduced: list[str]


class EvaluationResult(BaseModel):
    checkpoint: str
    round: int
    successes: int
    n: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0


class Event(BaseModel):
    t: float
    run_id: str
    type: EventType
    data: dict[str, object] = Field(default_factory=dict)
