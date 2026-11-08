"""A generated task (SPEC §3.7): one box the hug can hold, one place to put it, a declarative success rule and the
moves allowed mid-run. Parsed from Argon's JSON at the boundary; nothing in it is ever executed as code."""

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from isaacs_mom.sim.mujoco.task import TaskConfig

Id = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,23}$")]
Range = tuple[float, float]


class Strict(BaseModel):
    """Validated from JSON: a number is a finite number, never a string, a bool or Infinity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Box(Strict):
    kind: Literal["box"]
    id: Id
    half_m: tuple[float, float, float]  # depth, width, height
    mass_kg: float
    offset_m: float  # starts on the table within this of its centre line


class Bin(Strict):
    kind: Literal["bin"]
    id: Id
    distance_m: Range  # from where the robot starts
    bearing_deg: Range  # anticlockwise from straight ahead, where the table is


class Platform(Strict):
    """A shelf or platform: a flat top with no walls."""

    kind: Literal["shelf", "platform"]
    id: Id
    top_m: float
    distance_m: Range
    bearing_deg: Range


class Scene(Strict):
    box: Box
    destination: Annotated[Bin | Platform, Field(discriminator="kind")]


class Inside(Strict):
    """On the bin's floor, its centre within the walls."""

    fact: Literal["inside"]
    object: Id
    container: Id


class On(Strict):
    """Resting on the support's top, in contact, its centre over it."""

    fact: Literal["on"]
    object: Id
    support: Id


class AtRest(Strict):
    """Neither moving nor turning for 1 s of sim time."""

    fact: Literal["at_rest"]
    object: Id


class RobotStanding(Strict):
    fact: Literal["robot_standing"]


Fact = Annotated[Inside | On | AtRest | RobotStanding, Field(discriminator="fact")]


class Success(Strict):
    all_of: list[Fact] = Field(min_length=1, max_length=6)


class PermittedMove(Strict):
    """The object may be moved mid-run, within its own placement: the box on the table, the destination at its
    distance and bearing from the robot."""

    object: Id


class TaskSpec(Strict):
    name: Id
    instruction: str = Field(min_length=1, max_length=300)
    scene: Scene
    success: Success
    perturbations: list[PermittedMove] = Field(default=[], max_length=2)
    stance: Literal["walking", "fixed_feet"] = "walking"  # fixed feet: the robot twists its waist to a bin beside it

    def fingerprint(self) -> str:
        """Equal for equal specs, whatever the key order of the JSON they came from."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def task_config(self) -> TaskConfig:
        """The scene as a MuJoCo task; its name carries the fingerprint into `sim_version`, so a recording replays
        only on this exact spec. Raises ValueError, with every reason, for a scene the G1 can't do."""
        box, dest = self.scene.box, self.scene.destination
        return TaskConfig(
            f"gen-{self.name}-{self.fingerprint()}",
            walking=self.stance == "walking",
            box_half=box.half_m,
            box_mass_kg=box.mass_kg,
            box_offset_m=box.offset_m,
            bin_distance_m=dest.distance_m,
            bin_bearing_deg=dest.bearing_deg,
            platform_top_m=dest.top_m if isinstance(dest, Platform) else None,
        )

    def perturbation_kinds(self) -> list[Literal["move_box", "move_bin"]]:
        return ["move_box" if move.object == self.scene.box.id else "move_bin" for move in self.perturbations]


class Refusal(Strict):
    instruction: str
    reasons: list[str] = Field(min_length=1)
