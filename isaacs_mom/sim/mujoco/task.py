"""Configurable tasks for the MuJoCo scene (SPEC §3.1): the box, where it starts, and where the bin goes.

Every task is hug-and-place with the same skills and success rule, so a task changes only the scene's numbers and the
student, teacher and validator need nothing new. A task kind with its own skills or success rule would extend this.
"""

from dataclasses import dataclass

BIN_INNER_HALF = 0.22
# What the G1's arms can hug: the hands close 1.5 cm inside the box's sides and open 18 cm outside them.
BOX_HALF_WIDTH_RANGE = (0.10, 0.17)
BOX_HALF_DEPTH_RANGE = (0.08, 0.15)
BOX_HALF_HEIGHT_RANGE = (0.08, 0.15)
MAX_BOX_MASS_KG = 2.0
MAX_BOX_OFFSET_M = 0.12  # the arms' sideways reach
MIN_BIN_DISTANCE_M = 0.5  # the robot stops ~0.45 m short of the bin's centre to place


@dataclass(frozen=True)
class TaskConfig:
    name: str
    box_half: tuple[float, float, float] = (0.125, 0.15, 0.125)  # depth, width, height
    box_mass_kg: float = 0.8
    box_offset_m: float = 0.10  # the box starts within this of the table's centre line
    bin_distance_m: tuple[float, float] = (1.0, 1.4)
    bin_bearing_deg: tuple[float, float] = (75.0, 285.0)  # anticlockwise from straight ahead; the table is ahead

    def __post_init__(self) -> None:
        problems = []
        for value, (lo, hi), what in [
            (self.box_half[0], BOX_HALF_DEPTH_RANGE, "box half-depth"),
            (self.box_half[1], BOX_HALF_WIDTH_RANGE, "box half-width"),
            (self.box_half[2], BOX_HALF_HEIGHT_RANGE, "box half-height"),
            (self.box_mass_kg, (0.05, MAX_BOX_MASS_KG), "box mass"),
            (self.box_offset_m, (0.0, MAX_BOX_OFFSET_M), "box offset"),
        ]:
            if not lo <= value <= hi:
                problems.append(f"{what} {value} is outside {lo}-{hi}")
        # A box whose footprint diagonal spans the bin could wedge across it on the way down.
        if (2 * self.box_half[0]) ** 2 + (2 * self.box_half[1]) ** 2 >= (2 * BIN_INNER_HALF) ** 2:
            problems.append("the box's footprint diagonal is as wide as the bin")
        near, far = self.bin_distance_m
        if not MIN_BIN_DISTANCE_M <= near <= far:
            problems.append(f"bin distance {self.bin_distance_m} must run from at least {MIN_BIN_DISTANCE_M} m upwards")
        left, right = self.bin_bearing_deg
        if not 0.0 < left <= right < 360.0:
            problems.append(f"bin bearing {self.bin_bearing_deg} must lie between 0 and 360 degrees, low to high")
        if problems:
            raise ValueError(f"task {self.name!r}: " + "; ".join(problems))


TASKS = {
    task.name: task
    for task in [
        TaskConfig("across_room"),
        # Easier lessons: the box starts centred, or the bin is a turn away rather than a walk.
        TaskConfig("centred_box", box_offset_m=0.02),
        TaskConfig("near_bin", bin_distance_m=(0.7, 0.9), bin_bearing_deg=(95.0, 265.0)),
        # Harder ones: a heavier or smaller box to hold, or a longer walk.
        TaskConfig("heavy_box", box_mass_kg=2.0),
        TaskConfig("small_box", box_half=(0.10, 0.11, 0.10)),
        TaskConfig("far_bin", bin_distance_m=(1.6, 2.2)),
    ]
}
DEFAULT_TASK = TASKS["across_room"]
