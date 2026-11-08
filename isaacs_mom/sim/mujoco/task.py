"""Configurable tasks for the MuJoCo scene (SPEC §3.1): walking or fixed feet, the box, and where the bin goes.

Every task is hug-and-place with the same success rule. Walking tasks carry the box to the bin on the legs' walking
policy; `fixed_feet` twists the waist to a bin beside the robot. A task with its own success rule would extend this.
"""

from dataclasses import dataclass

BIN_INNER_HALF = 0.22
# What the G1's arms can hug: the hands close 1.5 cm inside the box's sides and open 18 cm outside them.
BOX_HALF_WIDTH_RANGE = (0.10, 0.17)
BOX_HALF_DEPTH_RANGE = (0.08, 0.15)
BOX_HALF_HEIGHT_RANGE = (0.08, 0.15)
MAX_BOX_MASS_KG = 2.0
MAX_BOX_OFFSET_M = 0.12  # the arms' sideways reach
MIN_BIN_DISTANCE_M = 0.5  # a walking robot stops ~0.45 m short of the bin's centre to place
MAX_BIN_DISTANCE_M = 2.2  # the longest walk measured, far_bin's
TWIST_BIN_DISTANCE_M = (0.40, 0.50)  # a fixed-feet bin must sit where the twisted, outstretched arms put the box
# Where `place` lands the box on a wall-less top every time: higher, it lands still between the open hands.
PLATFORM_TOP_RANGE_M = (0.40, 0.45)
# Boxes measured to settle on the top every time: the default and the small box up to 1 kg (20/20 seeds). With no wall
# to hold it off, a 2 kg box stays against the hands (14/20) and a taller one can tip (19/20).
MAX_PLATFORM_BOX_HALF = (0.125, 0.15, 0.125)
MAX_PLATFORM_BOX_MASS_KG = 1.0


@dataclass(frozen=True)
class Rig:
    """How the robot meets a kind of task: its head camera, the third-person view and the arms' carry and drop."""

    head_camera_pos: tuple[float, float, float]
    head_camera_tilt_deg: float
    head_camera_fovy_deg: float
    third_person_eye: tuple[float, float, float]
    third_person_target: tuple[float, float, float]
    third_person_fovy_deg: float
    lift_m: float
    open_clearance_m: float  # how far outside the box's sides the open hands wait
    place_x_m: float  # how far ahead of the pelvis the box is let go
    place_z_m: float


# A wide lens tilted 45 degrees sees the bin above the box carried 7 cm up, from 1-1.4 m away.
WALKING_RIG = Rig(
    (0.10, 0.0, 0.45),
    45.0,
    100.0,
    (2.3, -2.3, 2.9),
    (-0.25, 0.1, 0.3),
    55.0,
    lift_m=0.07,
    open_clearance_m=0.18,
    place_x_m=0.40,
    place_z_m=0.84,
)
FIXED_FEET_RIG = Rig(
    (0.09, 0.0, 0.40),
    50.0,
    75.0,
    (1.3, 1.5, 1.6),
    (0.05, 0.15, 0.75),
    50.0,
    lift_m=0.12,
    open_clearance_m=0.15,
    place_x_m=0.38,
    place_z_m=0.86,
)


@dataclass(frozen=True)
class TaskConfig:
    name: str
    walking: bool = True
    box_half: tuple[float, float, float] = (0.125, 0.15, 0.125)  # depth, width, height
    box_mass_kg: float = 0.8
    box_offset_m: float = 0.10  # the box starts within this of the table's centre line
    bin_distance_m: tuple[float, float] = (1.0, 1.4)
    bin_bearing_deg: tuple[float, float] = (75.0, 285.0)  # anticlockwise from straight ahead; the table is ahead
    platform_top_m: float | None = None  # unset, the bin; set, a wall-less platform with its top this high

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
        if self.walking and not MIN_BIN_DISTANCE_M <= near <= far <= MAX_BIN_DISTANCE_M:
            walkable = f"{MIN_BIN_DISTANCE_M}-{MAX_BIN_DISTANCE_M} m"
            problems.append(f"bin distance {self.bin_distance_m} must run low to high within {walkable}")
        if not self.walking and not TWIST_BIN_DISTANCE_M[0] <= near <= far <= TWIST_BIN_DISTANCE_M[1]:
            problems.append(f"bin distance {self.bin_distance_m} is outside a waist twist's {TWIST_BIN_DISTANCE_M} m")
        left, right = self.bin_bearing_deg
        if not 0.0 < left <= right < 360.0:
            problems.append(f"bin bearing {self.bin_bearing_deg} must lie between 0 and 360 degrees, low to high")
        if self.platform_top_m is not None:
            lo, hi = PLATFORM_TOP_RANGE_M
            if not lo <= self.platform_top_m <= hi:
                problems.append(f"platform top {self.platform_top_m} m is outside the {lo}-{hi} m the drop lands on")
            if not self.walking:
                problems.append("a platform is measured only for walking tasks")
            if any(half > most for half, most in zip(self.box_half, MAX_PLATFORM_BOX_HALF, strict=True)):
                problems.append(
                    f"box half-size {self.box_half} is past the {MAX_PLATFORM_BOX_HALF} m measured on a platform"
                )
            if self.box_mass_kg > MAX_PLATFORM_BOX_MASS_KG:
                problems.append(
                    f"box mass {self.box_mass_kg} kg is past the {MAX_PLATFORM_BOX_MASS_KG} kg measured on a platform"
                )
        if problems:
            raise ValueError(f"task {self.name!r}: " + "; ".join(problems))

    @property
    def rig(self) -> Rig:
        return WALKING_RIG if self.walking else FIXED_FEET_RIG


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
        # The robot stays on its feet and twists its waist to a bin beside it.
        TaskConfig("fixed_feet", walking=False, bin_distance_m=(0.43, 0.45), bin_bearing_deg=(97.0, 112.0)),
        # Fixed-feet layouts the fixed_feet task never uses (§6.10's new_layout): the bin further ahead, or further out.
        TaskConfig("fixed_feet_ahead", walking=False, bin_distance_m=(0.43, 0.45), bin_bearing_deg=(85.0, 97.0)),
        TaskConfig("fixed_feet_far", walking=False, bin_distance_m=(0.45, 0.50), bin_bearing_deg=(97.0, 112.0)),
    ]
}
DEFAULT_TASK = TASKS["across_room"]
