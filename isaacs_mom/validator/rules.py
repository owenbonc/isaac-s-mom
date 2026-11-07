import math

from isaacs_mom.contracts import GroundTruth, Validation

ROBOT_FELL = "robot fell"
JOINT_LIMIT = "joint limit violated"
PENETRATION = "penetration"
# Robot-relative: short of the bin is between the robot and it, past is beyond it, left and right as the head camera
# sees it.
DIRECTIONS = ("short", "past", "left", "right", "short-left", "short-right", "past-left", "past-right")
MISSES = {d: "box past the bin" if d == "past" else f"box {d} of the bin" for d in DIRECTIONS}
ON_RIM = "box on the rim"
STILL_MOVING = "box still moving"
RECORDING_ERROR = "recording error"
CATEGORIES = {
    "robot_fallen": ROBOT_FELL,
    "joint_limit": JOINT_LIMIT,
    "penetration": PENETRATION,
    **{"past_bin" if d == "past" else f"{d.replace('-', '_')}_of_bin": MISSES[d] for d in DIRECTIONS},
    "on_rim": ON_RIM,
    "still_moving": STILL_MOVING,
    "recording_error": RECORDING_ERROR,
}
SEPARATOR = "; "


def validate(gt: GroundTruth) -> Validation:
    """SPEC §3.4: the box at rest on the bin's floor with its centre inside the walls; no fall, joint-limit violation
    or penetration.

    The reason lists every problem in words, with the direction of a miss in the robot's frame but no measurements,
    because the teacher reads it and ground truth must not reach the teacher (SPEC §2 rule 1).
    """
    dx, dy = gt.box_xyz[0] - gt.bin_xy[0], gt.box_xyz[1] - gt.bin_xy[1]
    along_wall, across_wall = _turned(dx, dy, gt.bin_yaw)
    over_bin = max(abs(along_wall), abs(across_wall)) <= gt.bin_inner_half
    inside = gt.box_on_bin_floor and over_bin
    problems = [
        message
        for flagged, message in (
            (gt.robot_fallen, ROBOT_FELL),
            (gt.joint_limit_violation, JOINT_LIMIT),
            (gt.penetration, PENETRATION),
        )
        if flagged
    ]
    if not inside:
        problems.append(ON_RIM if over_bin else MISSES[_miss_direction(gt, dx, dy)])
    if not gt.box_at_rest:
        problems.append(STILL_MOVING)
    return Validation(
        task_success=inside and gt.box_at_rest,
        physics_valid=not (gt.robot_fallen or gt.joint_limit_violation or gt.penetration),
        reason=SEPARATOR.join(problems) or "box at rest inside bin",
    )


def _miss_direction(gt: GroundTruth, dx: float, dy: float) -> str:
    """Which way the box missed in the robot's torso frame at the end of the episode.

    An axis is named only when the box is clearly beyond the bin's half-width along it (a dead zone), so a near-miss
    gets no jittery second label. A miss inside both dead zones is named by its larger axis.
    """
    forward, left = _turned(dx, dy, gt.robot_yaw)
    name_along, name_side = abs(forward) > gt.bin_inner_half, abs(left) > gt.bin_inner_half
    if not (name_along or name_side):
        name_along = abs(forward) >= abs(left)
        name_side = not name_along
    along = ("past" if forward > 0 else "short") if name_along else ""
    side = ("left" if left > 0 else "right") if name_side else ""
    return "-".join(part for part in (along, side) if part)


def _turned(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    """An offset in the frame turned by `yaw`: (forward, left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def recording_error_verdict(error: Exception) -> Validation:
    """The verdict for an episode the recorder refused: a simulator bug, never a lesson, counted apart from misses."""
    return Validation(task_success=False, physics_valid=False, reason=f"{RECORDING_ERROR}: {error}")


def failure_categories(reason: str) -> list[str]:
    """The problem kinds named in a verdict's reason, in the order `validate` lists them."""
    parts = reason.split(SEPARATOR)
    return [category for category, marker in CATEGORIES.items() if any(part.startswith(marker) for part in parts)]
