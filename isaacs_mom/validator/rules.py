import math

from isaacs_mom.contracts import GroundTruth, Validation

ROBOT_FELL = "robot fell"
JOINT_LIMIT = "joint limit violated"
PENETRATION = "penetration"
# Robot-relative: along and across the torso heading, with left as the head camera sees it.
MISSES = {
    "short": "box short of the bin",
    "past": "box past the bin",
    "left": "box left of the bin",
    "right": "box right of the bin",
    "short-left": "box short-left of the bin",
    "short-right": "box short-right of the bin",
    "past-left": "box past-left of the bin",
    "past-right": "box past-right of the bin",
}
ON_RIM = "box on the rim"
STILL_MOVING = "box still moving"
RECORDING_ERROR = "recording error"
CATEGORIES = {
    "robot_fallen": ROBOT_FELL,
    "joint_limit": JOINT_LIMIT,
    "penetration": PENETRATION,
    "short_of_bin": MISSES["short"],
    "past_bin": MISSES["past"],
    "left_of_bin": MISSES["left"],
    "right_of_bin": MISSES["right"],
    "short_left_of_bin": MISSES["short-left"],
    "short_right_of_bin": MISSES["short-right"],
    "past_left_of_bin": MISSES["past-left"],
    "past_right_of_bin": MISSES["past-right"],
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
        problems.append(ON_RIM if over_bin else MISSES[_miss_direction(dx, dy, gt.robot_yaw, gt.bin_inner_half)])
    if not gt.box_at_rest:
        problems.append(STILL_MOVING)
    return Validation(
        task_success=inside and gt.box_at_rest,
        physics_valid=not (gt.robot_fallen or gt.joint_limit_violation or gt.penetration),
        reason=SEPARATOR.join(problems) or "box at rest inside bin",
    )


def _miss_direction(dx: float, dy: float, robot_yaw: float, half: float) -> str:
    """Which way the box missed in the robot's torso frame at the end of the episode.

    An axis is named only when the box is clearly beyond the bin's half-width along it (a dead zone), so a near-miss
    gets no jittery second label. A miss inside both dead zones is named by its larger axis.
    """
    forward, left = _turned(dx, dy, robot_yaw)
    along = "past" if forward > 0 else "short"
    side = "left" if left > 0 else "right"
    clear_along, clear_side = abs(forward) > half, abs(left) > half
    if clear_along and clear_side:
        return f"{along}-{side}"
    if clear_along or clear_side:
        return along if clear_along else side
    return along if abs(forward) >= abs(left) else side


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
