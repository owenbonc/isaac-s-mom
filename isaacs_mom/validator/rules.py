import math

from isaacs_mom.contracts import GroundTruth, Validation

ROBOT_FELL = "robot fell"
JOINT_LIMIT = "joint limit violated"
PENETRATION = "penetration"
SHORT_OF_BIN = "box short of the bin"
PAST_BIN = "box past the bin"
ON_RIM = "box on the rim"
STILL_MOVING = "box still moving"
RECORDING_ERROR = "recording error"
CATEGORIES = {
    "robot_fallen": ROBOT_FELL,
    "joint_limit": JOINT_LIMIT,
    "penetration": PENETRATION,
    "short_of_bin": SHORT_OF_BIN,
    "past_bin": PAST_BIN,
    "on_rim": ON_RIM,
    "still_moving": STILL_MOVING,
    "recording_error": RECORDING_ERROR,
}
SEPARATOR = "; "


def validate(gt: GroundTruth) -> Validation:
    """SPEC §3.4: the box at rest on the bin's floor with its centre inside the walls; no fall, joint-limit violation
    or penetration.

    The reason lists every problem in words, with the direction of a miss but no measurements, because the teacher
    reads it and ground truth must not reach the teacher (SPEC §2 rule 1).
    """
    dx, dy = gt.box_xyz[0] - gt.bin_xy[0], gt.box_xyz[1] - gt.bin_xy[1]
    c, s = math.cos(gt.bin_yaw), math.sin(gt.bin_yaw)
    # Both simulators turn the bin to face the robot, so its x axis points away from the robot.
    away, across = c * dx + s * dy, -s * dx + c * dy
    over_bin = max(abs(away), abs(across)) <= gt.bin_inner_half
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
        problems.append(ON_RIM if over_bin else SHORT_OF_BIN if away < 0 else PAST_BIN)
    if not gt.box_at_rest:
        problems.append(STILL_MOVING)
    return Validation(
        task_success=inside and gt.box_at_rest,
        physics_valid=not (gt.robot_fallen or gt.joint_limit_violation or gt.penetration),
        reason=SEPARATOR.join(problems) or "box at rest inside bin",
    )


def recording_error_verdict(error: Exception) -> Validation:
    """The verdict for an episode the recorder refused: a simulator bug, never a lesson, counted apart from misses."""
    return Validation(task_success=False, physics_valid=False, reason=f"{RECORDING_ERROR}: {error}")


def failure_categories(reason: str) -> list[str]:
    """The problem kinds named in a verdict's reason, in the order `validate` lists them."""
    parts = reason.split(SEPARATOR)
    return [category for category, marker in CATEGORIES.items() if any(part.startswith(marker) for part in parts)]
