import math

from isaacs_mom.contracts import GroundTruth, Validation

ROBOT_FELL = "robot fell"
JOINT_LIMIT = "joint limit violated"
PENETRATION = "objects penetrated each other"
MISSED_BIN = "box not in bin"
STILL_MOVING = "still moving"
CATEGORIES = {
    "robot_fallen": ROBOT_FELL,
    "joint_limit": JOINT_LIMIT,
    "penetration": PENETRATION,
    "missed_bin": MISSED_BIN,
    "still_moving": STILL_MOVING,
}
SEPARATOR = "; "


def validate(gt: GroundTruth) -> Validation:
    """SPEC §3.4: the box at rest on the bin's floor with its centre inside the walls; no fall, joint-limit violation
    or penetration. The reason lists every problem, so the teacher and dashboard see all of them."""
    dx, dy = gt.box_xyz[0] - gt.bin_xy[0], gt.box_xyz[1] - gt.bin_xy[1]
    c, s = math.cos(gt.bin_yaw), math.sin(gt.bin_yaw)
    offset = max(abs(c * dx + s * dy), abs(-s * dx + c * dy))
    inside = gt.box_on_bin_floor and offset <= gt.bin_inner_half
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
        problems.append(
            f"{MISSED_BIN}: {offset * 100:.0f} cm from its centre"
            f" (inner half-width {gt.bin_inner_half * 100:.0f} cm), {gt.box_xyz[2] * 100:.0f} cm high"
        )
    if not gt.box_at_rest:
        problems.append(f"box inside bin but {STILL_MOVING}" if inside else f"box {STILL_MOVING}")
    return Validation(
        task_success=inside and gt.box_at_rest,
        physics_valid=not (gt.robot_fallen or gt.joint_limit_violation or gt.penetration),
        reason=SEPARATOR.join(problems) or "box at rest inside bin",
    )


def failure_categories(reason: str) -> list[str]:
    """The problem kinds named in a verdict's reason, in the order `validate` lists them."""
    parts = reason.split(SEPARATOR)
    return [category for category, marker in CATEGORIES.items() if any(marker in part for part in parts)]
