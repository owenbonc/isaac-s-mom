import math

from isaacs_mom.contracts import GroundTruth, Validation

JOINT_LIMIT = "joint limit violated"
TABLE_PENETRATION = "block penetrated the table"
MISSED_TARGET = "cm from target centre"
STILL_MOVING = "still moving"
CATEGORIES = {
    "joint_limit": JOINT_LIMIT,
    "table_penetration": TABLE_PENETRATION,
    "missed_target": MISSED_TARGET,
    "still_moving": STILL_MOVING,
}
SEPARATOR = "; "


def validate(gt: GroundTruth) -> Validation:
    """SPEC §3.4. The reason lists every problem, so the teacher and dashboard see all of them."""
    distance = math.dist(gt.block_xy, gt.target_xy)
    inside = distance <= gt.target_radius
    problems: list[str] = []
    if gt.joint_limit_violation:
        problems.append(JOINT_LIMIT)
    if gt.table_penetration:
        problems.append(TABLE_PENETRATION)
    if not inside:
        problems.append(f"block {distance * 100:.1f} {MISSED_TARGET} (radius {gt.target_radius * 100:.1f} cm)")
    if not gt.block_at_rest:
        problems.append(f"block inside target but {STILL_MOVING}" if inside else f"block {STILL_MOVING}")
    return Validation(
        task_success=inside and gt.block_at_rest,
        physics_valid=not (gt.joint_limit_violation or gt.table_penetration),
        reason=SEPARATOR.join(problems) or "block at rest inside target",
    )


def failure_categories(reason: str) -> list[str]:
    """The problem kinds named in a verdict's reason, in the order `validate` lists them."""
    parts = reason.split(SEPARATOR)
    return [category for category, marker in CATEGORIES.items() if any(marker in part for part in parts)]
