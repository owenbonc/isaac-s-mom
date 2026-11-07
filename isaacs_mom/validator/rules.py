import math

from isaacs_mom.contracts import GroundTruth, Validation


def validate(gt: GroundTruth) -> Validation:
    """SPEC §3.4. Workstream 3 adds the 1 s at-rest window from the real MuJoCo scene."""
    distance = math.dist(gt.block_xy, gt.target_xy)
    inside = distance <= gt.target_radius
    physics_valid = not (gt.joint_limit_violation or gt.table_penetration)
    task_success = inside and gt.block_at_rest
    if not physics_valid:
        reason = "joint limit violated" if gt.joint_limit_violation else "block penetrated the table"
    elif not inside:
        reason = f"block {distance * 100:.1f} cm from target centre (radius {gt.target_radius * 100:.1f} cm)"
    elif not gt.block_at_rest:
        reason = "block inside target but still moving"
    else:
        reason = "block at rest inside target"
    return Validation(task_success=task_success, physics_valid=physics_valid, reason=reason)
