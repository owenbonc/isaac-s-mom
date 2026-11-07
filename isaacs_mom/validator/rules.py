import math

from isaacs_mom.contracts import GroundTruth, Validation


def validate(gt: GroundTruth) -> Validation:
    """SPEC §3.4: the box at rest on the bin's floor, with no fall, joint-limit violation or penetration."""
    inside = gt.box_on_bin_floor
    physics_valid = not (gt.robot_fallen or gt.joint_limit_violation or gt.penetration)
    task_success = inside and gt.box_at_rest
    if not physics_valid:
        reason = (
            "robot fell"
            if gt.robot_fallen
            else "joint limit violated"
            if gt.joint_limit_violation
            else "objects penetrated each other"
        )
    elif not inside:
        dx, dy = gt.box_xyz[0] - gt.bin_xy[0], gt.box_xyz[1] - gt.bin_xy[1]
        c, s = math.cos(gt.bin_yaw), math.sin(gt.bin_yaw)
        offset = max(abs(c * dx + s * dy), abs(-s * dx + c * dy))
        reason = (
            f"box not in bin: {offset * 100:.0f} cm from its centre"
            f" (inner half-width {gt.bin_inner_half * 100:.0f} cm), {gt.box_xyz[2] * 100:.0f} cm high"
        )
    elif not gt.box_at_rest:
        reason = "box inside bin but still moving"
    else:
        reason = "box at rest inside bin"
    return Validation(task_success=task_success, physics_valid=physics_valid, reason=reason)
