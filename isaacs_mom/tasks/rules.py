"""Judging a generated task's success rule from simulator ground truth (SPEC §3.7).

Only the validator and orchestrator side calls this (rule 1). As in the validator, the reason names every problem in
words and without measurements, and a fall, a joint-limit violation or a penetration always fails physics.
"""

import math

from isaacs_mom.contracts import GroundTruth, Validation
from isaacs_mom.sim.mujoco.scene import BIN_FLOOR_Z, BIN_WALL
from isaacs_mom.sim.mujoco.task import BIN_INNER_HALF
from isaacs_mom.tasks.spec import AtRest, Inside, On, Platform, RobotStanding, TaskSpec

SEPARATOR = "; "


def judge(spec: TaskSpec, gt: GroundTruth) -> Validation:
    dest = spec.scene.destination
    # The bin's floor and a platform's top are the same geom in the scene, so this is contact with either.
    touching = gt.box_on_bin_floor
    problems: list[str] = []
    for fact in spec.success.all_of:
        match fact:
            case Inside():
                if not (touching and _centre_within(gt, BIN_INNER_HALF)):
                    problems.append(f"{fact.object} not inside the {dest.kind}")
            case On():
                top = dest.top_m if isinstance(dest, Platform) else BIN_FLOOR_Z
                # Centre above the top and over it: resting on it, not leaning on its edge from the floor.
                if not (touching and gt.box_xyz[2] > top and _centre_within(gt, BIN_INNER_HALF + BIN_WALL)):
                    problems.append(f"{fact.object} not on the {dest.kind}")
            case AtRest():
                if not gt.box_at_rest:
                    problems.append(f"{fact.object} still moving")
            case RobotStanding():
                pass  # a fall fails physics below, as in every task
    physics = [
        message
        for flagged, message in (
            (gt.robot_fallen, "robot fell"),
            (gt.joint_limit_violation, "joint limit violated"),
            (gt.penetration, "penetration"),
        )
        if flagged
    ]
    standing = not (gt.robot_fallen and any(isinstance(f, RobotStanding) for f in spec.success.all_of))
    return Validation(
        task_success=not problems and standing,
        physics_valid=not physics,
        reason=SEPARATOR.join(physics + problems) or f"{spec.name}: every rule holds",
    )


def _centre_within(gt: GroundTruth, half: float) -> bool:
    """The box's centre within `half` of the destination's centre along both of its axes."""
    dx, dy = gt.box_xyz[0] - gt.bin_xy[0], gt.box_xyz[1] - gt.bin_xy[1]
    c, s = math.cos(gt.bin_yaw), math.sin(gt.bin_yaw)
    return max(abs(c * dx + s * dy), abs(-s * dx + c * dy)) <= half
