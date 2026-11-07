import math

from isaacs_mom.contracts import GroundTruth
from isaacs_mom.validator.rules import failure_categories, validate


def _truth(
    box_x: float = 0.0,
    *,
    on_floor: bool = True,
    at_rest: bool = True,
    bin_yaw: float = 0.0,
    robot_fallen: bool = False,
    joint_limit_violation: bool = False,
    penetration: bool = False,
) -> GroundTruth:
    # A bin at the origin keeps the offsets exact in floating point.
    return GroundTruth(
        box_xyz=(box_x, 0.0, 0.445),
        bin_xy=(0.0, 0.0),
        bin_yaw=bin_yaw,
        bin_inner_half=0.22,
        box_on_bin_floor=on_floor,
        box_at_rest=at_rest,
        robot_fallen=robot_fallen,
        joint_limit_violation=joint_limit_violation,
        penetration=penetration,
    )


def test_box_at_rest_on_the_bin_floor_is_accepted() -> None:
    verdict = validate(_truth(0.21))
    assert verdict.accepted is True
    assert verdict.reason == "box at rest inside bin"


def test_box_off_the_bin_floor_fails_with_its_offset_in_the_reason() -> None:
    verdict = validate(_truth(0.30, on_floor=False))
    assert verdict.task_success is False
    assert "30 cm from its centre" in verdict.reason


def test_box_touching_the_floor_with_its_centre_outside_the_walls_fails() -> None:
    verdict = validate(_truth(0.30, on_floor=True))
    assert verdict.accepted is False
    assert "30 cm from its centre" in verdict.reason


def test_offset_is_measured_in_the_turned_bins_frame() -> None:
    # 20 cm along x is 14 cm along each side of a bin turned by 45 degrees.
    verdict = validate(_truth(0.20, on_floor=False, bin_yaw=math.pi / 4))
    assert "14 cm from its centre" in verdict.reason


def test_moving_box_in_the_bin_is_not_success() -> None:
    verdict = validate(_truth(at_rest=False))
    assert verdict.task_success is False
    assert verdict.reason == "box inside bin but still moving"


def test_fallen_robot_is_physics_invalid() -> None:
    verdict = validate(_truth(robot_fallen=True))
    assert verdict.physics_valid is False
    assert verdict.reason == "robot fell"
    assert verdict.accepted is False


def test_joint_limit_violation_is_physics_invalid() -> None:
    verdict = validate(_truth(joint_limit_violation=True))
    assert verdict.physics_valid is False
    assert verdict.reason == "joint limit violated"
    assert verdict.accepted is False


def test_penetration_is_physics_invalid() -> None:
    verdict = validate(_truth(penetration=True))
    assert verdict.physics_valid is False
    assert verdict.reason == "objects penetrated each other"


def test_every_problem_is_listed_in_order() -> None:
    verdict = validate(
        _truth(0.30, on_floor=False, at_rest=False, robot_fallen=True, joint_limit_violation=True, penetration=True)
    )
    problems = verdict.reason.split("; ")
    assert problems[:3] == ["robot fell", "joint limit violated", "objects penetrated each other"]
    assert problems[3].startswith("box not in bin: 30 cm from its centre (inner half-width 22 cm)")
    assert problems[4] == "box still moving"
    assert len(problems) == 5


def test_failure_categories_name_each_problem() -> None:
    every_problem = validate(
        _truth(0.30, on_floor=False, at_rest=False, robot_fallen=True, joint_limit_violation=True, penetration=True)
    )
    assert failure_categories(every_problem.reason) == [
        "robot_fallen",
        "joint_limit",
        "penetration",
        "missed_bin",
        "still_moving",
    ]
    assert failure_categories(validate(_truth(at_rest=False)).reason) == ["still_moving"]
    assert failure_categories(validate(_truth()).reason) == []
