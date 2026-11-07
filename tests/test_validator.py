import math

from isaacs_mom.contracts import GroundTruth
from isaacs_mom.validator.recorder import RecordingError
from isaacs_mom.validator.rules import failure_categories, recording_error_verdict, validate


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
    # A bin at the origin keeps the offsets exact; with bin_yaw 0 its x axis, away from the robot, is world +x.
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


def test_box_beyond_the_bin_is_past_it() -> None:
    verdict = validate(_truth(0.30, on_floor=False))
    assert verdict.task_success is False
    assert verdict.reason == "box past the bin"


def test_box_between_the_robot_and_the_bin_is_short_of_it() -> None:
    assert validate(_truth(-0.30, on_floor=False)).reason == "box short of the bin"


def test_box_over_the_bin_but_off_its_floor_is_on_the_rim() -> None:
    verdict = validate(_truth(0.10, on_floor=False))
    assert verdict.task_success is False
    assert verdict.reason == "box on the rim"


def test_box_touching_the_floor_with_its_centre_outside_the_walls_fails() -> None:
    verdict = validate(_truth(0.30, on_floor=True))
    assert verdict.accepted is False
    assert verdict.reason == "box past the bin"


def test_offset_is_measured_in_the_turned_bins_frame() -> None:
    # 25 cm along x is about 18 cm along each side of a bin turned by 45 degrees, so it is inside.
    assert validate(_truth(0.25, bin_yaw=math.pi / 4)).accepted is True
    assert validate(_truth(0.25)).accepted is False


def test_short_and_past_follow_the_bins_turn() -> None:
    # Turned half a circle, the bin's away-from-the-robot axis is world -x.
    assert validate(_truth(0.30, on_floor=False, bin_yaw=math.pi)).reason == "box short of the bin"


def test_moving_box_in_the_bin_is_not_success() -> None:
    verdict = validate(_truth(at_rest=False))
    assert verdict.task_success is False
    assert verdict.reason == "box still moving"


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
    assert verdict.reason == "penetration"


def test_every_problem_is_listed_in_order() -> None:
    verdict = validate(
        _truth(0.30, on_floor=False, at_rest=False, robot_fallen=True, joint_limit_violation=True, penetration=True)
    )
    assert verdict.reason.split("; ") == [
        "robot fell",
        "joint limit violated",
        "penetration",
        "box past the bin",
        "box still moving",
    ]


def test_reasons_carry_no_measurements() -> None:
    # SPEC §2 rule 1: the teacher reads the reason, so it must not leak distances or heights from ground truth.
    truths = [
        _truth(0.30, on_floor=False),
        _truth(-0.30, on_floor=False),
        _truth(0.10, on_floor=False),
        _truth(0.30, on_floor=False, at_rest=False, robot_fallen=True, joint_limit_violation=True, penetration=True),
    ]
    assert not any(char.isdigit() for gt in truths for char in validate(gt).reason)


def test_failure_categories_name_each_problem() -> None:
    every_problem = validate(
        _truth(0.30, on_floor=False, at_rest=False, robot_fallen=True, joint_limit_violation=True, penetration=True)
    )
    assert failure_categories(every_problem.reason) == [
        "robot_fallen",
        "joint_limit",
        "penetration",
        "past_bin",
        "still_moving",
    ]
    assert failure_categories(validate(_truth(-0.30, on_floor=False)).reason) == ["short_of_bin"]
    assert failure_categories(validate(_truth(0.10, on_floor=False)).reason) == ["on_rim"]
    assert failure_categories(validate(_truth()).reason) == []


def test_a_recording_error_is_its_own_rejection_kind() -> None:
    verdict = recording_error_verdict(RecordingError("2 executed actions but 1 timestamps"))
    assert verdict.accepted is False
    assert verdict.reason == "recording error: 2 executed actions but 1 timestamps"
    assert failure_categories(verdict.reason) == ["recording_error"]
