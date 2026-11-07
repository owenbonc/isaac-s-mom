from isaacs_mom.contracts import GroundTruth
from isaacs_mom.validator.rules import failure_categories, validate


def _truth(
    block_x: float,
    *,
    at_rest: bool = True,
    joint_limit_violation: bool = False,
    table_penetration: bool = False,
) -> GroundTruth:
    # A target at the origin keeps the edge distance exact in floating point.
    return GroundTruth(
        block_xy=(block_x, 0.0),
        target_xy=(0.0, 0.0),
        target_radius=0.05,
        block_at_rest=at_rest,
        joint_limit_violation=joint_limit_violation,
        table_penetration=table_penetration,
    )


def test_block_on_the_edge_counts_as_inside() -> None:
    assert validate(_truth(0.05)).task_success is True


def test_block_just_outside_fails_with_distance_in_reason() -> None:
    verdict = validate(_truth(0.051))
    assert verdict.task_success is False
    assert "cm from target centre" in verdict.reason


def test_moving_block_inside_is_not_success() -> None:
    verdict = validate(_truth(0.0, at_rest=False))
    assert verdict.task_success is False
    assert verdict.reason == "block inside target but still moving"


def test_joint_limit_violation_is_physics_invalid() -> None:
    verdict = validate(_truth(0.0, joint_limit_violation=True))
    assert verdict.physics_valid is False
    assert verdict.reason == "joint limit violated"
    assert verdict.accepted is False


def test_table_penetration_is_physics_invalid() -> None:
    verdict = validate(_truth(0.0, table_penetration=True))
    assert verdict.physics_valid is False
    assert verdict.reason == "block penetrated the table"


def test_every_problem_is_listed_in_order() -> None:
    verdict = validate(_truth(0.1, at_rest=False, joint_limit_violation=True, table_penetration=True))
    assert verdict.reason.split("; ") == [
        "joint limit violated",
        "block penetrated the table",
        "block 10.0 cm from target centre (radius 5.0 cm)",
        "block still moving",
    ]


def test_failure_categories_name_each_problem() -> None:
    every_problem = validate(_truth(0.1, at_rest=False, joint_limit_violation=True, table_penetration=True))
    assert failure_categories(every_problem.reason) == [
        "joint_limit",
        "table_penetration",
        "missed_target",
        "still_moving",
    ]
    assert failure_categories(validate(_truth(0.0, at_rest=False)).reason) == ["still_moving"]
    assert failure_categories(validate(_truth(0.0)).reason) == []
