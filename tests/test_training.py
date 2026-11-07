from pathlib import Path

import pytest

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    SkillCall,
    StepRecord,
    Validation,
)
from isaacs_mom.seeds import DEV_SEED_START, SealedSeedError, heldout_seeds
from isaacs_mom.validator.training import training_examples

ACCEPTED = Validation(task_success=True, physics_valid=True, reason="ok")


def step(index: int, controller: Controller) -> StepRecord:
    return StepRecord(
        step=index,
        sim_time=float(index),
        frame=f"frames/{index:04d}.jpg",
        joint_state=[0.0],
        skill_call=SkillCall(skill="reach_left", params={"distance_cm": 5.0}),
        controller=controller,
        checkpoint=None,
        actions=f"actions/{index:04d}.npy",
        timestamps=[0.0],
        ok=True,
    )


def manifest(
    validation: Validation | None = ACCEPTED,
    seed: int = DEV_SEED_START,
    controllers: tuple[Controller, ...] = ("teacher",),
) -> EpisodeManifest:
    return EpisodeManifest(
        episode_id="ep",
        run_id="run",
        seed=seed,
        split="dev",
        round=0,
        instruction="hug the box into the bin",
        sim_version="test",
        robot_config="test",
        steps=[step(i, c) for i, c in enumerate(controllers)],
        validation=validation,
    )


def test_heldout_seed_labelled_dev_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SealedSeedError):
        training_examples(tmp_path, manifest(seed=min(heldout_seeds())))


def test_episode_without_validation_gives_no_examples(tmp_path: Path) -> None:
    assert training_examples(tmp_path, manifest(validation=None)) == []


def test_student_steps_are_excluded(tmp_path: Path) -> None:
    examples = training_examples(tmp_path, manifest(controllers=("student", "teacher", "teacher")))
    assert [e.image for e in examples] == [
        str(tmp_path / "frames/0001.jpg"),
        str(tmp_path / "frames/0002.jpg"),
    ]


def test_rejected_teacher_episode_gives_no_examples(tmp_path: Path) -> None:
    rejected = Validation(task_success=False, physics_valid=True, reason="x")
    assert training_examples(tmp_path, manifest(validation=rejected)) == []


def test_physics_invalid_episode_gives_no_examples(tmp_path: Path) -> None:
    invalid = Validation(task_success=True, physics_valid=False, reason="x")
    assert training_examples(tmp_path, manifest(validation=invalid)) == []
