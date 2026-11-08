from pathlib import Path

import pytest

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    SkillCall,
    StepRecord,
    TrainingSetSummary,
    Validation,
)
from isaacs_mom.seeds import DEV_SEED_START, SealedSeedError, heldout_seeds
from isaacs_mom.validator.recorder import RecordingError
from isaacs_mom.validator.rules import recording_error_verdict
from isaacs_mom.validator.training import build_training_set, training_examples

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
    driver: Controller | None = None,
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
        driver=driver or (controllers[0] if controllers else "teacher"),
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


def _write(run_dir: Path, episode_id: str, episode: EpisodeManifest) -> None:
    episode_dir = run_dir / "episodes" / episode_id
    episode_dir.mkdir(parents=True)
    (episode_dir / "manifest.json").write_text(episode.model_copy(update={"episode_id": episode_id}).model_dump_json())


def test_each_training_set_comes_with_a_report_card(tmp_path: Path) -> None:
    missed = Validation(task_success=False, physics_valid=True, reason="box short of the bin")
    unsafe = Validation(task_success=False, physics_valid=False, reason="joint limit violated; box still moving")
    exam = manifest(seed=min(heldout_seeds()), controllers=("student",)).model_copy(update={"split": "heldout"})
    _write(tmp_path, "a-accepted", manifest(controllers=("student", "teacher", "teacher")))
    _write(tmp_path, "b-missed", manifest(validation=missed))
    _write(tmp_path, "c-unsafe", manifest(validation=unsafe))
    _write(tmp_path, "d-unfinished", manifest(validation=None))
    _write(tmp_path, "e-student-trial", manifest(validation=missed, controllers=("student",)))
    _write(tmp_path, "f-exam", exam)
    _write(
        tmp_path, "g-unexplained", manifest(validation=Validation(task_success=False, physics_valid=True, reason="x"))
    )
    _write(tmp_path, "h-empty", manifest(validation=missed, controllers=()))
    # A held-out episode the teacher drove is still an exam, never a demonstration.
    _write(tmp_path, "j-exam-teacher", manifest(seed=min(heldout_seeds())).model_copy(update={"split": "heldout"}))
    _write(
        tmp_path, "i-sim-bug", manifest(validation=recording_error_verdict(RecordingError("timestamps go backwards")))
    )

    training_set = build_training_set(tmp_path, 1)

    summary = TrainingSetSummary.model_validate_json(training_set.with_suffix(".summary.json").read_text())
    assert summary == TrainingSetSummary(
        round=1,
        episodes_scanned=10,
        demonstrations=7,
        accepted=1,
        rejected=5,
        unfinished=1,
        rejected_by={"short_of_bin": 2, "joint_limit": 1, "still_moving": 1, "other": 1, "recording_error": 1},
        empty_dev_episodes=1,
        examples=2,
    )
    assert len(training_set.read_text().splitlines()) == summary.examples


def test_a_teacher_refused_at_its_first_step_is_a_rejected_demonstration(tmp_path: Path) -> None:
    refused = recording_error_verdict(RecordingError("timestamps go backwards"))
    _write(tmp_path, "refused-at-once", manifest(validation=refused, controllers=(), driver="teacher"))

    summary = TrainingSetSummary.model_validate_json(
        build_training_set(tmp_path, 1).with_suffix(".summary.json").read_text()
    )

    assert (summary.demonstrations, summary.rejected) == (1, 1)
    assert summary.rejected_by == {"recording_error": 1}
    assert summary.empty_dev_episodes == 1


def test_a_students_recording_error_is_not_counted_as_a_demonstration(tmp_path: Path) -> None:
    refused = recording_error_verdict(RecordingError("timestamps go backwards"))
    _write(tmp_path, "student-trial", manifest(validation=refused, controllers=(), driver="student"))

    summary = TrainingSetSummary.model_validate_json(
        build_training_set(tmp_path, 1).with_suffix(".summary.json").read_text()
    )

    assert (summary.demonstrations, summary.rejected) == (0, 0)
    assert summary.rejected_by == {}
    assert summary.empty_dev_episodes == 1


def ending_with(last: Controller) -> EpisodeManifest:
    """An accepted demo whose final step is a `reobserve` by `last`."""
    episode = manifest(controllers=("teacher", "teacher", last))
    closing = episode.steps[-1].model_copy(update={"skill_call": SkillCall(skill="reobserve")})
    return episode.model_copy(update={"steps": [*episode.steps[:-1], closing]})


def test_a_demo_ending_with_the_teachers_reobserve_also_teaches_the_second_one(tmp_path: Path) -> None:
    examples = training_examples(tmp_path, ending_with("teacher"))
    assert [e.target.skill for e in examples] == ["reach_left", "reach_left", "reobserve", "reobserve"]
    closing = examples[-1]
    # The student's done rule is a second reobserve in a row (SPEC §3.2), so the last example shows exactly that.
    assert closing.feedback == "reobserve: ok"
    assert closing.image == examples[-2].image
    assert closing.target.params == {}


def test_a_demo_that_doesnt_end_with_the_teachers_reobserve_gets_no_closing_example(tmp_path: Path) -> None:
    assert [e.feedback for e in training_examples(tmp_path, manifest(controllers=("teacher", "teacher")))] == [
        "start",
        "reach_left: ok",
    ]
    assert all(e.feedback != "reobserve: ok" for e in training_examples(tmp_path, ending_with("student")))
