from pathlib import Path

import pytest

from isaacs_mom.contracts import EpisodeManifest, ExecResult, Observation, Perturbation, SkillCall, Validation
from isaacs_mom.validator.recorder import Recorder, RecordingError


def _start(run_dir: Path) -> Recorder:
    return Recorder.start(
        run_dir,
        episode_id="ep",
        run_id="run",
        seed=1000,
        split="dev",
        round_=0,
        instruction="push",
        sim_version="test",
        robot_config="test",
    )


def _obs(step: int) -> Observation:
    return Observation(episode_id="ep", step=step, sim_time=0.1 * step, head_rgb_jpeg=b"jpg", joint_state=[0.0])


def _result(actions: int = 2, timestamps: tuple[float, ...] = (0.02, 0.04)) -> ExecResult:
    return ExecResult(ok=True, executed_actions=[[0.0, 0.0]] * actions, timestamps=list(timestamps))


def _saved(run_dir: Path) -> EpisodeManifest:
    return EpisodeManifest.model_validate_json((run_dir / "episodes" / "ep" / "manifest.json").read_text())


PUSH = SkillCall(skill="push_forward")


def test_diary_is_saved_as_soon_as_the_episode_starts(tmp_path: Path) -> None:
    _start(tmp_path)
    saved = _saved(tmp_path)
    assert saved.steps == []
    assert saved.validation is None


def test_diary_survives_a_crash_before_the_verdict(tmp_path: Path) -> None:
    recorder = _start(tmp_path)
    recorder.add_step(_obs(0), PUSH, _result(), "student", "ckpt-r0")
    recorder.add_perturbation(Perturbation(kind="move_block", seed=7))
    recorder.mark_takeover()
    recorder.add_step(_obs(1), PUSH, _result(), "teacher", None)
    saved = _saved(tmp_path)
    assert [s.controller for s in saved.steps] == ["student", "teacher"]
    assert [p.before_step for p in saved.perturbations] == [1]
    assert saved.takeover_step == 1
    assert saved.validation is None


def test_finish_writes_the_verdict(tmp_path: Path) -> None:
    verdict = Validation(task_success=True, physics_valid=True, reason="block at rest inside target")
    _start(tmp_path).finish(verdict)
    assert _saved(tmp_path).validation == verdict


def test_actions_and_timestamps_must_line_up(tmp_path: Path) -> None:
    recorder = _start(tmp_path)
    with pytest.raises(RecordingError, match="2 executed actions but 1 timestamps"):
        recorder.add_step(_obs(0), PUSH, _result(actions=2, timestamps=(0.02,)), "teacher", None)
    assert _saved(tmp_path).steps == []
    assert list((tmp_path / "episodes" / "ep" / "frames").iterdir()) == []


def test_timestamps_must_not_go_backwards(tmp_path: Path) -> None:
    with pytest.raises(RecordingError, match="timestamps go backwards"):
        _start(tmp_path).add_step(_obs(0), PUSH, _result(timestamps=(0.04, 0.02)), "teacher", None)
