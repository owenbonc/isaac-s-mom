import math
import sys
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import pytest

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    ExecResult,
    Observation,
    ReplayFailure,
    ReplayReport,
    SkillCall,
)
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.validator import replay, replay_accepted


class GreedyHugger:
    """Test-only: lines up, hugs, lifts, turns to the bin and places, using ground truth real policies never see."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        gt = self.sim.ground_truth()
        work_y, waist, closed, lifted = obs.joint_state
        if gt.box_on_bin_floor:
            return None
        if not closed:
            dy = gt.box_xyz[1] - work_y
            if abs(dy) > 0.01:
                return SkillCall(skill="reach_left" if dy > 0 else "reach_right", params={"distance_cm": abs(dy) * 100})
            return SkillCall(skill="hug")
        if not lifted:
            return SkillCall(skill="lift")
        turn = math.degrees(math.atan2(gt.bin_xy[1], gt.bin_xy[0]) - waist)
        if abs(turn) > 2:
            return SkillCall(skill="turn_left" if turn > 0 else "turn_right", params={"angle_deg": abs(turn)})
        return SkillCall(skill="place")


class ReobserveOnce:
    controller: Controller = "student"
    checkpoint: str | None = "ckpt-r0"

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return SkillCall(skill="reobserve") if obs.step == 0 else None


def _record(tmp_path: Path, make_policy: Callable[[FakeSim], Policy], seed: int, perturb_at: int | None = None) -> Path:
    sim = FakeSim()
    manifest = run_episode(
        sim,
        make_policy(sim),
        run_dir=tmp_path,
        events=EventLog(tmp_path, "t"),
        seed=seed,
        split="dev",
        round_=0,
        instruction="hug",
        perturb_at=perturb_at,
    )
    return tmp_path / "episodes" / manifest.episode_id / "manifest.json"


def _accepted(path: Path) -> bool:
    verdict = _load(path).validation
    return verdict is not None and verdict.accepted


def _load(path: Path) -> EpisodeManifest:
    return EpisodeManifest.model_validate_json(path.read_text())


def test_replay_reproduces_an_accepted_episode_and_marks_the_manifest(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1600)
    assert _accepted(path)
    assert replay(FakeSim(), path)
    assert _load(path).replay_valid is True


def test_replay_reproduces_a_rejected_episode(tmp_path: Path) -> None:
    path = _record(tmp_path, lambda _: ReobserveOnce(), seed=1601)
    assert not _accepted(path)
    assert replay(FakeSim(), path)


def test_replay_applies_the_recorded_mid_run_perturbation(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1602, perturb_at=1)
    manifest = _load(path)
    assert [p.before_step for p in manifest.perturbations] == [1]
    assert replay(FakeSim(), path)

    manifest.perturbations = []
    path.write_text(manifest.model_dump_json())
    assert not replay(FakeSim(), path)


def test_replay_rejects_tampered_actions(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1603)
    first = path.parent / _load(path).steps[0].actions
    np.save(first, np.zeros_like(np.load(first)))
    assert not replay(FakeSim(), path)
    assert _load(path).replay_valid is False


class StopImmediately:
    controller: Controller = "student"
    checkpoint: str | None = "ckpt-r0"

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return None


def test_replay_of_a_zero_step_episode_reproduces(tmp_path: Path) -> None:
    path = _record(tmp_path, lambda _: StopImmediately(), seed=1604)
    assert _load(path).steps == []
    assert replay(FakeSim(), path)


def test_replay_without_validation_is_false(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1605)
    manifest = _load(path)
    manifest.validation = None
    path.write_text(manifest.model_dump_json())
    assert not replay(FakeSim(), path)


def test_replay_accepted_checks_every_accepted_episode_in_a_run(tmp_path: Path) -> None:
    good = _record(tmp_path, GreedyHugger, seed=1606)
    _record(tmp_path, lambda _: ReobserveOnce(), seed=1607)  # rejected, so not checked
    tampered = _record(tmp_path, GreedyHugger, seed=1608)
    assert all(_accepted(p) for p in (good, tampered))
    first = tampered.parent / _load(tampered).steps[0].actions
    np.save(first, np.zeros_like(np.load(first)))

    report = replay_accepted(FakeSim(), tmp_path)

    assert report == ReplayReport(
        replayed=2,
        reproduced=1,
        failures=[ReplayFailure(episode_id=_load(tampered).episode_id, reason="step 0: executed actions differ")],
    )
    assert ReplayReport.model_validate_json((tmp_path / "replay_report.json").read_text()) == report
    assert _load(good).replay_valid is True


def test_one_broken_episode_does_not_stop_the_whole_run_replay(tmp_path: Path) -> None:
    broken = _record(tmp_path, GreedyHugger, seed=1609)
    good = _record(tmp_path, GreedyHugger, seed=1610)
    (broken.parent / _load(broken).steps[0].actions).unlink()

    report = replay_accepted(FakeSim(), tmp_path)

    assert (report.replayed, report.reproduced) == (2, 1)
    assert [f.episode_id for f in report.failures] == [_load(broken).episode_id]
    assert report.failures[0].reason.startswith("replay raised FileNotFoundError")
    assert _load(broken).replay_valid is False
    assert _load(good).replay_valid is True


def test_replay_notices_a_step_with_fewer_timestamps(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1611)
    manifest = _load(path)
    # Two timestamps against ten neither match nor broadcast, so only the length check gives a clean reason.
    manifest.steps[0].timestamps = manifest.steps[0].timestamps[:2]
    path.write_text(manifest.model_dump_json())

    report = replay_accepted(FakeSim(), tmp_path)

    assert [f.reason for f in report.failures] == ["step 0: timestamps differ"]


def test_an_unreadable_manifest_counts_as_replayed_and_failed(tmp_path: Path) -> None:
    _record(tmp_path, GreedyHugger, seed=1612)
    garbled = tmp_path / "episodes" / "garbled"
    garbled.mkdir()
    (garbled / "manifest.json").write_text("{not json")

    report = replay_accepted(FakeSim(), tmp_path)

    assert (report.replayed, report.reproduced) == (2, 1)
    assert report.failures[0].episode_id == "garbled"
    assert report.failures[0].reason.startswith("manifest unreadable")


def test_a_replay_result_that_cannot_be_saved_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _record(tmp_path, GreedyHugger, seed=1613)

    def read_only(path: Path, manifest: EpisodeManifest) -> None:
        raise PermissionError("read-only folder")

    # The package exports the `replay` function under the module's own name, so patch through sys.modules.
    monkeypatch.setattr(sys.modules["isaacs_mom.validator.replay"], "write_manifest", read_only)

    report = replay_accepted(FakeSim(), tmp_path)

    assert (report.replayed, report.reproduced) == (1, 0)
    assert report.failures[0].reason == "could not save the replay result: read-only folder"


class CrashingSim(FakeSim):
    def execute(self, call: SkillCall) -> ExecResult:
        raise mujoco.FatalError("simulation unstable")  # pyright: ignore[reportAttributeAccessIssue]  (mujoco ships no types)


def test_a_simulator_crash_is_a_failed_replay_not_a_lost_run(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1614)

    report = replay_accepted(CrashingSim(), tmp_path)

    assert (report.replayed, report.reproduced) == (1, 0)
    assert report.failures[0].reason == "replay raised FatalError: simulation unstable"
    assert _load(path).replay_valid is False
