from collections.abc import Callable
from pathlib import Path

import numpy as np

from isaacs_mom.contracts import Controller, EpisodeManifest, Observation, ReplayReport, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.validator import replay, replay_accepted


class GreedyPusher:
    """Test-only: lines up behind the block using ground truth, then pushes. Real policies never see ground truth."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        gt = self.sim.ground_truth()
        dy = gt.block_xy[1] - obs.joint_state[1]
        if abs(dy) > 0.01:
            return SkillCall(
                skill="approach_right" if dy > 0 else "approach_left", params={"distance_cm": abs(dy) * 100}
            )
        gap = gt.target_xy[0] - gt.block_xy[0]
        if gap > 0.01:
            behind = gt.block_xy[0] - 0.025 - obs.joint_state[0]
            return SkillCall(skill="push_forward", params={"distance_cm": min((behind + gap) * 100, 20)})
        return None


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
        instruction="push",
        perturb_at=perturb_at,
    )
    return tmp_path / "episodes" / manifest.episode_id / "manifest.json"


def _load(path: Path) -> EpisodeManifest:
    return EpisodeManifest.model_validate_json(path.read_text())


def test_replay_reproduces_an_accepted_episode_and_marks_the_manifest(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyPusher, seed=1600)
    assert _load(path).validation.accepted  # type: ignore[union-attr]
    assert replay(FakeSim(), path)
    assert _load(path).replay_valid is True


def test_replay_reproduces_a_rejected_episode(tmp_path: Path) -> None:
    path = _record(tmp_path, lambda _: ReobserveOnce(), seed=1601)
    assert not _load(path).validation.accepted  # type: ignore[union-attr]
    assert replay(FakeSim(), path)


def test_replay_applies_the_recorded_mid_run_perturbation(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyPusher, seed=1602, perturb_at=1)
    manifest = _load(path)
    assert [p.before_step for p in manifest.perturbations] == [1]
    assert replay(FakeSim(), path)

    manifest.perturbations = []
    path.write_text(manifest.model_dump_json())
    assert not replay(FakeSim(), path)


def test_replay_rejects_tampered_actions(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyPusher, seed=1603)
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
    path = _record(tmp_path, GreedyPusher, seed=1605)
    manifest = _load(path)
    manifest.validation = None
    path.write_text(manifest.model_dump_json())
    assert not replay(FakeSim(), path)


def test_replay_accepted_checks_every_accepted_episode_in_a_run(tmp_path: Path) -> None:
    good = _record(tmp_path, GreedyPusher, seed=1606)
    _record(tmp_path, lambda _: ReobserveOnce(), seed=1607)  # rejected, so not checked
    tampered = _record(tmp_path, GreedyPusher, seed=1608)
    assert _load(good).validation.accepted and _load(tampered).validation.accepted  # type: ignore[union-attr]
    first = tampered.parent / _load(tampered).steps[0].actions
    np.save(first, np.zeros_like(np.load(first)))

    report = replay_accepted(FakeSim(), tmp_path)

    assert report == ReplayReport(checked=2, reproduced=1, not_reproduced=[_load(tampered).episode_id])
    assert ReplayReport.model_validate_json((tmp_path / "replay_report.json").read_text()) == report
    assert _load(good).replay_valid is True
