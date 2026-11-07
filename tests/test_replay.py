import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from isaacs_mom.contracts import Controller, EpisodeManifest, Observation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.validator import replay


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


def _load(path: Path) -> EpisodeManifest:
    return EpisodeManifest.model_validate_json(path.read_text())


def test_replay_reproduces_an_accepted_episode_and_marks_the_manifest(tmp_path: Path) -> None:
    path = _record(tmp_path, GreedyHugger, seed=1600)
    assert _load(path).validation.accepted  # type: ignore[union-attr]
    assert replay(FakeSim(), path)
    assert _load(path).replay_valid is True


def test_replay_reproduces_a_rejected_episode(tmp_path: Path) -> None:
    path = _record(tmp_path, lambda _: ReobserveOnce(), seed=1601)
    assert not _load(path).validation.accepted  # type: ignore[union-attr]
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
