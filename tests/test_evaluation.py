"""The student evaluation report (SPEC §6 criterion 10): student only, on the sealed held-out seeds, successes out of
(n - errored) per condition, every episode recorded with its latency and device, and nothing claimed for a condition
that wasn't run (rule 6)."""

import json
import time
from pathlib import Path
from typing import Any

import pytest

from isaacs_mom.contracts import EpisodeManifest, GroundTruth, Observation, SkillCall
from isaacs_mom.orchestrator import evaluation
from isaacs_mom.orchestrator.evaluation import (
    CONDITIONS,
    NOT_RUN,
    EpisodeRecord,
    evaluation_report,
    markdown,
    summarise,
)
from isaacs_mom.policy import Policy
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.sim.fake import BIN_Z, FakeSim
from isaacs_mom.student import StubStudent
from isaacs_mom.validator.recorder import RecordingError
from isaacs_mom.validator.rules import recording_error_verdict

SEEDS = [1, 2, 3, 4]


class RiggedSim(FakeSim):
    """FakeSim whose box ends in the bin on even seeds, so the report's counts are known in advance."""

    def __init__(self, task: str) -> None:
        self.sim_version = f"fake-rigged task={task}"
        super().__init__()

    def reset(self, seed: int) -> str:
        self.seed = seed
        return super().reset(seed)

    def ground_truth(self) -> GroundTruth:
        gt = super().ground_truth()
        if self.seed % 2:
            return gt
        return gt.model_copy(update={"box_xyz": (gt.bin_xy[0], gt.bin_xy[1], BIN_Z), "box_on_bin_floor": True})


class CrashesOnSeed3(StubStudent):
    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.episode_id.startswith("ep-3-"):
            raise RuntimeError("the model crashed")
        return super().act(obs, instruction, feedback)


class TakesTwoMs(StubStudent):
    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        time.sleep(0.002)
        return super().act(obs, instruction, feedback)


def _run(tmp_path: Path, make_student: type[Policy] | type[StubStudent] = StubStudent, budget_ms: float = 1000.0):
    asked: list[str] = []

    def make_sim(task: str) -> TruthSim:
        asked.append(task)
        return RiggedSim(task)

    reports = evaluation_report(
        ["ckpt-r0", "run/ckpt-r1"],
        make_sim=make_sim,
        make_student=make_student,  # type: ignore[arg-type]
        run_dir=tmp_path,
        seeds=SEEDS,
        device="test-device",
        budget_ms=budget_ms,
    )
    return reports, asked


def test_each_condition_scores_successes_out_of_n_minus_errored(tmp_path: Path) -> None:
    reports, _ = _run(tmp_path, CrashesOnSeed3)
    first = next(r for r in reports if r.checkpoint == "ckpt-r0" and r.condition == "training_distribution")
    assert (first.n, first.errored, first.successes, first.scored) == (4, 1, 2, 3)
    crashed = json.loads((tmp_path / "evaluation.jsonl").read_text().splitlines()[2])
    assert crashed["seed"] == 3 and crashed["outcome"] == "errored" and crashed["episode_id"].startswith("ep-3-")
    assert (tmp_path / "episodes" / crashed["episode_id"] / "manifest.json").exists()  # its partial recording
    assert first.score() == "2/3"
    layout = next(r for r in reports if r.checkpoint == "ckpt-r0" and r.condition == "new_layout")
    assert (layout.n, layout.errored, layout.successes) == (8, 2, 4)  # near_bin and far_bin, 4 seeds each


class BrokenOnSeed1(RiggedSim):
    def reset(self, seed: int) -> str:
        if seed == 1:
            raise ValueError("no bin pose clears the table")
        return super().reset(seed)


def test_a_scene_that_fails_to_reset_is_errored_and_the_report_goes_on(tmp_path: Path) -> None:
    reports = evaluation_report(
        ["ckpt-r0"],
        make_sim=BrokenOnSeed1,
        make_student=StubStudent,
        run_dir=tmp_path,
        seeds=SEEDS,
        device="test-device",
    )
    first = reports[0]
    assert (first.n, first.errored, first.successes) == (4, 1, 2)
    record = json.loads((tmp_path / "evaluation.jsonl").read_text().splitlines()[0])
    assert record["outcome"] == "errored" and record["reason"] == "ValueError: no bin pose clears the table"
    assert record["latencies_ms"] == []


def test_every_checkpoint_runs_the_same_tasks_and_seeds(tmp_path: Path) -> None:
    _, asked = _run(tmp_path)
    assert asked == ["across_room", "near_bin", "far_bin"]  # each scene built once, reused by every checkpoint
    records = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
    scenarios = {ckpt: {(r["condition"], r["task"], r["seed"]) for r in records if r["checkpoint"] == ckpt}
                 for ckpt in ("ckpt-r0", "run/ckpt-r1")}  # fmt: skip
    assert scenarios["ckpt-r0"] == scenarios["run/ckpt-r1"] and len(scenarios["ckpt-r0"]) == 12
    assert {r["device"] for r in records} == {"test-device"}
    assert {r["sim_version"] for r in records} == {f"fake-rigged task={t}" for t in asked}


def test_a_condition_that_cannot_run_yet_is_reported_as_not_run(tmp_path: Path) -> None:
    reports, _ = _run(tmp_path)
    pending = [r for r in reports if r.condition == "move_bin_after_lift"]
    assert len(pending) == 2 and all(r.n == 0 and r.score() == NOT_RUN["move_bin_after_lift"] for r in pending)
    records = (tmp_path / "evaluation.jsonl").read_text()
    assert "move_bin_after_lift" not in records


def test_episodes_are_student_only_and_held_out(tmp_path: Path) -> None:
    _run(tmp_path)
    manifests = [json.loads(p.read_text()) for p in (tmp_path / "episodes").glob("*/manifest.json")]
    assert len(manifests) == 2 * 12
    assert {m["split"] for m in manifests} == {"heldout"}
    assert {s["controller"] for m in manifests for s in m["steps"]} == {"student"}
    assert all(m["takeover_step"] is None for m in manifests)


def test_seeds_outside_the_sealed_held_out_set_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed held-out seeds"):
        evaluation_report(
            ["ckpt-r0"],
            make_sim=RiggedSim,
            make_student=StubStudent,
            run_dir=tmp_path,
            seeds=[1, 1000],
            device="test-device",
        )


@pytest.mark.parametrize(("budget_ms", "all_over"), [(1.0, True), (1e9, False)])
def test_latency_comes_from_each_decision_and_is_counted_against_the_budget(
    tmp_path: Path, budget_ms: float, all_over: bool
) -> None:
    reports, _ = _run(tmp_path, TakesTwoMs, budget_ms=budget_ms)
    records = [json.loads(line) for line in (tmp_path / "evaluation.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    student_decisions = [e for e in events if e["type"] == "decision" and e["data"]["controller"] == "student"]
    assert sum(len(r["latencies_ms"]) for r in records) == len(student_decisions) > 0
    for report in reports:
        if report.decisions:
            assert report.over_budget == (report.decisions if all_over else 0)
            assert report.latency_median_ms is not None and report.latency_p95_ms is not None


def test_the_summary_states_the_sample_size_device_and_budget(tmp_path: Path) -> None:
    reports, _ = _run(tmp_path, CrashesOnSeed3)
    table = markdown(reports, 1000.0, "test-device", SEEDS)
    assert "held-out seeds 1, 2, 3, 4 (n=4 per task), on test-device; budget 1000 ms per decision" in table
    assert "| ckpt-r0 | training_distribution | 2/3 | 1 |" in table
    assert f"| ckpt-r0 | move_bin_after_lift | {NOT_RUN['move_bin_after_lift']} | - |" in table
    assert "held-out seeds 1, 3, 20 (n=3 per task)" in markdown(reports, 1000.0, "d", [1, 3, 20])
    saved = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert saved["budget_ms"] == 1000.0 and saved["device"] == "test-device" and saved["seeds"] == SEEDS
    assert len(saved["conditions"]) == 2 * len(CONDITIONS)


def test_a_recording_error_verdict_is_errored_not_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = evaluation.run_episode

    def refused_on_seed_2(sim: TruthSim, policy: Policy, **kwargs: Any) -> EpisodeManifest:
        manifest = real(sim, policy, **kwargs)
        if kwargs["seed"] == 2:
            manifest.validation = recording_error_verdict(RecordingError("2 executed actions but 1 timestamps"))
        return manifest

    monkeypatch.setattr(evaluation, "run_episode", refused_on_seed_2)
    reports = evaluation_report(
        ["ckpt-r0"],
        make_sim=RiggedSim,
        make_student=StubStudent,
        run_dir=tmp_path,
        seeds=SEEDS,
        device="test-device",
        conditions=CONDITIONS[:1],
    )
    assert (reports[0].n, reports[0].errored, reports[0].successes) == (4, 1, 1)  # seed 2 would have succeeded
    record = json.loads((tmp_path / "evaluation.jsonl").read_text().splitlines()[1])
    assert record["outcome"] == "errored" and record["reason"] == "recording error: 2 executed actions but 1 timestamps"


def test_p95_is_the_nearest_rank_not_the_maximum() -> None:
    record = EpisodeRecord(
        checkpoint="ckpt-r0",
        condition="training_distribution",
        task="across_room",
        seed=1,
        episode_id="ep",
        sim_version="v",
        outcome="rejected",
        reason="",
        latencies_ms=[float(ms) for ms in range(1, 21)],
        device="d",
    )
    report = summarise("ckpt-r0", "training_distribution", [record], 1000.0)
    assert report.latency_p95_ms == 19.0 and report.latency_median_ms == 10.5


def test_no_seeds_and_a_used_run_dir_are_refused(tmp_path: Path) -> None:
    common: dict[str, Any] = {"make_sim": RiggedSim, "make_student": StubStudent, "device": "d"}
    with pytest.raises(ValueError, match="no seeds"):
        evaluation_report(["ckpt-r0"], run_dir=tmp_path / "a", seeds=[], **common)
    evaluation_report(["ckpt-r0"], run_dir=tmp_path / "b", seeds=[1], conditions=CONDITIONS[:1], **common)
    with pytest.raises(FileExistsError, match="already holds an evaluation"):
        evaluation_report(["ckpt-r0"], run_dir=tmp_path / "b", seeds=[1], conditions=CONDITIONS[:1], **common)
