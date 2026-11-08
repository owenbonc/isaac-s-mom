"""The student evaluation report (SPEC §6 criterion 10): checkpoints, student only, on the sealed held-out seeds, in
three conditions. Reported only: the loop's stop rule (§3.6) stays on `across_room`.

Each episode is recorded as usual (frames, skill calls, perturbations and verdict in its manifest). This adds one line
per episode to `evaluation.jsonl`, with the condition, the decision latencies and the device, and a summary per
checkpoint and condition in `evaluation_report.json`.
"""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.policy import Policy
from isaacs_mom.seeds import heldout_seeds
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.student import BASE_CHECKPOINT, StubStudent, StudentPolicy
from isaacs_mom.validator.recorder import write_atomically
from isaacs_mom.validator.rules import RECORDING_ERROR

# SPEC §6.10's edge-device budget (Owen's pick, #22): 2 s per decision and 3 GB of memory on a laptop.
DEFAULT_BUDGET_MS = 2000.0
MEMORY_BUDGET_GB = 3.0

ConditionName = Literal["training_distribution", "new_layout", "move_bin_after_lift"]
Outcome = Literal["accepted", "rejected", "errored"]


@dataclass(frozen=True)
class Condition:
    name: ConditionName
    tasks: tuple[str, ...]


CONDITIONS = (
    Condition("training_distribution", ("across_room",)),
    Condition("new_layout", ("near_bin", "far_bin")),  # bin distances the training task never uses
    Condition("move_bin_after_lift", ("across_room",)),
)
# The lift trigger belongs in `run_episode`, which lands after WS5's dashboard PR; until then the condition isn't run.
NOT_RUN = {"move_bin_after_lift": "not run: needs the lift-triggered move_bin in run_episode"}


class EpisodeRecord(BaseModel):
    checkpoint: str
    condition: ConditionName
    task: str
    seed: int
    episode_id: str | None
    sim_version: str
    outcome: Outcome
    reason: str
    latencies_ms: list[float]
    device: str


class ConditionReport(BaseModel):
    checkpoint: str
    condition: ConditionName
    n: int
    errored: int
    successes: int
    decisions: int
    latency_median_ms: float | None
    latency_p95_ms: float | None
    over_budget: int  # decisions slower than the budget
    note: str | None = None

    @property
    def scored(self) -> int:
        return self.n - self.errored

    def score(self) -> str:
        return f"{self.successes}/{self.scored}" if self.note is None else self.note


def evaluation_report(
    checkpoints: Sequence[str],
    *,
    make_sim: Callable[[str], TruthSim],
    make_student: Callable[[str], Policy],
    run_dir: Path,
    seeds: Sequence[int],
    device: str,
    budget_ms: float = DEFAULT_BUDGET_MS,
    conditions: Sequence[Condition] = CONDITIONS,
    peak_memory_gb: Callable[[], float | None] = lambda: None,
) -> list[ConditionReport]:
    """Run every checkpoint on every condition's tasks and seeds; write the records and the report."""
    if not seeds:
        raise ValueError("no seeds to evaluate on")
    if outside := sorted(set(seeds) - heldout_seeds()):
        raise ValueError(f"the report runs only on the sealed held-out seeds, not {outside}")
    records_path, report_path = run_dir / "evaluation.jsonl", run_dir / "evaluation_report.json"
    if records_path.exists() or report_path.exists():
        raise FileExistsError(f"{run_dir} already holds an evaluation; each report gets its own run dir")
    events = EventLog(run_dir, run_dir.name)
    sims: dict[str, TruthSim] = {}
    reports: list[ConditionReport] = []
    every_record: list[EpisodeRecord] = []
    for checkpoint in checkpoints:
        student = make_student(checkpoint)
        for condition in conditions:
            if condition.name in NOT_RUN:
                reports.append(_not_run(checkpoint, condition))
                continue
            records = []
            for task in condition.tasks:
                if task not in sims:
                    sims[task] = make_sim(task)
                sim = sims[task]
                for seed in seeds:
                    record = _run_one(sim, student, checkpoint, condition, task, seed, run_dir, events, device)
                    records.append(record)
                    every_record.append(record)
                    write_atomically(records_path, "".join(r.model_dump_json() + "\n" for r in every_record))
            reports.append(summarise(checkpoint, condition.name, records, budget_ms))
    report = {
        "budget_ms": budget_ms,
        "memory_budget_gb": MEMORY_BUDGET_GB,
        "peak_memory_gb": peak_memory_gb(),
        "device": device,
        "seeds": list(seeds),
        "conditions": [r.model_dump() for r in reports],
    }
    write_atomically(report_path, json.dumps(report, indent=2))
    return reports


def _run_one(
    sim: TruthSim,
    student: Policy,
    checkpoint: str,
    condition: Condition,
    task: str,
    seed: int,
    run_dir: Path,
    events: EventLog,
    device: str,
) -> EpisodeRecord:
    start = events.path.stat().st_size if events.path.exists() else 0
    try:
        manifest = run_episode(
            sim, student, run_dir=run_dir, events=events, seed=seed, split="heldout", round_=0, instruction=INSTRUCTION
        )
    except Exception as error:  # an errored episode is reported apart, not as the student's failure (§3.5)
        events.emit("error", seed=seed, split="heldout", round=0, error=f"{type(error).__name__}: {error}")
        episode_id, outcome, reason = (
            _started_episode(events.path, start),
            "errored",
            f"{type(error).__name__}: {error}",
        )
    else:
        verdict = manifest.validation
        episode_id = manifest.episode_id
        if verdict is None or verdict.reason.startswith(RECORDING_ERROR):
            outcome, reason = "errored", verdict.reason if verdict else "no verdict"
        else:
            outcome, reason = "accepted" if verdict.accepted else "rejected", verdict.reason
    return EpisodeRecord(
        checkpoint=checkpoint,
        condition=condition.name,
        task=task,
        seed=seed,
        episode_id=episode_id,
        sim_version=sim.sim_version,
        outcome=outcome,
        reason=reason,
        latencies_ms=_latencies(events.path, start),
        device=device,
    )


def _events_since(events_path: Path, start: int) -> list[dict[str, Any]]:
    if not events_path.exists():  # the episode errored before anything was logged
        return []
    with events_path.open() as f:
        f.seek(start)
        return [json.loads(line) for line in f if line.strip()]


def _latencies(events_path: Path, start: int) -> list[float]:
    """The latency of every decision written since `start`: this episode's, all the student's (no takeover)."""
    return [float(e["data"]["latency_ms"]) for e in _events_since(events_path, start) if e["type"] == "decision"]


def _started_episode(events_path: Path, start: int) -> str | None:
    """A crashed episode's id, so its partial manifest can be found; None if it crashed before it started."""
    started = [e["data"]["episode_id"] for e in _events_since(events_path, start) if e["type"] == "episode_started"]
    return started[0] if started else None


def summarise(
    checkpoint: str, condition: ConditionName, records: Sequence[EpisodeRecord], budget_ms: float
) -> ConditionReport:
    latencies = sorted(ms for r in records for ms in r.latencies_ms)
    return ConditionReport(
        checkpoint=checkpoint,
        condition=condition,
        n=len(records),
        errored=sum(r.outcome == "errored" for r in records),
        successes=sum(r.outcome == "accepted" for r in records),
        decisions=len(latencies),
        latency_median_ms=statistics.median(latencies) if latencies else None,
        latency_p95_ms=latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,  # nearest rank
        over_budget=sum(ms > budget_ms for ms in latencies),
    )


def _not_run(checkpoint: str, condition: Condition) -> ConditionReport:
    return ConditionReport(
        checkpoint=checkpoint,
        condition=condition.name,
        n=0,
        errored=0,
        successes=0,
        decisions=0,
        latency_median_ms=None,
        latency_p95_ms=None,
        over_budget=0,
        note=NOT_RUN[condition.name],
    )


def markdown(
    reports: Sequence[ConditionReport],
    budget_ms: float,
    device: str,
    seeds: Sequence[int],
    peak_memory_gb: float | None = None,
) -> str:
    """The summary table: successes out of (n - errored), with the errored count and the latency against budget."""
    memory = f"peak GPU memory {peak_memory_gb:.2f} GB" if peak_memory_gb is not None else "GPU memory not measured"
    lines = [
        f"Student only, held-out seeds {', '.join(map(str, seeds))} (n={len(seeds)} per task), on {device}; "
        f"budget {budget_ms:.0f} ms per decision and {MEMORY_BUDGET_GB:.0f} GB, {memory}.",
        "",
        "| Checkpoint | Condition | Successes / (n - errored) | Errored | Median ms | p95 ms | Over budget |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        median = f"{r.latency_median_ms:.0f}" if r.latency_median_ms is not None else "-"
        p95 = f"{r.latency_p95_ms:.0f}" if r.latency_p95_ms is not None else "-"
        over = f"{r.over_budget}/{r.decisions}" if r.decisions else "-"
        errored = "-" if r.note else str(r.errored)
        lines.append(f"| {r.checkpoint} | {r.condition} | {r.score()} | {errored} | {median} | {p95} | {over} |")
    return "\n".join(lines)


def _cuda_peak_memory_gb() -> float:
    import torch

    return torch.cuda.max_memory_allocated() / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description="SPEC §6 criterion 10: the student evaluation report")
    parser.add_argument("--ckpt", action="append", default=[], help="a fine-tuned checkpoint id (ckpt-r0 always runs)")
    parser.add_argument("--student", choices=["smolvlm", "stub"], default="smolvlm")
    parser.add_argument("--seeds", type=int, default=None, help="only the first N held-out seeds")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    args = parser.parse_args()
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be at least 1")
    from isaacs_mom.sim.mujoco import MujocoSim
    from isaacs_mom.sim.mujoco.task import TASKS

    if args.student == "stub":
        make_student, device = StubStudent, "stub, no model"
    else:
        from isaacs_mom.student import vlm  # loads torch, only for the real model

        make_student, device = StudentPolicy, vlm.pick_device()
    peak_memory = _cuda_peak_memory_gb if device == "cuda" else lambda: None
    seeds = sorted(heldout_seeds())[: args.seeds]
    run_dir = Path("runs") / time.strftime("eval-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)  # refuses a second run started in the same second
    reports = evaluation_report(
        [BASE_CHECKPOINT, *args.ckpt],
        make_sim=lambda task: MujocoSim(TASKS[task]),
        make_student=make_student,
        run_dir=run_dir,
        seeds=seeds,
        device=device,
        budget_ms=args.budget_ms,
        peak_memory_gb=peak_memory,
    )
    print(markdown(reports, args.budget_ms, device, seeds, peak_memory()))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
