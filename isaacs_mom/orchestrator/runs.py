"""The runs directory the dashboard serves (SPEC §3.5 "Teach and runs"): which runs it holds and what each is doing."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from isaacs_mom.contracts import Event
from isaacs_mom.orchestrator.events import read_events
from isaacs_mom.student import valid_run_id

FRESH_SECONDS = 20.0
FINETUNE_SECONDS = 30 * 60.0
NOT_STARTED = "not started: "  # the error a Teach the task generator refused leaves as its run's last word


class RunSummary(BaseModel):
    run_id: str
    instruction: str | None
    status: Literal["live", "recorded", "not_started"]
    reason: str | None
    started_t: float


def find_run(runs_dir: Path, run_id: str) -> Path | None:
    """The run an id from a URL names: a direct child of the runs directory with an events file; None otherwise."""
    if not valid_run_id(run_id):  # no separator and no leading dot, so it can only name a child
        return None
    run_dir = runs_dir / run_id
    return run_dir if (run_dir / "events.jsonl").is_file() else None


def summarize(run_id: str, events: list[Event], now: float) -> RunSummary:
    """Live while the last event is under 20 s old, or a fine-tune started under 30 minutes ago hasn't completed."""
    instruction = reason = None
    finetune_t: float | None = None
    for event in events:
        match event.type, event.data:
            case "instruction", {"text": str(text)} if instruction is None:
                instruction = text
            case "error", {"error": str(error)} if error.startswith(NOT_STARTED):
                reason = error.removeprefix(NOT_STARTED)
            case "finetune_started", _:
                finetune_t = event.t
            case "finetune_completed", _:
                finetune_t = None
    fresh = now - events[-1].t < FRESH_SECONDS
    tuning = finetune_t is not None and now - finetune_t < FINETUNE_SECONDS
    status = "not_started" if reason is not None else "live" if fresh or tuning else "recorded"
    return RunSummary(run_id=run_id, instruction=instruction, status=status, reason=reason, started_t=events[0].t)


def list_runs(runs_dir: Path, now: float) -> list[RunSummary]:
    """Live runs first, then the newest. A run with no event yet is left out: it has nothing to show."""
    if not runs_dir.is_dir():
        return []
    # shortcut: every call re-reads each run's events; cache by file size and mtime once a runs dir holds many runs.
    found = [(child.name, read_events(child)) for child in runs_dir.iterdir() if find_run(runs_dir, child.name)]
    runs = [summarize(run_id, events, now) for run_id, events in found if events]
    return sorted(runs, key=lambda run: (run.status != "live", -run.started_t))
