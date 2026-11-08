"""The runs directory the dashboard serves (SPEC §3.5 "Teach and runs"): which runs it holds, what each is doing, and
a new run for each instruction typed into the Teach box."""

import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from isaacs_mom.contracts import Event
from isaacs_mom.orchestrator.events import EventLog, read_events
from isaacs_mom.student import valid_run_id
from isaacs_mom.tasks import Refusal, TaskSpec, argon_proposer, generate_task
from isaacs_mom.tasks.spec import Bin, Inside
from isaacs_mom.teacher.argon import Argon, Generate, MissingKey, hide_key

FRESH_SECONDS = 20.0
FINETUNE_SECONDS = 30 * 60.0
NOT_STARTED = "not started: "  # the error a Teach that started nothing leaves as its run's last word
STOPPED = "mom run stopped"  # the error a Teach-started run that exited non-zero leaves as its last word
NOT_YET = "Not yet: the loop teaches one task"
NO_KEY = "Argon isn't reachable: GEMINI_API_KEY is not set on the machine serving the dashboard"
REASON_CHARS = 300
LOG_TAIL_BYTES = 64 * 1024
Launch = Callable[[list[str], Path], object]  # a command, and the log in its run's directory its output goes to


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
    """Live while the last event is under 20 s old, or a fine-tune started under 30 minutes ago hasn't completed, and
    never once its `mom run` has stopped."""
    instruction = reason = None
    finetune_t: float | None = None
    stopped = False
    for event in events:
        match event.type, event.data:
            case "instruction", {"text": str(text)} if instruction is None:
                instruction = text
            case "error", {"error": str(error)} if error.startswith(NOT_STARTED):
                reason = error.removeprefix(NOT_STARTED)
            case "error", {"error": str(error)} if error.startswith(STOPPED):
                stopped = True
            case "finetune_started", _:
                finetune_t = event.t
            case "finetune_completed", _:
                finetune_t = None
    fresh = now - events[-1].t < FRESH_SECONDS
    tuning = finetune_t is not None and now - finetune_t < FINETUNE_SECONDS
    live = (fresh or tuning) and not stopped
    status = "not_started" if reason is not None else "live" if live else "recorded"
    return RunSummary(run_id=run_id, instruction=instruction, status=status, reason=reason, started_t=events[0].t)


def list_runs(runs_dir: Path, now: float) -> list[RunSummary]:
    """Live runs first, then the newest. A run with no event yet is left out: it has nothing to show."""
    if not runs_dir.is_dir():
        return []
    # shortcut: every call re-reads each run's events; cache by file size and mtime once a runs dir holds many runs.
    found = [(child.name, read_events(child)) for child in runs_dir.iterdir() if find_run(runs_dir, child.name)]
    runs = [summarize(run_id, events, now) for run_id, events in found if events]
    return sorted(runs, key=lambda run: (run.status != "live", -run.started_t))


def launch_run(command: list[str], log: Path) -> tuple[subprocess.Popen[bytes], threading.Thread]:
    """Starts the command in its own session, so a run outlives the dashboard and a Ctrl-C there never reaches it. A
    daemon thread waits for it, so it's never left a zombie, and records why in its run if it fails. Returns the child
    and that thread."""
    with log.open("ab") as out:
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    watcher = threading.Thread(target=_record_exit, args=(child, log), daemon=True)
    watcher.start()
    return child, watcher


def _record_exit(child: subprocess.Popen[bytes], log: Path) -> None:
    """A non-zero exit becomes one `error` event in the run, its log's last line as why. The child has exited, so
    nothing else is writing the run's events."""
    code = child.wait()
    if code == 0:
        return
    run_dir = log.parent
    why = _readable(_last_line(log) or "run.log is empty")
    if any(event.type == "episode_started" for event in read_events(run_dir)):
        error = f"{STOPPED} (exit {code}): {why}"
    else:
        error = f"{NOT_STARTED}{STOPPED}: {why}"
    _end_line(run_dir / "events.jsonl")
    EventLog(run_dir, run_dir.name).emit("error", error=error)


def _last_line(log: Path) -> str:
    """The log's last non-empty line, read from its tail only, since a long run's log is large; empty when it can't be
    read, so the error is still written."""
    try:
        with log.open("rb") as f:
            start = max(0, f.seek(0, os.SEEK_END) - LOG_TAIL_BYTES)
            f.seek(start)
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    if start > 0:
        lines = lines[1:]  # the tail can start mid-line, even mid-key, so its first line may be a fragment
    return next((line.strip() for line in reversed(lines) if line.strip()), "")


def _end_line(events_path: Path) -> None:
    """A child killed mid-write leaves half a line, so the error starts a line of its own and still parses."""
    if not events_path.is_file() or events_path.stat().st_size == 0:
        return
    with events_path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")


def _readable(text: str) -> str:
    """At most 300 characters, with the Argon key never shown (rule 5). It's replaced before the 300 cut, and the log's
    tail drops its first, partial line, so neither cut leaves part of it."""
    return hide_key(text)[:REASON_CHARS]


def is_the_loops_task(spec: TaskSpec) -> bool:
    """The loop teaches one task: the box inside the bin (SPEC §3.5)."""
    return isinstance(spec.scene.destination, Bin) and any(isinstance(fact, Inside) for fact in spec.success.all_of)


def teach(
    runs_dir: Path, instruction: str, *, task: str, generate: Generate | None, launch: Launch
) -> tuple[str, list[str]]:
    """A new run whose first event is the instruction. The task generator judges it, its Argon call counted in the
    run's own budget (rule 5), and `mom run` starts there when it asks for the loop's task. Returns the run id and why
    nothing started, empty when the run did."""
    run_id, run_dir = _new_run(runs_dir)
    events = EventLog(run_dir, run_id)
    events.emit("instruction", text=instruction, source="teach")
    reasons = _judge(instruction, events, generate)
    if not reasons:
        # One argument whatever the text holds, even a leading "-" that would read as an option.
        cli = [sys.executable, "-m", "isaacs_mom.orchestrator.cli", "run", "--runs", str(runs_dir.resolve())]
        try:
            launch([*cli, "--run-id", run_id, f"--instruction={instruction}", "--task", task], run_dir / "run.log")
        except OSError as error:
            reasons = [f"mom run could not start: {error}"]
    if reasons:
        events.emit("error", error=NOT_STARTED + "; ".join(reasons))
    return run_id, reasons


def _new_run(runs_dir: Path) -> tuple[str, Path]:
    """A fresh run directory, made exclusively, so two Teaches in the same second never share one."""
    while True:
        run_id = f"{time.strftime('teach-%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        try:
            (runs_dir / run_id).mkdir(parents=True)
        except FileExistsError:
            continue
        return run_id, runs_dir / run_id


def _judge(instruction: str, events: EventLog, generate: Generate | None) -> list[str]:
    """Why the instruction can't start the loop; empty when it asks for the loop's task."""
    try:
        task = generate_task(instruction, argon_proposer(Argon(events, generate=generate)))
    except MissingKey:  # the CLI's hint about `--teacher stub` means nothing to someone at the dashboard
        return [NO_KEY]
    except Exception as error:  # a fatal Argon error refuses this Teach, and its run says why
        return [_readable(f"task generator failed: {type(error).__name__}: {error}")]
    if isinstance(task, Refusal):
        return task.reasons
    return [] if is_the_loops_task(task) else [NOT_YET]
