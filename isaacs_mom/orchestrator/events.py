import threading
import time
from pathlib import Path

from isaacs_mom.contracts import Event, EventType

# One lock per events file, shared by every EventLog on it: the loop's, its heartbeat thread's and the teacher's.
_LOCKS: dict[str, threading.Lock] = {}


class EventLog:
    """Appends SPEC §4.4 events to runs/<run_id>/events.jsonl; the dashboard tails this file."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "events.jsonl"
        self.run_id = run_id
        self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.Lock())

    def emit(self, type_: EventType, **data: object) -> None:
        with self._lock:
            event = Event(t=time.time(), run_id=self.run_id, type=type_, data=data)
            with self.path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
