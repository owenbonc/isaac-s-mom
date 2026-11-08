"""Dashboard server (SPEC §3.5): one HTML page, `events.jsonl` as server-sent events, frame JPEGs from the run."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from isaacs_mom.contracts import Event

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
POLL_SECONDS = 0.25
FRAME_SUFFIXES = {".jpg", ".jpeg"}
# An executed step's frame, and the view right before a box or bin move (SPEC §3.5).
FRAME_KEYS = {"frame", "before_frame"}
# Sent once a connection has replayed every event already written, so the page shows only where the replay settles.
REPLAYED = "event: replayed\ndata: {}\n\n"


class FreshStatic(StaticFiles):
    """Static files the browser revalidates every load, so an edited page never serves stale."""

    def file_response(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201  # signature is Starlette's
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def frame_url(frame: object, run_dir: Path) -> str | None:
    """The `/frames/...` URL for an event's frame path, which is relative to the run directory; None outside it."""
    if not isinstance(frame, str):
        return None
    try:
        return "/frames/" + (run_dir / frame).resolve().relative_to(run_dir.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):  # outside the run, a NUL byte, or a symlink loop
        return None


def to_sse(line: str, number: int, run_dir: Path) -> str | None:
    """Line `number` of events.jsonl as an SSE message, its frame paths as URLs; None, logged, if malformed."""
    try:
        event = Event.model_validate_json(line)
    except ValidationError as error:
        log.warning("events.jsonl line %d is not an event, skipped: %s", number, error)
        return None
    for key in FRAME_KEYS & event.data.keys():
        event.data[key] = frame_url(event.data[key], run_dir)
    return f"data: {event.model_dump_json()}\n\n"


async def tail_events(
    run_dir: Path, stopped: Callable[[], Awaitable[bool]], poll: float = POLL_SECONDS
) -> AsyncIterator[str]:
    """Every complete event line from the start of events.jsonl, then `REPLAYED`, then new appends until `stopped()`."""
    path = run_dir / "events.jsonl"
    replayed = not path.exists()
    if replayed:
        yield REPLAYED  # nothing written yet, so nothing to replay
    while not path.exists():
        if await stopped():
            return
        await asyncio.sleep(poll)
    # A line torn by a crash mid-write decodes with replacement characters, fails validation and is skipped.
    with path.open(errors="replace") as f:
        number = 0
        while True:
            position = f.tell()
            line = f.readline()
            if line.endswith("\n"):
                number += 1
                if message := to_sse(line, number, run_dir):
                    yield message
                continue
            f.seek(position)  # a partial line is still being written
            if not replayed:
                replayed = True
                yield REPLAYED
            elif await stopped():
                return
            else:
                await asyncio.sleep(poll)


def create_app(run_dir: Path) -> FastAPI:
    app = FastAPI(title="Isaac's Mom dashboard")

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/events")
    def events(request: Request) -> StreamingResponse:
        return StreamingResponse(
            tail_events(run_dir, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/frames/{path:path}")
    def frames(path: str) -> FileResponse:
        try:
            target = (run_dir / path).resolve()
        except (ValueError, OSError, RuntimeError):  # a NUL byte or a symlink loop
            raise HTTPException(status_code=404) from None
        if not target.is_relative_to(run_dir.resolve()) or target.suffix not in FRAME_SUFFIXES or not target.is_file():
            raise HTTPException(status_code=404)
        # The recorder writes each frame once under a unique episode id, so a URL's bytes never change.
        return FileResponse(
            target, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"}
        )

    app.mount("/static", FreshStatic(directory=STATIC))
    return app


def serve(run_dir: Path, port: int) -> None:
    import uvicorn

    # An open event stream never ends on its own, so shutdown would wait for it forever.
    uvicorn.run(create_app(run_dir), host="127.0.0.1", port=port, timeout_graceful_shutdown=1)
