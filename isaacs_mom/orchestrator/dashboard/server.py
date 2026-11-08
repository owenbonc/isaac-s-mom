"""Dashboard server (SPEC §3.5) over a runs directory: one HTML page, and for each run its `events.jsonl` as
server-sent events, its frame JPEGs and its training and replay summaries; the Teach box starts a new run."""

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from isaacs_mom.contracts import Event, ReplayReport, TrainingSetSummary
from isaacs_mom.orchestrator.loop import instruction_text
from isaacs_mom.orchestrator.runs import Launch, RunSummary, find_run, launch_run, list_runs, teach
from isaacs_mom.sim.mujoco.task import DEFAULT_TASK
from isaacs_mom.teacher.argon import Generate

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
POLL_SECONDS = 0.25
FRAME_SUFFIXES = {".jpg", ".jpeg"}
# An executed step's frame, and the view right before a box or bin move (SPEC §3.5).
FRAME_KEYS = {"frame", "before_frame"}
# Sent once a connection has replayed every event already written, so the page shows only where the replay settles.
REPLAYED = "event: replayed\ndata: {}\n\n"
TRAINING_SUMMARY = re.compile(r"training_r[0-9]+\.summary\.json")
LOCAL_CLIENTS = {"127.0.0.1", "::1"}
# The page sends at most 200 characters; a body far past that is refused before it is read.
TEACH_BODY_BYTES = 4096


class TeachRequest(BaseModel):
    instruction: str


class FreshStatic(StaticFiles):
    """Static files the browser revalidates every load, so an edited page never serves stale."""

    def file_response(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201  # signature is Starlette's
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def frame_url(frame: object, run_dir: Path) -> str | None:
    """The `/frames/<run_id>/...` URL for an event's frame path, which is relative to the run directory; None outside
    it."""
    if not isinstance(frame, str):
        return None
    try:
        return f"/frames/{run_dir.name}/" + (run_dir / frame).resolve().relative_to(run_dir.resolve()).as_posix()
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
    replayed = False
    # One byte that isn't UTF-8 would otherwise end the stream; its line fails validation and is skipped instead.
    with (run_dir / "events.jsonl").open(errors="replace") as f:
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


def create_app(
    runs_dir: Path,
    *,
    port: int = 8200,
    task: str = DEFAULT_TASK.name,
    generate: Generate | None = None,
    launch: Launch = launch_run,
) -> FastAPI:
    """`task` is what a Teach-started run trains on; `generate` answers for Argon (None: Gemini) and `launch` starts
    `mom run`."""
    app = FastAPI(title="Isaac's Mom dashboard")
    # A DNS-rebinding page reaches 127.0.0.1 under its own host name, so only a local name is served.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    own_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def served(run_id: str) -> Path:
        """The run directory an id from a URL names, or 404."""
        if (run_dir := find_run(runs_dir, run_id)) is None:
            raise HTTPException(status_code=404)
        return run_dir

    @app.get("/")
    def page() -> FileResponse:
        # Never framed, so no other page can lead a click into the Teach box under its own decoy.
        headers = {"Cache-Control": "no-cache", "Content-Security-Policy": "frame-ancestors 'none'"}
        return FileResponse(STATIC / "index.html", headers=headers)

    @app.get("/runs")
    def runs() -> list[RunSummary]:
        return list_runs(runs_dir, time.time())

    @app.get("/events")
    def events(run: str, request: Request) -> StreamingResponse:
        return StreamingResponse(
            tail_events(served(run), request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/frames/{run_id}/{path:path}")
    def frames(run_id: str, path: str) -> FileResponse:
        run_dir = served(run_id).resolve()
        try:
            target = (run_dir / path).resolve()
        except (ValueError, OSError, RuntimeError):  # a NUL byte or a symlink loop
            raise HTTPException(status_code=404) from None
        # os.path.isfile is False on any OSError, where Path.is_file re-raises a name too long.
        if not target.is_relative_to(run_dir) or target.suffix not in FRAME_SUFFIXES or not os.path.isfile(target):
            raise HTTPException(status_code=404)
        # The recorder writes each frame once under a unique episode id, so a URL's bytes never change.
        return FileResponse(
            target, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"}
        )

    @app.get("/run-files/{run_id}/{name}", response_model=None)
    def run_files(run_id: str, name: str) -> TrainingSetSummary | ReplayReport | Response:
        if name == "replay_report.json":
            model = ReplayReport
        elif TRAINING_SUMMARY.fullmatch(name):
            model = TrainingSetSummary
        else:
            raise HTTPException(status_code=404)
        try:
            return model.model_validate_json((served(run_id) / name).read_bytes())
        except FileNotFoundError:  # not written yet, which is no error: the page says so in words
            return Response(status_code=204)
        except (OSError, ValidationError):
            raise HTTPException(status_code=404) from None

    @app.post("/teach")
    async def teach_box(request: Request) -> JSONResponse:
        # Checked before the body is read: only this dashboard's own page, on this machine, starts a run (SPEC §3.5).
        client = request.client
        if client is None or client.host not in LOCAL_CLIENTS or request.headers.get("origin") not in own_origins:
            raise HTTPException(status_code=403, detail="Teach takes an instruction only from this dashboard's page")
        # shortcut: a chunked body has no length to check and is read whole; only a local non-browser client sends one.
        if int(request.headers.get("content-length") or 0) > TEACH_BODY_BYTES:
            raise HTTPException(status_code=413, detail=f"send at most {TEACH_BODY_BYTES} bytes")
        try:
            instruction = instruction_text(TeachRequest.model_validate_json(await request.body()).instruction)
        except ValidationError:
            raise HTTPException(status_code=400, detail='send JSON: {"instruction": "<text>"}') from None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        run_id, reasons = await run_in_threadpool(
            teach, runs_dir, instruction, task=task, generate=generate, launch=launch
        )
        if reasons:
            return JSONResponse({"run_id": run_id, "started": False, "reasons": reasons})
        return JSONResponse({"run_id": run_id, "started": True}, status_code=201)

    app.mount("/static", FreshStatic(directory=STATIC))
    return app


def serve(runs_dir: Path, port: int, task: str) -> None:
    import uvicorn

    # An open event stream never ends on its own, so shutdown would wait for it forever.
    uvicorn.run(create_app(runs_dir, port=port, task=task), host="127.0.0.1", port=port, timeout_graceful_shutdown=1)
