import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from isaacs_mom.contracts import Event
from isaacs_mom.orchestrator.dashboard.server import create_app, tail_events
from isaacs_mom.orchestrator.events import EventLog

JPEG = b"\xff\xd8\xff\xd9"


def make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "episodes" / "ep-1" / "frames").mkdir(parents=True)
    (run_dir / "episodes" / "ep-1" / "frames" / "0000.jpg").write_bytes(JPEG)
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    (run_dir / "notes.txt").write_text("not a frame")
    return run_dir


def test_frames_are_served_from_inside_the_run_directory(tmp_path: Path) -> None:
    client = TestClient(create_app(make_run(tmp_path)))
    response = client.get("/frames/episodes/ep-1/frames/0000.jpg")
    assert response.status_code == 200
    assert response.content == JPEG
    assert response.headers["content-type"] == "image/jpeg"


def test_frames_reject_traversal_and_non_frames(tmp_path: Path) -> None:
    client = TestClient(create_app(make_run(tmp_path)))
    for path in ["../../secret.jpg", "%2e%2e/%2e%2e/secret.jpg", "notes.txt", "episodes/ep-1/frames/missing.jpg"]:
        assert client.get(f"/frames/{path}").status_code == 404
    assert client.get("/frames//etc/passwd").status_code == 404


def collect(run_dir: Path, appended: list[str] | None = None) -> list[dict]:
    """Read the stream, appending `appended` after the existing lines have been read, then stop."""
    seen: list[dict] = []
    pending = list(appended or [])

    async def stopped() -> bool:
        if pending:
            with (run_dir / "events.jsonl").open("a") as f:
                f.write(pending.pop(0))
            return False
        return True

    async def read() -> None:
        async for message in tail_events(run_dir, stopped, poll=0):
            assert message.startswith("data: ") and message.endswith("\n\n")
            seen.append(json.loads(message.removeprefix("data: ")))

    asyncio.run(read())
    return seen


def test_stream_replays_from_the_start_then_follows_appends(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    log = EventLog(run_dir, "r1")
    log.emit("episode_started", seed=1)
    log.emit("decision", controller="student")
    appended = [Event(t=3.0, run_id="r1", type="validation", data={}).model_dump_json() + "\n"]
    types = [e["type"] for e in collect(run_dir, appended)]
    assert types == ["episode_started", "decision", "validation"]


def test_stream_waits_for_a_partial_line_to_complete(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    line = Event(t=1.0, run_id="r1", type="error", data={}).model_dump_json()
    (run_dir / "events.jsonl").write_text(line[:20])
    assert collect(run_dir, [line[20:], "\n"]) == [json.loads(line)]


def test_stream_skips_malformed_lines(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "events.jsonl").write_text("not json\n")
    EventLog(run_dir, "r1").emit("error", message="boom")
    assert [e["type"] for e in collect(run_dir)] == ["error"]


def test_stream_turns_frame_paths_into_urls_and_hides_outside_paths(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    log = EventLog(run_dir, "r1")
    log.emit("action_executed", frame=str(run_dir / "episodes" / "ep-1" / "frames" / "0000.jpg"))
    log.emit("action_executed", frame=str(tmp_path / "secret.jpg"))
    frames = [e["data"]["frame"] for e in collect(run_dir)]
    assert frames == ["/frames/episodes/ep-1/frames/0000.jpg", None]


def test_page_is_served(tmp_path: Path) -> None:
    client = TestClient(create_app(make_run(tmp_path)))
    assert "EventSource" in client.get("/").text
