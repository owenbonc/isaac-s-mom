import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from isaacs_mom.contracts import Controller, Event, ExecResult, Observation, SkillCall
from isaacs_mom.orchestrator.dashboard.server import REPLAYED, create_app, tail_events
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.student import BASE_CHECKPOINT, StubStudent, stub_finetune
from isaacs_mom.teacher.argon import Argon, Part, Purpose, stub_generate

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


def test_frames_are_cached_for_good_because_a_recorded_frame_never_changes(tmp_path: Path) -> None:
    client = TestClient(create_app(make_run(tmp_path)))
    response = client.get("/frames/episodes/ep-1/frames/0000.jpg")
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert "cache-control" not in client.get("/frames/episodes/ep-1/frames/missing.jpg").headers


def test_frames_reject_traversal_and_non_frames(tmp_path: Path) -> None:
    client = TestClient(create_app(make_run(tmp_path)))
    for path in ["../../secret.jpg", "%2e%2e/%2e%2e/secret.jpg", "notes.txt", "episodes/ep-1/frames/missing.jpg"]:
        assert client.get(f"/frames/{path}").status_code == 404
    assert client.get("/frames//etc/passwd").status_code == 404


def test_frames_that_cannot_resolve_are_not_found(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "episodes" / "ep-1" / "frames" / "loop.jpg").symlink_to("loop.jpg")
    client = TestClient(create_app(run_dir), raise_server_exceptions=False)
    for path in ["episodes/ep-1/frames/loop.jpg", "episodes/ep-1/frames/0000%00.jpg", "%00"]:
        assert client.get(f"/frames/{path}").status_code == 404


def stream_of(run_dir: Path, appended: list[str] | None = None) -> list[str]:
    """Every SSE message, appending `appended` one line at a time once the existing lines have been read, then stop."""
    seen: list[str] = []
    pending = list(appended or [])

    async def stopped() -> bool:
        if pending:
            with (run_dir / "events.jsonl").open("a") as f:
                f.write(pending.pop(0))
            return False
        return True

    async def read() -> None:
        try:
            async for message in tail_events(run_dir, stopped, poll=0):
                seen.append(message)
        except Exception as error:  # a raise ends the response: the client sees the stream stop and reconnects
            seen.append(f"stream died: {error!r}")

    asyncio.run(read())
    assert all(message.endswith("\n\n") for message in seen), seen
    return seen


def collect(run_dir: Path, appended: list[str] | None = None) -> list[dict]:
    """The events in the stream, without its end-of-replay marker."""
    return [json.loads(m.removeprefix("data: ")) for m in stream_of(run_dir, appended) if m != REPLAYED]


def test_stream_replays_from_the_start_then_follows_appends(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    log = EventLog(run_dir, "r1")
    log.emit("episode_started", seed=1)
    log.emit("decision", controller="student")
    appended = [Event(t=3.0, run_id="r1", type="validation", data={}).model_dump_json() + "\n"]
    types = [e["type"] for e in collect(run_dir, appended)]
    assert types == ["episode_started", "decision", "validation"]


def test_stream_marks_where_its_replay_of_the_events_already_written_ends(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    log = EventLog(run_dir, "r1")
    log.emit("episode_started", seed=1)
    log.emit("decision", controller="student")
    appended = [Event(t=3.0, run_id="r1", type="validation", data={}).model_dump_json() + "\n"]
    messages = stream_of(run_dir, appended)
    assert messages.count(REPLAYED) == 1 and messages.index(REPLAYED) == 2  # after both written, before the new one
    assert stream_of(tmp_path / "not-started")[0] == REPLAYED  # nothing written yet: nothing to replay


def test_stream_waits_for_a_partial_line_to_complete(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    line = Event(t=1.0, run_id="r1", type="error", data={}).model_dump_json()
    (run_dir / "events.jsonl").write_text(line[:20])
    assert collect(run_dir, [line[20:], "\n"]) == [json.loads(line)]


def test_stream_skips_a_line_that_is_not_an_event_and_logs_its_line_number(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = make_run(tmp_path)
    log = EventLog(run_dir, "r1")
    log.emit("episode_started", seed=1)
    with (run_dir / "events.jsonl").open("a") as f:
        f.write("not json\n")
    log.emit("error", message="boom")
    with caplog.at_level(logging.WARNING, logger="isaacs_mom.orchestrator.dashboard.server"):
        assert [e["type"] for e in collect(run_dir)] == ["episode_started", "error"]
    assert [r.getMessage().split(":")[0] for r in caplog.records] == ["events.jsonl line 2 is not an event, skipped"]


def test_stream_goes_on_past_a_line_that_is_not_utf8(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    first, last = (Event(t=t, run_id="r1", type="error", data={}).model_dump_json() for t in (1.0, 2.0))
    torn = b'{"t": 1.5, "data": "\xff\xfe"}\n'  # a crash mid-write
    (run_dir / "events.jsonl").write_bytes(f"{first}\n".encode() + torn + f"{last}\n".encode())
    messages = stream_of(run_dir)
    assert [m if m == REPLAYED else json.loads(m.removeprefix("data: "))["t"] for m in messages] == [1.0, 2.0, REPLAYED]


def test_frame_paths_that_cannot_resolve_are_hidden_and_the_stream_goes_on(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "episodes" / "ep-1" / "frames" / "loop.jpg").symlink_to("loop.jpg")
    log = EventLog(run_dir, "r1")
    log.emit("action_executed", frame="episodes/ep-1/frames/loop.jpg")
    log.emit("action_executed", frame="episodes/ep-1/frames/0000\x00.jpg")
    log.emit("action_executed", frame="episodes/ep-1/frames/0000.jpg")
    assert [e["data"]["frame"] for e in collect(run_dir)] == [None, None, "/frames/episodes/ep-1/frames/0000.jpg"]


def test_stream_turns_frame_paths_into_urls_and_hides_outside_paths(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    log = EventLog(run_dir, "r1")
    log.emit("action_executed", frame="episodes/ep-1/frames/0000.jpg")
    log.emit("action_executed", frame="../../secret.jpg")
    log.emit("action_executed", frame=str(tmp_path / "secret.jpg"))
    log.emit("action_executed", frame=None)
    log.emit("perturbation", before_frame="episodes/ep-1/before_move_0000.jpg")
    log.emit("perturbation", before_frame="../../secret.jpg")
    frames = [e["data"].get("frame", e["data"].get("before_frame")) for e in collect(run_dir)]
    assert frames == [
        "/frames/episodes/ep-1/frames/0000.jpg",
        None,
        None,
        None,
        "/frames/episodes/ep-1/before_move_0000.jpg",
        None,
    ]


def test_frame_paths_find_the_same_file_whatever_directory_the_dashboard_starts_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "loop").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "loop")
    run_dir = Path("runs") / "r1"
    run_episode(
        FakeSim(),
        StubStudent(BASE_CHECKPOINT),
        run_dir=run_dir,
        events=EventLog(run_dir, "r1"),
        seed=1,
        split="dev",
        round_=0,
        instruction="push",
    )
    monkeypatch.chdir(tmp_path / "elsewhere")
    served = tmp_path / "loop" / "runs" / "r1"
    urls = [e["data"]["frame"] for e in collect(served) if e["type"] == "action_executed"]
    client = TestClient(create_app(served))
    assert urls and all(client.get(url).content == (served / url.removeprefix("/frames/")).read_bytes() for url in urls)


def test_episode_started_names_the_simulator_so_the_page_can_label_fakesim_placeholder(tmp_path: Path) -> None:
    sim = FakeSim()
    events = EventLog(tmp_path, "r1")
    run_episode(
        sim,
        StubStudent(BASE_CHECKPOINT),
        run_dir=tmp_path,
        events=events,
        seed=1,
        split="dev",
        round_=0,
        instruction="push",
    )
    first = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert first["type"] == "episode_started"
    assert (first["data"]["sim_version"], first["data"]["robot_config"]) == (sim.sim_version, sim.robot_config)


def test_each_teacher_demonstration_names_the_failed_episode_whose_diagnosis_it_acts_on(tmp_path: Path) -> None:
    argon = Argon(EventLog(tmp_path, "r1"), generate=stub_generate)  # no network in tests
    run_loop(
        FakeSim(),
        tmp_path,
        "r1",
        max_rounds=1,
        demos=3,
        dev_trials=2,
        argon=argon,
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    starts = {e["data"]["episode_id"]: e["data"] for e in events if e["type"] == "episode_started"}
    diagnosed = [e["data"]["episode_id"] for e in events if e["type"] == "diagnosis"]
    teacher = {
        e["data"]["episode_id"] for e in events if e["type"] == "decision" and e["data"]["controller"] == "teacher"
    }
    assert len(diagnosed) == 2 and len(teacher) == 3  # the stand-in student fails both trials
    assert [starts[t]["acting_on"] for t in sorted(teacher, key=list(starts).index)] == [*diagnosed, diagnosed[0]]
    assert all(start["acting_on"] is None for episode_id, start in starts.items() if episode_id not in teacher)
    assert not any(start["demo"] for start in starts.values())  # only `mom demo` runs the judged take


def test_page_and_its_scripts_are_served_without_caching(tmp_path: Path) -> None:
    client = TestClient(create_app(make_run(tmp_path)))
    page = client.get("/")
    assert "/static/app.js" in page.text and page.headers["cache-control"] == "no-cache"
    assert "new EventSource" in client.get("/static/app.js").text
    assert client.get("/static/style.css").headers["cache-control"] == "no-cache"


class SlowSim(FakeSim):
    """A sim whose every skill takes 0.2 s, as a long MuJoCo motion does."""

    def execute(self, call: SkillCall) -> ExecResult:
        time.sleep(0.2)
        return super().execute(call)


class SlowPolicy:
    """Decides once in 0.2 s, as a slow Argon call does, then stops; `fail` raises instead of deciding."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.step:
            return None
        time.sleep(0.2)
        if self.fail:
            raise RuntimeError("Argon unreachable")
        return SkillCall(skill="reobserve")


def logged(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_a_slow_decision_or_skill_sends_heartbeats_while_it_runs_and_none_after(tmp_path: Path) -> None:
    manifest = run_episode(
        SlowSim(),
        SlowPolicy(),
        run_dir=tmp_path,
        events=EventLog(tmp_path, "r1"),
        seed=1,
        split="dev",
        round_=0,
        instruction="hug",
        heartbeat_seconds=0.05,
    )
    time.sleep(0.15)
    types = [e["type"] for e in logged(tmp_path)]
    first, executed = types.index("decision"), types.index("action_executed")
    assert types[1:first].count("heartbeat") >= 2  # while deciding
    assert types[first:executed].count("heartbeat") >= 2  # while the skill executes
    assert "heartbeat" not in types[types.index("decision", executed) :]
    assert {e["data"]["episode_id"] for e in logged(tmp_path) if e["type"] == "heartbeat"} == {manifest.episode_id}


def test_heartbeats_stop_when_a_decision_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Argon unreachable"):
        run_episode(
            FakeSim(),
            SlowPolicy(fail=True),
            run_dir=tmp_path,
            events=EventLog(tmp_path, "r1"),
            seed=1,
            split="dev",
            round_=0,
            instruction="hug",
            heartbeat_seconds=0.05,
        )
    beats = sum(e["type"] == "heartbeat" for e in logged(tmp_path))
    time.sleep(0.15)
    assert beats >= 2 and sum(e["type"] == "heartbeat" for e in logged(tmp_path)) == beats


def test_two_event_logs_on_one_file_write_whole_lines_in_time_order_from_two_threads(tmp_path: Path) -> None:
    # The CLI's Argon logs through its own EventLog on the run's file, beside the loop's and its heartbeat thread.
    logs = [EventLog(tmp_path, "r1"), EventLog(tmp_path, "r1")]
    padding = "x" * 20_000  # longer than one buffered write, so unlocked writes would interleave

    def emit(log: EventLog) -> None:
        for _ in range(50):
            log.emit("heartbeat", padding=padding)

    threads = [threading.Thread(target=emit, args=(log,)) for log in logs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    times = [Event.model_validate_json(line).t for line in logs[0].path.read_text().splitlines()]
    assert len(times) == 100 and times == sorted(times)


def test_a_slow_diagnosis_sends_heartbeats_for_the_episode_it_diagnoses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isaacs_mom.orchestrator.loop.HEARTBEAT_SECONDS", 0.05)

    def slow(purpose: Purpose, parts: list[Part]) -> str:
        if purpose == "diagnose":
            time.sleep(0.2)
        return stub_generate(purpose, parts)

    argon = Argon(EventLog(tmp_path, "r1"), generate=slow)
    run_loop(
        FakeSim(),
        tmp_path,
        "r1",
        max_rounds=1,
        demos=1,
        dev_trials=1,
        argon=argon,
        make_student=StubStudent,
        train_student=stub_finetune,
    )
    events = logged(tmp_path)
    at = next(i for i, e in enumerate(events) if e["type"] == "diagnosis")
    beats = [e["data"]["episode_id"] for e in events[:at] if e["type"] == "heartbeat"]
    assert len(beats) >= 2 and set(beats) == {events[at]["data"]["episode_id"]}
