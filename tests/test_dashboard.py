import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from isaacs_mom.contracts import Controller, Event, ExecResult, Observation, SkillCall
from isaacs_mom.orchestrator.dashboard.server import create_app, tail_events
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.student import BASE_CHECKPOINT, StudentPolicy
from isaacs_mom.teacher.argon import Argon, stub_generate

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
    log.emit("action_executed", frame="episodes/ep-1/frames/0000.jpg")
    log.emit("action_executed", frame="../../secret.jpg")
    log.emit("action_executed", frame=str(tmp_path / "secret.jpg"))
    log.emit("action_executed", frame=None)
    frames = [e["data"]["frame"] for e in collect(run_dir)]
    assert frames == ["/frames/episodes/ep-1/frames/0000.jpg", None, None, None]


def test_frame_paths_find_the_same_file_whatever_directory_the_dashboard_starts_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "loop").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "loop")
    run_dir = Path("runs") / "r1"
    run_episode(
        FakeSim(),
        StudentPolicy(BASE_CHECKPOINT),
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
        StudentPolicy(BASE_CHECKPOINT),
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
    run_loop(FakeSim(), tmp_path, "r1", max_rounds=1, demos=3, dev_trials=2, argon=argon)
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


def test_two_threads_emitting_at_once_write_whole_lines_in_time_order(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "r1")
    padding = "x" * 20_000  # longer than one buffered write, so unlocked writes would interleave

    def emit() -> None:
        for _ in range(50):
            log.emit("heartbeat", padding=padding)

    threads = [threading.Thread(target=emit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    times = [Event.model_validate_json(line).t for line in log.path.read_text().splitlines()]
    assert len(times) == 100 and times == sorted(times)
