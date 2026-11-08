import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from isaacs_mom.contracts import (
    Controller,
    Event,
    EventType,
    ExecResult,
    Observation,
    ReplayFailure,
    ReplayReport,
    SkillCall,
    TrainingSetSummary,
)
from isaacs_mom.orchestrator.dashboard.server import REPLAYED, create_app, tail_events
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.fake import FakeSim
from isaacs_mom.student import BASE_CHECKPOINT, StubStudent, stub_finetune
from isaacs_mom.teacher.argon import Argon, Part, Purpose, stub_generate

JPEG = b"\xff\xd8\xff\xd9"


def make_run(tmp_path: Path, run_id: str = "r1") -> Path:
    """runs/<run_id>/ with an (empty) events file, one frame and a file that isn't one; a secret JPEG beside runs/."""
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "episodes" / "ep-1" / "frames").mkdir(parents=True)
    (run_dir / "episodes" / "ep-1" / "frames" / "0000.jpg").write_bytes(JPEG)
    (run_dir / "events.jsonl").touch()
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    (run_dir / "notes.txt").write_text("not a frame")
    return run_dir


def runs_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "runs"))


def test_frames_are_served_from_inside_their_run_directory(tmp_path: Path) -> None:
    make_run(tmp_path)
    response = runs_client(tmp_path).get("/frames/r1/episodes/ep-1/frames/0000.jpg")
    assert response.status_code == 200
    assert response.content == JPEG
    assert response.headers["content-type"] == "image/jpeg"


def test_frames_are_cached_for_good_because_a_recorded_frame_never_changes(tmp_path: Path) -> None:
    make_run(tmp_path)
    client = runs_client(tmp_path)
    response = client.get("/frames/r1/episodes/ep-1/frames/0000.jpg")
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert "cache-control" not in client.get("/frames/r1/episodes/ep-1/frames/missing.jpg").headers


def test_frames_reject_traversal_non_frames_and_anything_but_a_run(tmp_path: Path) -> None:
    make_run(tmp_path)
    make_run(tmp_path, "r2")
    (tmp_path / "runs" / ".hidden" / "episodes").mkdir(parents=True)  # a run's shape, but not a valid run id
    (tmp_path / "runs" / ".hidden" / "events.jsonl").touch()
    (tmp_path / "runs" / ".hidden" / "x.jpg").write_bytes(JPEG)
    (tmp_path / "runs" / "no-events").mkdir()  # not a run: no events file
    (tmp_path / "runs" / "no-events" / "x.jpg").write_bytes(JPEG)
    client = runs_client(tmp_path)
    for path in [
        "r1/../../secret.jpg",
        "r1/%2e%2e/%2e%2e/secret.jpg",
        "r1/%2e%2e/r2/episodes/ep-1/frames/0000.jpg",  # another run's frame, only through its own run id
        "r1/notes.txt",
        "r1/episodes/ep-1/frames/missing.jpg",
        "r1//etc/passwd",
        "%2e%2e/secret.jpg",
        "%2e%2e%2fsecret.jpg",
        ".hidden/x.jpg",
        "no-events/x.jpg",
        "r3/episodes/ep-1/frames/0000.jpg",
    ]:
        assert client.get(f"/frames/{path}").status_code == 404, path
    assert client.get("/frames/r2/episodes/ep-1/frames/0000.jpg").content == JPEG


@pytest.mark.parametrize("run", ["", "r3", ".hidden", "no-events", "..", "r1/..", "%2e%2e", "r 1"])
def test_the_event_stream_is_only_for_a_run_of_the_runs_directory(tmp_path: Path, run: str) -> None:
    make_run(tmp_path)
    (tmp_path / "runs" / ".hidden").mkdir()
    (tmp_path / "runs" / ".hidden" / "events.jsonl").touch()
    (tmp_path / "runs" / "no-events").mkdir()
    (tmp_path / "events.jsonl").touch()  # the runs dir's parent is never a run
    assert runs_client(tmp_path).get("/events", params={"run": run}).status_code == 404


def write_events(run_dir: Path, *events: tuple[float, EventType, dict]) -> None:
    """Events at the given times, as `mom run` and the Teach box write them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [Event(t=t, run_id=run_dir.name, type=kind, data=data).model_dump_json() for t, kind, data in events]
    (run_dir / "events.jsonl").write_text("".join(line + "\n" for line in lines))


def test_the_runs_list_puts_live_runs_first_then_the_newest_and_says_why_one_never_started(tmp_path: Path) -> None:
    runs, now = tmp_path / "runs", time.time()
    write_events(
        runs / "fresh", (now - 100, "instruction", {"text": "put it in", "source": "cli"}), (now - 5, "heartbeat", {})
    )
    write_events(
        runs / "fine-tuning", (now - 7200, "episode_started", {}), (now - 600, "finetune_started", {"round": 1})
    )
    write_events(
        runs / "refused",
        (now - 3, "instruction", {"text": "<b>juggle</b>", "source": "teach"}),
        (now - 2, "error", {"error": "not started: Argon declined: no juggling; Not yet"}),
    )
    write_events(runs / "old", (now - 3600, "instruction", {"text": "box in bin", "source": "teach"}))
    write_events(
        runs / "tuned",
        (now - 5000, "finetune_started", {"round": 1}),
        (now - 4000, "finetune_completed", {"round": 1}),
    )
    write_events(runs / "died-tuning", (now - 6000, "episode_started", {}), (now - 1900, "finetune_started", {}))
    write_events(runs / ".hidden", (now, "heartbeat", {}))  # not a valid run id
    write_events(runs / "two words", (now, "heartbeat", {}))
    (runs / "no-events").mkdir()
    (runs / "empty").mkdir()
    (runs / "empty" / "events.jsonl").write_text("not an event\n")  # nothing to show yet
    (runs / "stray.jsonl").touch()
    listed = runs_client(tmp_path).get("/runs").json()
    assert [(r["run_id"], r["status"]) for r in listed] == [
        ("fresh", "live"),
        ("fine-tuning", "live"),  # a fine-tune started under 30 minutes ago, not completed
        ("refused", "not_started"),
        ("old", "recorded"),
        ("tuned", "recorded"),
        ("died-tuning", "recorded"),  # its fine-tune started over 30 minutes ago and never completed
    ]
    by_id = {r["run_id"]: r for r in listed}
    assert by_id["refused"] == {
        "run_id": "refused",
        "instruction": "<b>juggle</b>",
        "status": "not_started",
        "reason": "Argon declined: no juggling; Not yet",
        "started_t": now - 3,
    }
    assert (by_id["fresh"]["instruction"], by_id["fresh"]["reason"]) == ("put it in", None)
    assert by_id["tuned"]["instruction"] is None and by_id["tuned"]["started_t"] == now - 5000


def test_the_runs_list_of_a_runs_directory_not_made_yet_is_empty(tmp_path: Path) -> None:
    assert runs_client(tmp_path).get("/runs").json() == []


SUMMARY = TrainingSetSummary(
    round=1,
    episodes_scanned=30,
    demonstrations=20,
    accepted=8,
    rejected=12,
    unfinished=0,
    rejected_by={"short_of_bin": 12},
    empty_dev_episodes=1,
    examples=56,
)
REPORT = ReplayReport(replayed=8, reproduced=7, failures=[ReplayFailure(episode_id="ep-3", reason="box moved")])


def test_run_files_serve_a_rounds_training_summary_and_the_replay_report_as_parsed(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "training_r12.summary.json").write_text(SUMMARY.model_dump_json(indent=2))
    (run_dir / "replay_report.json").write_text(REPORT.model_dump_json())
    client = runs_client(tmp_path)
    assert client.get("/run-files/r1/training_r12.summary.json").json() == SUMMARY.model_dump()
    assert client.get("/run-files/r1/replay_report.json").json() == REPORT.model_dump()


def test_run_files_are_only_the_summaries_and_the_replay_report_and_only_when_they_parse(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    (run_dir / "training_r1.summary.json").write_text('{"round": 1}')  # not a summary
    (run_dir / "training_r2.summary.json").write_text("{not json")
    (run_dir / "replay_report.json").write_bytes(b"\xff\xfe")
    for name in ["training_r1.jsonl", "training_r.summary.json", "training_r3.summary.json.bak", "events.jsonl"]:
        (run_dir / name).write_text(SUMMARY.model_dump_json())
    (run_dir / "training_r\u0661.summary.json").write_text(SUMMARY.model_dump_json())  # an Arabic-Indic one
    (tmp_path / "runs" / "summary.json").write_text(SUMMARY.model_dump_json())
    client = runs_client(tmp_path)
    for path in [
        "r1/training_r1.summary.json",
        "r1/training_r2.summary.json",
        "r1/replay_report.json",
        "r1/training_r9.summary.json",
        "r1/training_r1.jsonl",
        "r1/training_r.summary.json",
        "r1/training_r3.summary.json.bak",
        "r1/events.jsonl",
        "r1/training_r\u0661.summary.json",
        "r1/%2e%2e%2fsummary.json",
        "r9/training_r1.summary.json",
        "%2e%2e/summary.json",
    ]:
        assert client.get(f"/run-files/{path}").status_code == 404, path


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
        async for message in tail_events(run_dir, stopped, poll=0):
            assert message.endswith("\n\n")
            seen.append(message)

    asyncio.run(read())
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


def test_stream_turns_frame_paths_into_their_runs_urls_and_hides_outside_paths(tmp_path: Path) -> None:
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
        "/frames/r1/episodes/ep-1/frames/0000.jpg",
        None,
        None,
        None,
        "/frames/r1/episodes/ep-1/before_move_0000.jpg",
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
    client = TestClient(create_app(served.parent))
    assert urls and all(
        client.get(url).content == (served / url.removeprefix("/frames/r1/")).read_bytes() for url in urls
    )


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
    client = runs_client(tmp_path)
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
