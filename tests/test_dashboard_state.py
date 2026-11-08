"""Behaviour of the dashboard's event reducer (static/state.js), run under node."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

STATE_JS = Path(__file__).parent.parent / "isaacs_mom" / "orchestrator" / "dashboard" / "static" / "state.js"
DRIVER = """
const S = require(process.argv[1]);
const { events, script } = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(new Function("S", "events", script)(S, events)));
"""
FOLD = """
const state = S.initial();
for (const e of events) S.reduce(state, e);
const ep = S.current(state);
return { state, ep, summary: S.summary(state), filmstrip: S.filmstrip(ep) };
"""

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def stream(*events: tuple[str, dict]) -> list[dict]:
    return [{"t": float(i), "run_id": "r", "type": type_, "data": data} for i, (type_, data) in enumerate(events)]


def run(script: str, events: list[dict]) -> Any:
    """Run `script` (a JS function body with `S` = state.js and `events`) under node and return its result."""
    stdin = json.dumps({"events": events, "script": script})
    out = subprocess.run(["node", "-e", DRIVER, str(STATE_JS)], input=stdin, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fold(*events: tuple[str, dict]) -> dict:
    return run(FOLD, stream(*events))


def decision(step: int, controller: str = "student", latency: object = 100.0, reason: str = "why") -> tuple[str, dict]:
    skill = {"skill": "hug", "params": {}}
    return "decision", {
        "controller": controller,
        "checkpoint": "ckpt-r0",
        "step": step,
        "skill": skill,
        "reason": reason,
        "latency_ms": latency,
    }


def executed(step: int, ok: bool = True, frame: str | None = "/frames/a.jpg") -> tuple[str, dict]:
    return "action_executed", {"step": step, "ok": ok, "error": None if ok else "joint limit", "frame": frame}


def test_an_action_attaches_to_its_decision_and_keeps_a_long_reason() -> None:
    reason = "r" * 120
    out = fold(("episode_started", {"episode_id": "e", "seed": 1}), decision(0, reason=reason), executed(0, ok=False))
    (row,) = out["ep"]["rows"]
    assert (row["reason"], row["ok"], row["error"]) == (reason, False, "joint limit")
    assert out["ep"]["frames"][0]["url"] == "/frames/a.jpg"


def test_a_decision_keeps_its_reach_distance_or_turn_angle() -> None:
    turn = (
        "decision",
        {"controller": "student", "step": 0, "skill": {"skill": "turn_left", "params": {"angle_deg": 30}}},
    )
    reach = (
        "decision",
        {"controller": "student", "step": 1, "skill": {"skill": "reach_left", "params": {"distance_cm": 5}}},
    )
    rows = fold(started("a"), turn, reach)["ep"]["rows"]
    assert [(r["skill"], r["distance"], r["angle"]) for r in rows] == [("turn_left", None, 30), ("reach_left", 5, None)]


def test_median_latency_carries_its_sample_size_per_controller() -> None:
    out = fold(
        ("episode_started", {"episode_id": "e"}),
        decision(0, latency=10.0),
        decision(1, latency=30.0),
        decision(2, latency=20.0),
        decision(3, controller="teacher", latency=500.0),
    )
    assert out["summary"]["student"] == {"median": 20.0, "n": 3}
    assert out["summary"]["teacher"] == {"median": 500.0, "n": 1}
    assert out["summary"]["finetune"] is None


def test_latency_medians_count_only_decisions_that_chose_a_skill() -> None:
    stop = ("decision", {"controller": "teacher", "step": 2, "skill": None, "reason": "done", "latency_ms": 0.0})
    out = fold(
        ("episode_started", {"episode_id": "e"}),
        decision(0, controller="teacher", latency=800.0),
        decision(1, controller="teacher", latency=1200.0),
        stop,
    )
    assert out["summary"]["teacher"] == {"median": 1000.0, "n": 2}


def test_missing_or_bad_latency_is_not_counted() -> None:
    out = fold(("episode_started", {"episode_id": "e"}), decision(0, latency=None), decision(1, latency="fast"))
    assert out["summary"]["student"] is None


def test_the_first_frame_after_the_perturbation_is_flagged_and_every_later_one_marked_after_it() -> None:
    out = fold(
        ("episode_started", {"episode_id": "e"}),
        decision(0),
        executed(0),
        decision(1),
        ("perturbation", {"kind": "move_box", "step": 2}),
        decision(2),
        executed(1),
        executed(2),
        decision(3),
        executed(3),
    )
    assert [f["moved"] for f in out["filmstrip"]] == [False, False, True, False]
    assert [f["after"] for f in out["filmstrip"]] == [False, False, True, True]


def test_rounds_keep_their_n_and_a_new_episode_replaces_the_current_one() -> None:
    out = fold(
        ("evaluation_result", {"checkpoint": "ckpt-r0", "round": 0, "successes": 0, "n": 20}),
        ("evaluation_result", {"checkpoint": "ckpt-r1", "round": 1, "successes": 3, "n": 20}),
        ("evaluation_result", {"checkpoint": "bad", "round": 2, "successes": "x", "n": 20}),
        ("episode_started", {"episode_id": "a"}),
        decision(0),
        ("episode_started", {"episode_id": "b"}),
    )
    assert [(r["checkpoint"], r["successes"], r["n"]) for r in out["state"]["rounds"]] == [
        ("ckpt-r0", 0, 20),
        ("ckpt-r1", 3, 20),
    ]
    assert out["ep"]["id"] == "b" and out["ep"]["rows"] == []
    assert len(out["state"]["episodes"]) == 2


def test_finetune_duration_is_measured_between_its_events() -> None:
    out = fold(("finetune_started", {"round": 1}), ("finetune_completed", {"round": 1, "checkpoint": "ckpt-r1"}))
    assert out["summary"]["finetune"] == {"median": 1000.0, "n": 1}
    assert out["state"]["finetune"]["pending"] is None


def test_takeover_error_and_verdict_rows_appear_in_order() -> None:
    out = fold(
        ("episode_started", {"episode_id": "e"}),
        ("decision", {"controller": "teacher", "takeover": True, "step": 5, "skill": None}),
        ("error", {"message": "Argon unreachable"}),
        ("validation", {"task_success": False, "physics_valid": True, "reason": "far"}),
    )
    kinds = [r["kind"] for r in out["ep"]["rows"]]
    assert kinds == ["takeover", "error", "verdict"]
    assert out["ep"]["verdict"] == {"task": False, "physics": True, "reason": "far"}


def started(episode_id: str, sim_version: str | None = "mujoco") -> tuple[str, dict]:
    data = {"episode_id": episode_id, "seed": 1, "split": "dev", "round": 0}
    return "episode_started", data if sim_version is None else {
        **data,
        "sim_version": sim_version,
        "robot_config": "g1",
    }


def test_a_reconnect_replays_the_stream_into_a_fresh_state_not_on_top_of_the_old_one() -> None:
    events = stream(
        started("a"),
        decision(0),
        executed(0),
        started("b"),
        decision(0, latency=50.0),
        ("evaluation_result", {"checkpoint": "ckpt-r0", "round": 0, "successes": 0, "n": 20}),
    )
    script = """
    const v = S.view();
    for (const e of events) S.reduce(v.live, e);
    S.reconnect(v);
    for (const e of events) S.reduce(v.live, e);
    return { state: v.live, summary: S.summary(v.live) };
    """
    out = run(script, events)
    assert out["state"]["eventCount"] == len(events)
    assert [e["id"] for e in out["state"]["episodes"]] == ["a", "b"]
    assert len(out["state"]["rounds"]) == 1
    assert out["summary"]["student"] == {"median": 75.0, "n": 2}


def test_a_diagnosis_stays_with_the_episode_it_diagnoses() -> None:
    diagnosis = {"hypothesis": "pushed too early", "evidence": "block left of hand", "confidence": 0.7, "lesson": ""}
    out = fold(
        started("a"),
        started("b"),
        ("diagnosis", {"episode_id": "a", **diagnosis}),
        ("diagnosis", {"episode_id": "gone", **diagnosis}),
    )
    a, b = out["state"]["episodes"]
    assert a["diagnosis"]["hypothesis"] == "pushed too early" and a["diagnosis"]["confidence"] == 0.7
    assert b["diagnosis"] is None


def provenance(
    events: list[dict], now: float, episode: int | None = None, paused: bool = False, lost: bool = False
) -> str:
    script = f"""
    const v = S.view();
    for (const e of events) S.reduce(v.live, e);
    if ({"true" if paused else "false"}) S.keydown(v, {{ key: " " }}, {now});
    if ({"true" if lost else "false"}) S.disconnect(v);
    return S.provenance(v, {now}, {"null" if episode is None else f"S.shown(v).episodes[{episode}]"});
    """
    return run(script, events)


def test_a_fakesim_or_unstated_simulator_is_placeholder_even_while_events_arrive() -> None:
    assert provenance(stream(started("a", "fake-1")), now=1.0) == "placeholder"
    assert provenance(stream(started("a", None)), now=1.0) == "placeholder"
    assert provenance(stream(started("a", "fake-1"), started("b")), now=2.0) == "placeholder"
    assert provenance([], now=1.0) == "placeholder"


def test_a_real_run_is_live_while_events_arrive_or_a_finetune_is_pending_and_recorded_otherwise() -> None:
    events = stream(started("a"), started("b"))  # last event at t=1
    assert provenance(events, now=5.0) == "live"
    assert provenance(events, now=500.0) == "recorded"
    assert provenance(events, now=5.0, episode=0) == "recorded"  # a finished episode is not live
    assert provenance(events, now=5.0, episode=1) == "live"
    pending = stream(started("a"), ("finetune_started", {"round": 1}))
    assert provenance(pending, now=500.0) == "live"


def test_an_episode_with_a_verdict_is_recorded_even_while_its_run_is_live() -> None:
    events = stream(started("a"), started("b"), verdict(False))  # "b" judged at t=2
    assert provenance(events, now=5.0, episode=1) == "recorded"
    assert provenance(events, now=5.0) == "live"  # the run itself still is
    pending = stream(started("a"), verdict(True), ("finetune_started", {"round": 1}))
    assert provenance(pending, now=500.0, episode=0) == "recorded"
    assert provenance(pending, now=500.0, episode=0, paused=True) == "recorded"


def test_nothing_is_live_while_the_event_stream_is_disconnected() -> None:
    events = stream(started("a"), started("b"))
    assert provenance(events, now=5.0, lost=True) == "recorded"
    assert provenance(events, now=5.0, episode=1, lost=True) == "recorded"
    pending = stream(started("a"), ("finetune_started", {"round": 1}))
    assert provenance(pending, now=500.0, lost=True) == "recorded"
    assert provenance(pending, now=500.0, paused=True, lost=True) == "recorded"
    script = """
    const v = S.view();
    for (const e of events) S.reduce(v.live, e);
    S.disconnect(v);
    S.reconnect(v);
    for (const e of events) S.reduce(v.live, e);
    return S.provenance(v, 5);
    """
    assert run(script, events) == "live"


def test_a_run_stops_being_live_20_s_after_its_last_event() -> None:
    events = stream(started("a"))  # last event at t=0
    assert provenance(events, now=19.9) == "live"
    assert provenance(events, now=20.1) == "recorded"


def beats(*times: float) -> list[dict]:
    return [{"t": t, "run_id": "r", "type": "heartbeat", "data": {"episode_id": "a"}} for t in times]


def test_heartbeats_keep_a_slow_step_live_and_a_run_whose_heartbeats_stop_goes_recorded_after_20_s() -> None:
    step = [*stream(started("a"), decision(0)), *beats(10, 20, 30, 40), {**stream(executed(0))[0], "t": 45.0}]
    every_second = """
    const seen = new Set();
    for (let now = 1; now <= 45; now++) {
      const v = S.view();
      for (const e of events) if (e.t <= now) S.reduce(v.live, e);
      seen.add(S.provenance(v, now, S.shown(v).episodes[0]));
    }
    return [...seen];
    """
    assert run(every_second, step) == ["live"]
    assert run(every_second, [e for e in step if e["type"] != "heartbeat"]) == ["live", "recorded"]
    dead = [*stream(started("a"), decision(0)), *beats(10, 20)]
    assert provenance(dead, now=39.9) == "live"
    assert provenance(dead, now=40.1) == "recorded"


def test_a_heartbeat_is_no_row_no_buffered_event_and_never_moves_the_view() -> None:
    events = [*stream(started("a"), decision(0), executed(0)), *beats(10, 20, 30)]
    out = presenter(
        """
        feed(3);
        const before = [look(), S.shown(v).episodes[0].rows.length];
        press(" "); feed(3);
        const paused = look();
        press(" ");
        return [before, paused, look(), S.shown(v).episodes[0].rows.length, S.shown(v).eventCount];
        """,
        events,
    )
    before, paused, resumed, rows, count = out
    assert before[0] == {"shown": 3, "held": 0, "episode": "a", "focus": None} and before[1] == 1
    assert paused == {"shown": 3, "held": 0, "episode": "a", "focus": None}
    assert (resumed, rows, count) == (before[0], 1, 3)


def test_a_pending_finetune_keeps_a_run_live_for_at_most_30_minutes() -> None:
    pending = stream(started("a"), ("finetune_started", {"round": 1}))  # started at t=1
    assert provenance(pending, now=1 + 1799.9) == "live"
    assert provenance(pending, now=1 + 1800.1) == "recorded"
    done = stream(started("a"), ("finetune_started", {"round": 1}), ("finetune_completed", {"round": 1}))
    assert provenance(done, now=100.0) == "recorded"


def test_a_paused_view_of_a_live_run_is_labelled_paused_not_recorded() -> None:
    events = stream(started("a"), started("b"))
    assert provenance(events, now=5.0, paused=True) == "paused"
    assert provenance(events, now=5.0, episode=1, paused=True) == "paused"
    assert provenance(events, now=5.0, episode=0, paused=True) == "recorded"  # a finished episode stays recorded
    assert provenance(events, now=500.0, paused=True) == "recorded"
    assert provenance(stream(started("a", "fake-1")), now=1.0, paused=True) == "placeholder"


def test_an_unknown_controller_is_dropped_at_the_boundary() -> None:
    out = fold(started("a"), decision(0, controller='"><img src=x onerror=alert(1)>'), executed(0))
    (row,) = out["ep"]["rows"]
    assert row["controller"] is None and out["ep"]["controller"] is None
    assert out["ep"]["frames"][0]["controller"] is None
    assert out["summary"]["student"] is None and out["summary"]["teacher"] is None


def test_each_frame_keeps_the_controller_that_saw_it_after_a_takeover() -> None:
    out = fold(
        started("a"),
        decision(4),
        executed(4),
        ("decision", {"controller": "teacher", "takeover": True, "step": 5, "skill": None}),
    )
    (frame,) = out["ep"]["frames"]
    assert (frame["step"], frame["controller"], frame["checkpoint"]) == (4, "student", "ckpt-r0")
    assert out["ep"]["controller"] == "teacher"


VIEW = """
const v = S.view();
let now = 0;
// Each event arrives a second after it was written, as from a live run.
const feed = (n) => { for (const e of events.splice(0, n)) { now = e.t + 1; S.receive(v, e, now); } };
// The page opens on a run that finished a minute ago: every event arrives at once.
const open = () => { now = events[events.length - 1].t + 60; for (const e of events.splice(0)) S.receive(v, e, now); };
const press = (key) => S.keydown(v, { key }, now);
const look = () => {
  const ep = S.episode(v, now);
  const shown = S.shown(v).eventCount;
  return { shown, held: S.buffered(v), episode: ep && ep.id, focus: S.stageStep(v) };
};
"""


def presenter(script: str, events: list[dict]) -> Any:
    return run(VIEW + script, events)


def test_pausing_holds_the_shown_view_and_buffers_new_events_until_resume() -> None:
    events = stream(started("a"), decision(0), started("b"), decision(0), decision(1))
    out = presenter(
        "feed(2); press(' '); feed(3); const paused = look(); press(' '); return [paused, look()];",
        events,
    )
    assert out[0] == {"shown": 2, "held": 3, "episode": "a", "focus": None}
    assert out[1] == {"shown": 5, "held": 0, "episode": "b", "focus": None}


def test_the_pause_key_is_named_by_its_label_and_counts_events_buffered_since_the_pause() -> None:
    events = stream(started("a"), decision(0), started("b"), decision(0))
    out = presenter(
        """
        const all = [...events];
        const labels = [S.pauseLabel(v)];
        feed(2); press(" ");
        labels.push(S.pauseLabel(v));
        feed(2); labels.push(S.pauseLabel(v));
        S.reconnect(v); labels.push(S.pauseLabel(v));  // the replay starts from nothing
        for (const e of all) S.receive(v, e, now);
        labels.push(S.pauseLabel(v));
        press(" "); labels.push(S.pauseLabel(v));
        return labels;
        """,
        events,
    )
    assert out == ["Pause", "Resume · 0 new", "Resume · 2 new", "Resume · 0 new", "Resume · 2 new", "Pause"]


def test_a_reconnect_while_paused_keeps_the_frozen_view_and_resumes_onto_the_replayed_run() -> None:
    events = stream(started("a"), decision(0), started("b"))
    out = presenter(
        """
        const all = [...events];
        feed(2); press(" "); S.reconnect(v);
        for (const e of all) S.reduce(v.live, e);
        const paused = look(); press(" "); return [paused, look()];
        """,
        events,
    )
    assert out[0] == {"shown": 2, "held": 1, "episode": "a", "focus": None}
    assert out[1] == {"shown": 3, "held": 0, "episode": "b", "focus": None}


def test_arrows_move_between_episodes_and_moving_past_the_last_follows_the_run_again() -> None:
    events = stream(started("a"), started("b"), started("c"), started("d"))
    out = presenter(
        """
        feed(3);
        const seen = [];
        for (const key of ["ArrowLeft", "ArrowLeft", "ArrowLeft", "ArrowRight"]) {
          press(key);
          seen.push(look().episode);
        }
        feed(1); seen.push(look().episode);
        press("ArrowRight"); press("ArrowRight"); seen.push(look().episode);
        return seen;
        """,
        events,
    )
    # pinned to "b" while "d" starts; then two steps right reach the latest and follow it
    assert out == ["b", "a", "a", "b", "b", "d"]


def test_move_shows_the_shown_episodes_move_and_pins_it_until_pressed_again() -> None:
    events = stream(
        started("a"),
        *[x for step in range(6) for x in (decision(step), executed(step, frame=f"/f/{step}.jpg"))],
        ("perturbation", {"kind": "move_box", "step": 2}),
        started("b"),
    )
    events.insert(5, events.pop(-2))  # the perturbation arrives before decision 2
    out = presenter(
        """
        feed(events.length - 1);
        press("m");
        const jumped = look();
        const strip = S.filmstrip(S.episode(v, now), S.stageStep(v)).map((f) => [f.step, f.moved, f.now]);
        feed(1);
        const pinned = look();
        press("M");
        return { jumped, strip, pinned, again: look() };
        """,
        events,
    )
    assert out["jumped"]["focus"] == 2 and out["jumped"]["episode"] == "a"
    assert [f for f in out["strip"] if f[2]] == [[2, True, True]]
    assert out["pinned"]["episode"] == "a"
    # pressing Move again returns to where the view was: following the run, now on "b"
    assert out["again"] == {"shown": len(events), "held": 0, "episode": "b", "focus": None}


def test_p_is_no_longer_a_presenter_key() -> None:
    events = stream(started("a"), decision(0), executed(0), ("perturbation", {"kind": "move_box", "step": 1}))
    out = presenter("feed(events.length); const claimed = press('p'); return [claimed, look()];", events)
    assert out == [False, {"shown": 4, "held": 0, "episode": "a", "focus": None}]


def test_move_pressed_again_returns_to_a_pinned_episode() -> None:
    events = stream(
        started("a"),
        started("b"),
        decision(0),
        executed(0),
        ("perturbation", {"kind": "move_box", "step": 1}),
        decision(1),
        executed(1),
        started("c"),
    )
    out = presenter(
        """
        feed(events.length);
        press("ArrowLeft");
        const before = look();
        press("m");
        const during = look();
        press("m");
        return [before, during, look()];
        """,
        events,
    )
    assert [o["episode"] for o in out] == ["b", "b", "b"]
    assert [o["focus"] for o in out] == [None, 1, None]


def test_move_does_nothing_in_a_run_without_a_move() -> None:
    out = presenter(
        "feed(3); const target = S.target(v, now, 'm'); press('m'); return [target, look()];",
        stream(started("a"), decision(0), executed(0)),
    )
    assert out == [-1, {"shown": 3, "held": 0, "episode": "a", "focus": None}]


def test_no_shortcut_acts_while_the_focus_is_in_a_text_field() -> None:
    out = presenter(
        """
        feed(events.length);
        press("ArrowLeft");
        const before = look();
        const field = { tagName: "INPUT" };
        const keys = ["d", "t", "m", "M", " ", "ArrowLeft", "ArrowRight", "End"];
        const claimed = keys.map((key) => S.keydown(v, { key, target: field }, now));
        const area = S.keydown(v, { key: " ", target: { tagName: "TEXTAREA" } }, now);
        const editable = S.keydown(v, { key: "m", target: { tagName: "DIV", isContentEditable: true } }, now);
        return { claimed, area, editable, paused: Boolean(v.frozen), moved: Boolean(v.compare), before, after: look() };
        """,
        stream(*BEATS),
    )
    assert out["claimed"] == [False] * 8 and not out["area"] and not out["editable"]
    assert not out["paused"] and not out["moved"]
    assert out["after"] == out["before"]


def test_space_pauses_from_any_focus_claims_the_key_and_acts_once_while_held() -> None:
    out = presenter(
        """
        feed(events.length);
        const button = { tagName: "BUTTON" };  // a focused presenter button sends the key too
        const claimed = S.keydown(v, { key: " ", target: button }, now);
        const paused = Boolean(v.frozen);
        const held = S.keydown(v, { key: " ", repeat: true, target: button }, now);
        const stillPaused = Boolean(v.frozen);
        const ctrl = S.keydown(v, { key: " ", ctrlKey: true }, now);
        const other = S.keydown(v, { key: "q" }, now);
        press(" ");
        return { claimed, paused, held, stillPaused, ctrl, other, resumed: !v.frozen };
        """,
        stream(started("a")),
    )
    assert out == {
        "claimed": True,
        "paused": True,
        "held": True,
        "stillPaused": True,
        "ctrl": False,
        "other": False,
        "resumed": True,
    }


def test_a_teacher_demonstration_finds_the_diagnosis_and_last_frame_of_the_episode_it_acts_on() -> None:
    diagnosis = {"hypothesis": "pushed too early", "evidence": "block left of hand", "confidence": 0.7, "lesson": ""}
    events = stream(
        started("trial"),
        decision(0),
        executed(0, frame="/f/0.jpg"),
        decision(1),
        executed(1, frame="/f/1.jpg"),
        started("other"),
        ("diagnosis", {"episode_id": "trial", **diagnosis}),
        ("episode_started", {"episode_id": "demo", "acting_on": "trial"}),
        ("episode_started", {"episode_id": "lost", "acting_on": "not-in-this-run"}),
    )
    out = run(
        """
        const s = S.initial();
        for (const e of events) S.reduce(s, e);
        const [trial, other, demo, lost] = s.episodes;
        const pick = (a) => a && {
          id: a.id, number: a.number, hypothesis: a.diagnosis && a.diagnosis.hypothesis, frame: a.frame && a.frame.url,
        };
        return [demo, lost, trial].map((ep) => pick(S.actedOn(s, ep)));
        """,
        events,
    )
    assert out == [
        {"id": "trial", "number": 1, "hypothesis": "pushed too early", "frame": "/f/1.jpg"},
        {"id": "not-in-this-run", "number": None, "hypothesis": None, "frame": None},
        None,
    ]


def verdict(accepted: bool) -> tuple[str, dict]:
    return "validation", {"task_success": accepted, "physics_valid": True, "reason": "ok" if accepted else "far"}


def demo(episode_id: str, acting_on: str | None, accepted: bool) -> list[tuple[str, dict]]:
    return [
        ("episode_started", {**started(episode_id)[1], "acting_on": acting_on}),
        decision(0, controller="teacher"),
        verdict(accepted),
    ]


STORY = [
    started("trial"),
    decision(0),
    verdict(False),
    ("diagnosis", {"episode_id": "trial", "hypothesis": "h", "evidence": "", "confidence": 0.5, "lesson": ""}),
    *demo("demo-1", "trial", accepted=True),
    *demo("demo-2", "trial", accepted=True),
    *demo("demo-3", "trial", accepted=False),
    *demo("takeover", None, accepted=True),
    started("heldout"),
    decision(0),
    verdict(False),
]


def test_a_finished_run_opens_on_its_latest_accepted_demonstration_that_acted_on_a_diagnosis() -> None:
    out = presenter("open(); return look().episode;", stream(*STORY))
    assert out == "demo-2"


def test_a_page_opened_minutes_after_a_run_with_a_fine_tune_lands_on_the_story() -> None:
    finetune = [("finetune_started", {"round": 1}), ("finetune_completed", {"round": 1})]
    events = stream(*STORY[:-3], *finetune, *STORY[-3:])
    assert presenter("open(); return look().episode;", events) == "demo-2"


def test_the_story_skips_a_demonstration_whose_diagnosis_is_not_in_the_run() -> None:
    events = stream(*STORY[:-3], *demo("orphan", "not-in-this-run", accepted=True), *STORY[-3:])
    assert presenter("open(); return look().episode;", events) == "demo-2"


def test_a_finished_run_without_such_a_demonstration_opens_on_its_latest_episode() -> None:
    events = stream(*[e for e in STORY if e[0] != "episode_started" or e[1]["episode_id"] not in {"demo-1", "demo-2"}])
    assert presenter("open(); return look().episode;", events) == "heldout"


def test_a_view_following_a_live_run_stays_on_the_episode_it_showed_when_the_run_stops() -> None:
    out = presenter(
        """
        const all = [...events];
        feed(events.length);
        const live = look().episode;
        now += 21;
        const stopped = look().episode;
        S.reconnect(v);
        for (const e of all) S.receive(v, e, now);
        return [live, stopped, look().episode];
        """,
        stream(*STORY),
    )
    assert out == ["heldout", "heldout", "heldout"]


def test_a_paused_view_of_a_live_run_stays_put_when_the_run_ends_and_after_resuming() -> None:
    out = presenter(
        """
        feed(events.length);
        press(" ");
        now += 60;
        const paused = look().episode;
        press(" ");
        return [paused, look().episode];
        """,
        stream(*STORY),
    )
    assert out == ["heldout", "heldout"]


def test_while_the_student_fine_tunes_a_following_view_shows_the_time_since_it_started() -> None:
    events = stream(started("a"), verdict(True), started("b"), verdict(False), ("finetune_started", {"round": 1}))
    out = presenter(
        """
        feed(events.length);
        const tuning = S.finetuning(v, now + 90);
        press("ArrowLeft");
        const pinned = S.finetuning(v, now);
        press("End");
        const back = S.finetuning(v, now);
        S.receive(v, { t: now, run_id: "r", type: "finetune_completed", data: { round: 1 } }, now);
        return [tuning, pinned, back, S.finetuning(v, now)];
        """,
        events,
    )
    # finetune_started was written at t=4 and arrived at t=5
    assert out == [91, None, 1, None]


def trial(episode_id: str, diagnosed: bool = False, moved: bool = False) -> list[tuple[str, dict]]:
    events = [started(episode_id), decision(0)]
    if moved:
        events.append(("perturbation", {"kind": "move_box", "step": 1}))
    events.append(verdict(False))
    if diagnosed:
        events.append(
            ("diagnosis", {"episode_id": episode_id, "hypothesis": "h", "evidence": "", "confidence": 1, "lesson": ""})
        )
    return events


BEATS = [
    *trial("t0"),
    *trial("t1", diagnosed=True),
    *trial("t2"),
    *trial("t3", diagnosed=True),
    *demo("d4", "t3", accepted=False),
    *trial("t5", moved=True),
    *trial("t6"),
]


def test_story_keys_jump_to_the_nearest_other_episode_with_that_beat_and_ties_go_forward() -> None:
    out = presenter(
        """
        feed(events.length);
        const seen = [];
        for (const key of ["d", "d", "d", "t", "m", "ArrowLeft", "ArrowLeft", "d", "t", "t"]) {
          press(key);
          seen.push(look().episode);
        }
        return seen;
        """,
        stream(*BEATS),
    )
    # from t6: D t3, D t1, D t3, T d4, M t5, <- d4, <- t3, D t1, T d4; T again finds no other demonstration
    assert out == ["t3", "t1", "t3", "d4", "t5", "d4", "t3", "t1", "d4", "d4"]


def test_d_from_between_two_diagnoses_goes_forward_on_a_tie() -> None:
    out = presenter(
        """
        feed(events.length);
        for (const key of ["ArrowLeft", "ArrowLeft", "ArrowLeft", "ArrowLeft"]) press(key);
        const at = look().episode;
        press("d");
        return [at, look().episode];
        """,
        stream(*BEATS),
    )
    assert out == ["t2", "t3"]


def test_end_follows_a_live_run_and_lands_on_a_finished_runs_latest_episode() -> None:
    out = presenter(
        """
        const tail = events.splice(events.length - 3);
        feed(events.length);
        press("ArrowLeft"); press("End");
        const live = look().episode;
        events.push(...tail); feed(3);
        const followed = look().episode;
        now += 60;
        press("ArrowLeft");
        const back = look().episode;
        press("End");
        return [live, followed, back, look().episode];
        """,
        stream(*BEATS),
    )
    assert out == ["t5", "t6", "t5", "t6"]


def test_navigation_targets_say_where_each_key_leads_and_minus_one_for_nowhere_new() -> None:
    out = presenter(
        """
        feed(events.length);
        const keys = ["ArrowLeft", "ArrowRight", "d", "t", "m", "End"];
        const at = () => Object.fromEntries(keys.map((k) => [k, S.target(v, now, k)]));
        const latest = at();
        press("d"); press("d");
        return [latest, at()];
        """,
        stream(*BEATS),
    )
    assert out[0] == {"ArrowLeft": 5, "ArrowRight": -1, "d": 3, "t": 4, "m": 5, "End": -1}
    assert out[1] == {"ArrowLeft": 0, "ArrowRight": 2, "d": 3, "t": 4, "m": 5, "End": 6}


def test_move_from_an_episode_without_one_opens_the_nearest_moves_before_and_after_and_again_returns() -> None:
    events = stream(
        started("a"),
        *[x for step in range(6) for x in (decision(step), executed(step, frame=f"/f/{step}.jpg"))],
        ("perturbation", {"kind": "move_box", "step": 2, "before_frame": "/f/before.jpg"}),
        *trial("b"),
        *trial("c"),
    )
    events.insert(5, events.pop(-7))  # the perturbation arrives before decision 2
    out = presenter(
        """
        feed(events.length);
        press("ArrowLeft");
        const pair = () => { const m = S.move(S.episode(v, now)); return v.compare && [m.before.url, m.after.url]; };
        press("m");
        const opened = [look(), pair()];
        press("m");
        const back = [look(), pair()];
        press("m"); press("End");
        return [opened, back, look()];
        """,
        events,
    )
    assert out[0] == [{"shown": len(events), "held": 0, "episode": "a", "focus": 2}, ["/f/before.jpg", "/f/2.jpg"]]
    # Move again returns to where the view was, the pinned "b", not to the moved episode
    assert out[1] == [{"shown": len(events), "held": 0, "episode": "b", "focus": None}, None]
    # End leaves the move for the latest episode
    assert out[2] == {"shown": len(events), "held": 0, "episode": "c", "focus": None}


def test_the_judged_take_is_marked_on_its_episode() -> None:
    out = fold(("episode_started", {**started("take")[1], "demo": True}), started("trial"))
    assert [e["demo"] for e in out["state"]["episodes"]] == [True, False]


def test_the_move_pairs_the_frame_taken_right_before_it_with_the_first_after_it_and_the_first_decision_after() -> None:
    def moved(**before: str) -> list[dict]:
        return stream(
            started("a"),
            decision(0),
            executed(0, frame="/f/0.jpg"),
            ("perturbation", {"kind": "move_bin", "step": 1, **before}),
            decision(1),
            executed(1, frame="/f/1.jpg"),
            decision(2),
            executed(2, frame="/f/2.jpg"),
        )

    script = """
    const s = S.initial();
    for (const e of events) S.reduce(s, e);
    const m = S.move(S.current(s));
    const who = (f) => [f.step, f.url, f.controller, f.checkpoint];
    return { before: who(m.before), after: who(m.after), decision: m.decision.step, none: S.move(null) };
    """
    out = run(script, moved(before_frame="/f/before.jpg"))
    assert out == {
        "before": [1, "/f/before.jpg", "student", "ckpt-r0"],
        "after": [1, "/f/1.jpg", "student", "ckpt-r0"],
        "decision": 1,
        "none": None,
    }
    # A run recorded before the orchestrator took that frame shows it missing, never the previous step's frame.
    assert run(script, moved())["before"] == [1, None, "student", "ckpt-r0"]


def test_a_move_before_the_first_decision_names_the_controller_of_the_first_frame_after_it() -> None:
    events = stream(
        started("a"),
        ("perturbation", {"kind": "move_box", "step": 0, "before_frame": "/f/before.jpg"}),
        decision(0, controller="teacher"),
        executed(0, frame="/f/0.jpg"),
    )
    script = "const s = S.initial(); for (const e of events) S.reduce(s, e); return S.move(S.current(s)).before;"
    before = run(script, events)
    assert (before["url"], before["controller"], before["checkpoint"]) == ("/f/before.jpg", "teacher", "ckpt-r0")


def test_verdicts_are_worded_by_episode_kind_and_a_judged_take_says_who_finished_it() -> None:
    def kinded(episode_id: str, **data: object) -> tuple[str, dict]:
        return "episode_started", {**started(episode_id)[1], **data}

    took = ("decision", {"controller": "teacher", "takeover": True, "step": 8, "skill": None})
    events = stream(
        *demo("demo-ok", "trial", accepted=True),
        *demo("demo-bad", "trial", accepted=False),
        *[kinded("heldout-ok", split="heldout"), decision(0), verdict(True)],
        *[kinded("heldout-bad", split="heldout"), decision(0), verdict(False)],
        *[started("trial-ok"), decision(0), verdict(True)],
        *[started("trial-bad"), decision(0), verdict(False)],
        *[kinded("alone", demo=True), decision(0), verdict(True)],
        *[kinded("rescued", demo=True), decision(0), took, decision(8, controller="teacher"), verdict(True)],
        *[kinded("lost", demo=True), decision(0), took, decision(8, controller="teacher"), verdict(False)],
        *[started("open"), decision(0)],
    )
    out = run(
        """
        const s = S.initial();
        for (const e of events) S.reduce(s, e);
        return s.episodes.map((e) => [S.kind(e), S.verdictWords(e)]);
        """,
        events,
    )
    assert out == [
        ["demonstration", "Accepted for training"],
        ["demonstration", "Rejected, not used for training"],
        ["heldout", "Succeeded"],
        ["heldout", "Failed"],
        ["trial", "Succeeded"],
        ["trial", "Failed"],
        ["take", "Succeeded, by the student alone"],
        ["take", "Succeeded after the teacher took over at step 8"],
        ["take", "Failed after the teacher took over at step 8"],
        ["trial", None],
    ]


def test_an_episodes_kind_is_known_from_its_start_except_a_trial_which_its_first_decision_tells() -> None:
    def kinded(episode_id: str, **data: object) -> tuple[str, dict]:
        return "episode_started", {**started(episode_id)[1], **data}

    script = """
    const s = S.initial();
    const kinds = [];
    for (const e of events) { S.reduce(s, e); kinds.push(S.kind(S.current(s))); }
    return kinds;
    """
    events = stream(
        kinded("take", demo=True),
        kinded("heldout", split="heldout"),
        kinded("demo", acting_on="trial"),
        started("trial"),
        decision(0),
        started("rescue"),
        decision(0, controller="teacher"),
    )
    assert run(script, events) == ["take", "heldout", "demonstration", None, "trial", None, "demonstration"]


TAKE = [
    ("episode_started", {**started("take")[1], "demo": True}),
    *[x for step in range(3) for x in (decision(step), executed(step, frame=f"/f/{step}.jpg"))],
    ("decision", {"controller": "teacher", "takeover": True, "step": 3, "skill": None}),
    ("decision", {**decision(3, controller="teacher")[1], "checkpoint": None}),  # the teacher has no checkpoint
    executed(3, frame="/f/3.jpg"),
]
TAKE.insert(3, ("perturbation", {"kind": "move_box", "step": 1, "before_frame": "/f/before.jpg"}))


def test_while_the_stage_holds_a_step_the_bar_names_that_frames_controller_not_the_episodes_latest() -> None:
    out = presenter(
        """
        feed(events.length);
        const who = () => S.inControl(v, now);
        const seen = [who()];
        for (const key of ["m", "m", "m", "End"]) { press(key); seen.push(who()); }
        return seen;
        """,
        stream(*TAKE, started("next")),
    )
    teacher = {"controller": "teacher", "checkpoint": None, "step": None}
    student_at_the_move = {"controller": "student", "checkpoint": "ckpt-r0", "step": 1}
    nobody_yet = {"controller": None, "checkpoint": None, "step": None}
    assert out == [nobody_yet, student_at_the_move, nobody_yet, student_at_the_move, nobody_yet]
    take = presenter(
        "feed(events.length); const latest = S.inControl(v, now); press('m'); return [latest, S.inControl(v, now)];",
        stream(*TAKE),
    )
    assert take == [teacher, student_at_the_move]


def evaluation(checkpoint: str, round_: int, successes: int, n: int) -> tuple[str, dict]:
    return "evaluation_result", {"checkpoint": checkpoint, "round": round_, "successes": successes, "n": n}


def heldout(episode_id: str, checkpoint: str, judged: bool = True) -> list[tuple[str, dict]]:
    events = [("episode_started", {**started(episode_id)[1], "split": "heldout"}), decision(0)]
    events[-1][1]["checkpoint"] = checkpoint
    return [*events, verdict(False)] if judged else events


CLAIM = """
const s = S.initial();
for (const e of events) S.reduce(s, e);
return [S.claim(s), S.testing(s)];
"""


def test_the_held_out_claim_names_its_rounds_in_words_with_the_checkpoints_as_a_note() -> None:
    first = run(CLAIM, stream(*heldout("h1", "ckpt-r0"), evaluation("ckpt-r0", 0, 0, 20)))
    assert first == [
        {
            "figure": "0/20",
            "spoken": "0 of 20 before\u00a0fine-tuning",
            "sample": "n=20",
            "names": "before\u00a0fine-tuning",
            "ids": "ckpt-r0",
        },
        None,
    ]
    rounds = [evaluation("ckpt-r0", 0, 0, 20), evaluation("ckpt-r1", 1, 3, 20), evaluation("ckpt-r2", 2, 9, 20)]
    claim, _ = run(CLAIM, stream(*rounds))
    assert claim == {
        "figure": "0/20 → 9/20",
        "spoken": "0 of 20 before\u00a0fine-tuning, then 9 of 20 after\u00a0round\u00a02",
        "sample": "n=20 each",
        "names": "before\u00a0fine-tuning → after\u00a0round\u00a02",
        "ids": "ckpt-r0 → ckpt-r2",
    }


def test_a_held_out_evaluation_with_no_episodes_says_so_in_words() -> None:
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 0, 20), evaluation("ckpt-r1", 1, 0, 0)))
    assert (claim["figure"], claim["sample"]) == ("0/20 → none", "n=20 and no held-out episode ran")
    assert claim["spoken"] == "0 of 20 before\u00a0fine-tuning, then no held-out episode ran after\u00a0round\u00a01"


def test_a_running_evaluation_counts_its_judged_episodes_out_of_the_previous_evaluations_n() -> None:
    _, first = run(CLAIM, stream(*heldout("h1", "ckpt-r0"), *heldout("h2", "ckpt-r0", judged=False)))
    assert first == {"checkpoint": "ckpt-r0", "run": 1, "of": None}  # nothing in the events says 20 yet
    later = stream(
        *heldout("h1", "ckpt-r0"),
        evaluation("ckpt-r0", 0, 0, 20),
        started("trial"),
        decision(0),
        verdict(False),
        *heldout("h2", "ckpt-r1"),
        *heldout("h3", "ckpt-r1"),
        started("take-is-dev"),
    )
    assert run(CLAIM, later)[1] == {"checkpoint": "ckpt-r1", "run": 2, "of": 20}
    done = stream(*heldout("h1", "ckpt-r0"), evaluation("ckpt-r0", 0, 0, 20), started("trial"))
    assert run(CLAIM, done)[1] is None


def test_an_episode_is_finished_once_judged_moved_past_or_its_run_has_stopped() -> None:
    out = presenter(
        """
        feed(events.length);
        const [a, b, c] = S.shown(v).episodes;
        const live = [a, b, c].map((e) => S.finished(v, now, e));
        now += 60;
        return [live, S.finished(v, now, c)];
        """,
        stream(started("a"), decision(0), started("b"), decision(0), verdict(False), started("c"), decision(0)),
    )
    assert out == [[True, True, False], True]


def test_the_simulator_is_named_in_words_with_its_full_version_and_task_kept_for_the_legend() -> None:
    script = "const s = S.initial(); for (const e of events) S.reduce(s, e); return S.simulator(s);"
    walking = run(script, stream(started("a", "mujoco-4.0.0 task=across_room")))
    assert walking == {"name": "MuJoCo", "version": "mujoco-4.0.0 task=across_room", "task": "across_room"}
    assert run(script, stream(started("a", "mujoco-3.3.7")))["name"] == "MuJoCo"
    assert run(script, stream(started("a", "fake-3"), started("b"))) == {
        "name": "FakeSim",
        "version": "fake-3",
        "task": None,
    }
    assert run(script, stream(started("a", None))) == {"name": None, "version": None, "task": None}
    other = run(script, stream(started("a", "isaac-sim-4.5.0-preview.2 task=across_room")))
    assert (other["name"], other["task"]) == ("isaac-sim", "across_room")
    assert run(script, []) is None


def test_the_fine_tune_shows_each_accepted_demonstrations_last_frame_and_counts_every_judged_demonstration() -> None:
    def demonstrated(episode_id: str, frames: list[str | None], accepted: bool | None) -> list[tuple[str, dict]]:
        steps = [x for i, url in enumerate(frames) for x in (decision(i, controller="teacher"), executed(i, frame=url))]
        return [("episode_started", {**started(episode_id)[1], "acting_on": "t"}), *steps] + (
            [] if accepted is None else [verdict(accepted)]
        )

    events = stream(
        started("t"),
        decision(0),
        executed(0, frame="/f/t.jpg"),
        verdict(True),  # an accepted student trial is not a lesson
        *demonstrated("kept", ["/f/k0.jpg", "/f/k1.jpg"], accepted=True),
        *demonstrated("blind", [], accepted=True),  # no frame to show, but counted
        *demonstrated("dropped", ["/f/d0.jpg"], accepted=False),
        *demonstrated("missing", [None], accepted=True),
        *demonstrated("unjudged", ["/f/u0.jpg"], accepted=None),
        ("finetune_started", {"round": 1}),
    )
    script = "const s = S.initial(); for (const e of events) S.reduce(s, e); return S.lessons(s);"
    assert run(script, events) == {
        "frames": [{"url": "/f/k1.jpg", "number": 2}, {"url": None, "number": 5}],
        "accepted": 3,
        "rejected": 1,
        "n": 4,
    }


def test_a_view_holds_still_from_each_connection_until_its_replayed_backlog_has_settled() -> None:
    out = presenter(
        """
        const seen = [v.settling];
        S.settle(v); seen.push(v.settling);
        S.disconnect(v); S.reconnect(v); seen.push(v.settling);  // a reconnect replays the run from its start
        S.disconnect(v); seen.push(v.settling);  // a lost stream shows that it is lost
        return seen;
        """,
        [],
    )
    assert out == [True, False, True, False]
