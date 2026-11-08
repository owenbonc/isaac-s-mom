"""Behaviour of the dashboard's event reducer (static/state.js), run under node."""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
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


def run(script: str, events: list[Any]) -> Any:
    """Run `script` (a JS function body with `S` = state.js and `events`) under node and return its result."""
    stdin = json.dumps({"events": events, "script": script})
    out = subprocess.run(["node", "-e", DRIVER, str(STATE_JS)], input=stdin, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fold(*events: tuple[str, dict]) -> dict:
    return run(FOLD, stream(*events))


def ckpt(round_: int) -> str:
    """The student's checkpoint id after `round_` as the producer writes it: the base bare, the rest with the run."""
    return "ckpt-r0" if round_ == 0 else f"r/ckpt-r{round_}"


def decision(
    step: int, controller: str = "student", latency: object = 100.0, reason: str = "why", round_: int = 0
) -> tuple[str, dict]:
    skill = {"skill": "hug", "params": {}}
    return "decision", {
        "controller": controller,
        "checkpoint": ckpt(round_),
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


def test_esc_turns_markup_quotes_and_non_strings_into_inert_text() -> None:
    script = """
    const markup = ["<img src=x onerror=alert(1)>", '"', "'", "&", "&lt;"];
    const other = [5, null, undefined, ["<b>"], { toString: () => "<script>" }];
    return [...markup, ...other].map((v) => S.esc(v));
    """
    assert run(script, []) == [
        "&lt;img src=x onerror=alert(1)&gt;",
        "&quot;",
        "&#39;",
        "&amp;",
        "&amp;lt;",
        "5",
        "null",
        "undefined",
        "&lt;b&gt;",
        "&lt;script&gt;",
    ]


def test_event_text_that_is_not_a_string_reads_as_none() -> None:
    out = fold(
        ("episode_started", {"episode_id": 7, "split": ["heldout"], "sim_version": {"v": 1}, "acting_on": 3}),
        ("decision", {"controller": "student", "checkpoint": 5, "step": 0, "skill": {"skill": 1}, "reason": ["<b>"]}),
        ("error", {"message": 4, "error": {"why": "<b>"}}),
    )
    ep = out["ep"]
    assert (ep["id"], ep["split"], ep["sim"], ep["actingOn"]) == ("", "", None, None)
    decided, error = ep["rows"]
    assert (decided["checkpoint"], decided["skill"], decided["reason"]) == (None, None, "")
    assert error["text"] == "error event"


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
        evaluation("ckpt-r0", 0, 0, 20),
        evaluation("ckpt-r1", 1, 3, 20, errored=2),
        ("evaluation_result", {"checkpoint": "bad", "round": 2, "successes": "x", "n": 20, "errored": 0}),
        ("evaluation_result", {"checkpoint": "no errored", "round": 2, "successes": 1, "n": 20}),
        ("evaluation_result", {"checkpoint": "bad errored", "round": 2, "successes": 1, "n": 20, "errored": "2"}),
        ("episode_started", {"episode_id": "a"}),
        decision(0),
        ("episode_started", {"episode_id": "b"}),
    )
    assert [(r["checkpoint"], r["successes"], r["n"], r.get("errored")) for r in out["state"]["rounds"]] == [
        ("ckpt-r0", 0, 20, 0),
        ("ckpt-r1", 3, 20, 2),
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
        evaluation("ckpt-r0", 0, 0, 20),
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


def test_only_an_episode_naming_fakesim_makes_a_run_placeholder_and_one_with_no_episode_is_live_or_recorded() -> None:
    assert provenance(stream(started("a", "fake-1")), now=1.0) == "placeholder"
    assert provenance(stream(started("a", "fake-1"), started("b")), now=2.0) == "placeholder"
    assert provenance(stream(started("a", None)), now=1.0) == "live"
    taught = stream(("instruction", {"text": "put the box in the bin", "source": "teach"}), argon_call(1))
    assert provenance(taught, now=2.0) == "live"
    assert provenance(taught, now=30.0) == "recorded"
    assert provenance([], now=100.0) == "recorded"


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


def panel(events: list[dict], now: float, episode: int | None = None) -> dict:
    script = f"""
    const v = S.view();
    for (const e of events) S.reduce(v.live, e);
    return S.panelLabel(v, {now}, {"null" if episode is None else f"S.shown(v).episodes[{episode}]"});
    """
    return run(script, events)


def test_a_panel_shows_its_provenance_as_a_dot_and_spells_it_only_where_it_differs_from_the_runs() -> None:
    events = stream(started("a"), verdict(False), started("b"))
    assert panel(events, now=500.0, episode=0) == {"kind": "recorded", "spelled": False}  # as the run is
    assert panel(events, now=5.0, episode=1) == {"kind": "live", "spelled": False}
    # the run is live, the judged episode a panel shows is recorded
    assert panel(events, now=5.0, episode=0) == {"kind": "recorded", "spelled": True}
    assert panel(stream(started("a", "fake-1")), now=1.0) == {"kind": "placeholder", "spelled": False}


def test_a_run_whose_mom_run_stopped_mid_fine_tune_reads_recorded_at_once() -> None:
    stopped = ("error", {"error": "mom run stopped (exit 1): torch.OutOfMemoryError: CUDA out of memory"})
    events = stream(started("a"), verdict(True), ("finetune_started", {"round": 1}), stopped)
    assert provenance(events, now=4.0) == "recorded"  # its last event is fresh and its fine-tune pending
    assert run(VIEW + "feed(events.length); return S.following(v, now);", events) is False


def test_a_run_that_stopped_before_its_first_episode_reads_not_started() -> None:
    stopped = ("error", {"error": "not started: mom run stopped: ModuleNotFoundError: No module named 'mujoco'"})
    events = stream(("instruction", {"text": "Put the box in the bin", "source": "teach"}), stopped)
    out = run(VIEW + "feed(events.length); return [S.shown(v).refused, S.following(v, now)];", events)
    assert out == ["mom run stopped: ModuleNotFoundError: No module named 'mujoco'", False]


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
    assert out == [None, {"shown": 3, "held": 0, "episode": "a", "focus": None}]


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


def test_end_follows_the_episode_in_progress_while_the_run_is_live_and_the_view_stays_there_once_it_stops() -> None:
    out = presenter(
        """
        const tail = events.splice(events.length - 3);
        feed(events.length);
        press("ArrowLeft");
        const pinned = look().episode;
        S.visit(v, { seed: 1 });
        press("End");
        const live = [S.where(v), v.selected, look().episode];
        events.push(...tail); feed(3);
        const followed = look().episode;
        now += 60;
        return [pinned, live, followed, look().episode, S.target(v, now, "End")];
        """,
        stream(*BEATS),
    )
    assert out == ["d4", ["run", None, "t5"], "t6", "t6", None]


def test_navigation_targets_say_where_each_key_leads_and_null_for_nowhere_new() -> None:
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
    pin = lambda id_: {"selected": id_, "seed": None, "opened": None}  # noqa: E731
    assert out[0] == {
        "ArrowLeft": pin("t5"),
        "ArrowRight": None,
        "d": pin("t3"),
        "t": pin("d4"),
        "m": pin("t5"),
        "End": None,
    }
    assert out[1] == {
        "ArrowLeft": pin("t0"),
        "ArrowRight": pin("t2"),
        "d": pin("t3"),
        "t": pin("d4"),
        "m": pin("t5"),
        "End": pin(None),
    }


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


RECORDING_REFUSED = ("validation", {"task_success": False, "physics_valid": True, "reason": "recording error: x"})


def test_verdicts_are_worded_by_episode_kind_and_a_judged_take_says_who_finished_it() -> None:
    def kinded(episode_id: str, **data: object) -> tuple[str, dict]:
        return "episode_started", {**started(episode_id)[1], **data}

    took = ("decision", {"controller": "teacher", "takeover": True, "step": 8, "skill": None})
    events = stream(
        *demo("demo-ok", "trial", accepted=True),
        *demo("demo-bad", "trial", accepted=False),
        *[kinded("heldout-ok", split="heldout"), decision(0), verdict(True)],
        *[kinded("heldout-bad", split="heldout"), decision(0), verdict(False)],
        *[kinded("heldout-errored", split="heldout"), decision(0), RECORDING_REFUSED],
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
        ["heldout", "Errored, not counted"],
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


def evaluation(checkpoint: str, round_: int, successes: int, n: int, errored: int = 0) -> tuple[str, dict]:
    data = {"checkpoint": checkpoint, "round": round_, "successes": successes, "n": n, "errored": errored}
    return "evaluation_result", data


def heldout(episode_id: str, checkpoint: str, judged: bool = True, seed: int = 1) -> list[tuple[str, dict]]:
    events = [("episode_started", {**started(episode_id)[1], "split": "heldout", "seed": seed}), decision(0)]
    events[-1][1]["checkpoint"] = checkpoint
    return [*events, verdict(False)] if judged else events


CLAIM = """
const s = S.initial();
for (const e of events) S.reduce(s, e);
return [S.claim(s), S.testing(s)];
"""


def test_the_held_out_claim_names_its_rounds_in_words_with_the_checkpoint_ids_beside_them() -> None:
    rounds = [evaluation(ckpt(0), 0, 4, 20), evaluation(ckpt(1), 1, 9, 20), evaluation(ckpt(2), 2, 15, 20)]
    claim, _ = run(CLAIM, stream(*rounds))
    # the counts carry n, so there is no n line, and nothing errored, so there is no errored note
    assert claim == {
        "figure": "+55 pts",
        "counts": "4 → 15 of 20",
        "spoken": "Up 55 percentage points: 20%, 4 of 20, before\u00a0fine-tuning, "
        "then 75%, 15 of 20, after\u00a0round\u00a02",
        "errored": None,
        "names": "before\u00a0fine-tuning → after\u00a0round\u00a02",
        "ids": "ckpt-r0 → r/ckpt-r2",
    }


def test_with_one_held_out_evaluation_the_figure_is_its_rate_with_its_count_of_n() -> None:
    first = run(CLAIM, stream(*heldout("h1", "ckpt-r0"), evaluation("ckpt-r0", 0, 4, 20)))
    assert first == [
        {
            "figure": "20%",
            "counts": "4 of 20",
            "spoken": "20%, 4 of 20, before\u00a0fine-tuning",
            "errored": None,
            "names": "before\u00a0fine-tuning",
            "ids": "ckpt-r0",
        },
        None,
    ]


@pytest.mark.parametrize(
    ("first", "latest", "figure", "counts", "spoken"),
    [
        ((15, 20), (13, 20), "\u221210 pts", "15 → 13 of 20", "Down 10 percentage points: "),
        ((0, 20), (0, 20), "+0 pts", "0 → 0 of 20", "No change: "),
        # The change is taken between the rates as shown (33% → 67%), never from the unrounded 33.3 points.
        ((1, 3), (2, 3), "+34 pts", "1 → 2 of 3", "Up 34 percentage points: "),
    ],
)
def test_the_held_out_change_is_in_points_between_the_rates_shown(
    first: tuple[int, int], latest: tuple[int, int], figure: str, counts: str, spoken: str
) -> None:
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, *first), evaluation("ckpt-r1", 1, *latest)))
    assert (claim["figure"], claim["counts"]) == (figure, counts)
    assert claim["spoken"].startswith(spoken)


def test_a_held_out_evaluation_with_no_episodes_says_so_in_words() -> None:
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 0, 20), evaluation("ckpt-r1", 1, 0, 0)))
    assert (claim["figure"], claim["counts"]) == (None, "0/20 → none")
    assert (
        claim["spoken"] == "0%, 0 of 20, before\u00a0fine-tuning, then no held-out episode ran after\u00a0round\u00a01"
    )


def test_held_out_counts_whose_denominators_differ_show_each_and_the_errored_counts_apart() -> None:
    rounds = [evaluation("ckpt-r0", 0, 4, 20, errored=0), evaluation("ckpt-r1", 1, 3, 20, errored=1)]
    claim, _ = run(CLAIM, stream(*rounds))
    assert (claim["figure"], claim["counts"], claim["errored"]) == ("\u22124 pts", "4/20 → 3/19", "0 → 1 errored")
    assert claim["spoken"] == (
        "Down 4 percentage points: 20%, 4 of 20, before\u00a0fine-tuning, "
        "then 16%, 3 of 19, after\u00a0round\u00a01, 1 errored"
    )
    same = run(CLAIM, stream(evaluation("ckpt-r0", 0, 4, 20, errored=2), evaluation("ckpt-r1", 1, 9, 20, errored=2)))
    assert (same[0]["counts"], same[0]["errored"]) == ("4 → 9 of 18", "2 → 2 errored")


def test_a_held_out_evaluation_whose_every_episode_errored_says_so_in_words_and_shows_no_points() -> None:
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 0, 20, errored=20)))
    assert (claim["figure"], claim["counts"], claim["errored"]) == (None, "none", "20 errored")
    assert claim["spoken"] == "all 20 held-out episodes errored before\u00a0fine-tuning"
    later, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 4, 20), evaluation("ckpt-r1", 1, 0, 20, errored=20)))
    assert (later["figure"], later["counts"]) == (None, "4/20 → none")


def test_the_held_out_rate_is_out_of_the_episodes_that_did_not_error_with_the_errored_shown_apart() -> None:
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 9, 20, errored=2)))
    assert (claim["figure"], claim["counts"], claim["errored"]) == ("50%", "9 of 18", "2 errored")
    assert claim["spoken"] == "50%, 9 of 18, before\u00a0fine-tuning, 2 errored"
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 9, 20, errored=2), evaluation("ckpt-r1", 1, 15, 20)))
    assert (claim["figure"], claim["counts"], claim["errored"]) == ("+25 pts", "9/18 → 15/20", "2 → 0 errored")


def test_an_evaluation_whose_episodes_all_errored_says_so_in_words() -> None:
    claim, _ = run(CLAIM, stream(evaluation("ckpt-r0", 0, 0, 20), evaluation("ckpt-r1", 1, 0, 20, errored=20)))
    assert (claim["figure"], claim["counts"]) == (None, "0/20 → none")
    assert claim["spoken"] == (
        "0%, 0 of 20, before\u00a0fine-tuning, then all 20 held-out episodes errored after\u00a0round\u00a01"
    )


FROM = """
const s = S.initial();
for (const e of events) S.reduce(s, e);
const c = S.claim(s);
const checkpoints = S.results(s).checkpoints.map((x) => x.name);
return { names: c && c.names, spoken: c && c.spoken, checkpoints, carried: S.lessons(s).carried };
"""


def from_run(value: object) -> tuple[str, dict]:
    return "evaluation_result", {**evaluation("fixed-v2/ckpt-r1", 0, 4, 20)[1], "from_run": value}


def test_a_run_started_from_an_earlier_run_names_round_0_by_that_run_and_counts_its_carried_demonstrations() -> None:
    source = {"run": "fixed-v2", "checkpoint": "fixed-v2/ckpt-r1", "carried": 8}
    out = run(FROM, stream(*held_out(0, 1, "failed"), from_run(source), *evaluated(1, (1, "succeeded"))))
    assert out["names"] == "from fixed-v2, ckpt-r1 → after\u00a0round\u00a01"
    assert out["spoken"].endswith("20%, 4 of 20, from fixed-v2, ckpt-r1, then 100%, 1 of 1, after\u00a0round\u00a01")
    assert out["checkpoints"] == ["from fixed-v2, ckpt-r1", "after\u00a0round\u00a01"]
    assert out["carried"] == {"count": 8, "run": "fixed-v2"}
    # until its result names round 0's source, a round 0 on a checkpoint other than the base is only under way
    started_from = [
        ("episode_started", held_out(0, 1, None)[0][1]),
        ("decision", {**decision(0)[1], "checkpoint": "x"}),
    ]
    assert run(FROM, stream(*started_from))["checkpoints"] == ["the\u00a0evaluation\u00a0under\u00a0way"]
    # and so is one before its first decision, which could be on another checkpoint
    assert run(FROM, stream(*started_from[:1]))["checkpoints"] == ["the\u00a0evaluation\u00a0under\u00a0way"]
    assert run(FROM, stream(*held_out(0, 1, None)))["checkpoints"] == ["before\u00a0fine-tuning"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"run": "fixed-v2", "checkpoint": "ckpt-r1"},
        {"run": 5, "checkpoint": "ckpt-r1", "carried": 8},
        {"run": "", "checkpoint": "ckpt-r1", "carried": 8},
        {"run": "fixed-v2", "checkpoint": ["ckpt-r1"], "carried": 8},
        {"run": "fixed-v2", "checkpoint": "ckpt-r1", "carried": -1},
        {"run": "fixed-v2", "checkpoint": "ckpt-r1", "carried": 1.5},
        "fixed-v2",
    ],
)
def test_a_run_without_a_well_formed_from_run_reads_round_0_as_before_fine_tuning(value: object) -> None:
    out = run(FROM, stream(from_run(value)))
    assert out["names"] == out["checkpoints"][0] == "before\u00a0fine-tuning"
    assert out["carried"] is None


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
        *heldout("h3", "ckpt-r1", seed=2),
        started("take-is-dev"),
    )
    assert run(CLAIM, later)[1] == {"checkpoint": "ckpt-r1", "run": 2, "of": 20}
    done = stream(*heldout("h1", "ckpt-r0"), evaluation("ckpt-r0", 0, 0, 20), started("trial"))
    assert run(CLAIM, done)[1] is None
    # a seed counts once it has run, whether it reached a verdict or crashed before or during its episode
    first = evaluated(0, (1, "failed"), (2, "failed"), (3, "failed"))
    crashing = [
        *first,
        started("trial"),
        *held_out(1, 1, "failed"),
        crashed(2, 1),
        *held_out(1, 3, None),
        crashed(3, 1),
    ]
    assert run(CLAIM, stream(*crashing))[1] == {"checkpoint": "r/ckpt-r1", "run": 3, "of": 3}


VERDICTS = {
    "succeeded": (True, "box at rest inside bin"),
    "failed": (False, "box past the bin"),
    "errored": (False, "recording error: joint limit"),
}


def held_out(round_: int, seed: object, outcome: str | None) -> list[tuple[str, dict]]:
    """`round_`'s held-out episode on `seed`: succeeded, failed, a recording error, or (None) still running."""
    start = ("episode_started", {**started(f"h{round_}-{seed}")[1], "seed": seed, "split": "heldout", "round": round_})
    test = decision(0, round_=round_)
    if outcome is None:
        return [start, test]
    task, reason = VERDICTS[outcome]
    return [start, test, ("validation", {"task_success": task, "physics_valid": True, "reason": reason})]


def crashed(seed: int, round_: int) -> tuple[str, dict]:
    """The loop's error for a held-out episode on `seed` that raised, before or after it started."""
    return "error", {"seed": seed, "split": "heldout", "round": round_, "error": "RuntimeError: reset failed"}


def evaluated(round_: int, *outcomes: tuple[object, str | None]) -> list[tuple[str, dict]]:
    """`round_`'s whole evaluation: each (seed, outcome)'s episode, then its result."""
    episodes = [e for seed, outcome in outcomes for e in held_out(round_, seed, outcome)]
    good = sum(outcome == "succeeded" for _, outcome in outcomes)
    bad = sum(outcome in {"errored", None} for _, outcome in outcomes)
    return [*episodes, evaluation(ckpt(round_), round_, good, len(outcomes), errored=bad)]


RESULTS = """
const s = S.initial();
for (const e of events) S.reduce(s, e);
const r = S.results(s);
return {
  checkpoints: r.checkpoints,
  rows: r.seeds.map((x) => [x.seed, x.mark, x.episodes.map((e) => e.outcome)]),
  ids: r.seeds.map((x) => x.episodes.map((e) => e.id)),
  claim: S.claim(s),
};
"""


def results(*events: tuple[str, dict]) -> dict:
    return run(RESULTS, stream(*events))


def test_each_seed_is_marked_by_how_its_held_out_episode_changed_from_the_first_to_the_latest_evaluation() -> None:
    first = evaluated(0, (1, "failed"), (2, "succeeded"), (3, "succeeded"), (4, "failed"))
    latest = evaluated(1, (1, "succeeded"), (2, "failed"), (3, "succeeded"), (4, "failed"))
    out = results(*first, started("trial"), *latest)
    assert out["rows"] == [
        [1, "up", ["failed", "succeeded"]],
        [2, "down", ["succeeded", "failed"]],
        [3, "pass", ["succeeded", "succeeded"]],
        [4, "fail", ["failed", "failed"]],
    ]
    assert out["ids"][1] == ["h0-2", "h1-2"]
    assert out["checkpoints"] == [
        {"round": 0, "checkpoint": "ckpt-r0", "name": "before\u00a0fine-tuning", "done": True},
        {"round": 1, "checkpoint": "r/ckpt-r1", "name": "after\u00a0round\u00a01", "done": True},
    ]


def test_a_seed_that_errored_in_the_first_or_the_latest_evaluation_is_marked_errored() -> None:
    crash = ("error", {"seed": 4, "split": "heldout", "round": 0, "error": "RuntimeError: reset failed"})
    first = [*held_out(0, 1, "errored"), *held_out(0, 2, "failed"), *held_out(0, 3, None), crash]
    out = results(
        *first,
        evaluation("ckpt-r0", 0, 0, 4, errored=3),
        *evaluated(1, (1, "succeeded"), (2, "errored"), (3, "succeeded"), (4, "succeeded")),
    )
    # a recording error, an episode with no verdict once its evaluation finished, and a crash before the episode started
    assert out["rows"] == [
        [1, "error", ["errored", "succeeded"]],
        [2, "error", ["failed", "errored"]],
        [3, "error", ["errored", "succeeded"]],
        [4, "error", ["errored", "succeeded"]],
    ]
    assert out["ids"][3] == [None, "h1-4"]


def test_with_one_evaluation_every_seed_is_pending_and_mid_evaluation_so_is_every_seed_not_yet_judged() -> None:
    first = evaluated(0, (1, "failed"), (2, "succeeded"), (3, "errored"))
    one = results(*first)
    assert one["rows"] == [[1, "pending", ["failed"]], [2, "pending", ["succeeded"]], [3, "error", ["errored"]]]
    mid = results(*first, started("trial"), *held_out(1, 1, "succeeded"), *held_out(1, 2, None))
    assert mid["rows"] == [
        [1, "up", ["failed", "succeeded"]],
        [2, "pending", ["succeeded", "pending"]],
        [3, "error", ["errored", "pending"]],
    ]
    assert mid["checkpoints"][1] == {"round": 1, "checkpoint": "r/ckpt-r1", "name": "after round 1", "done": False}
    running = results(*held_out(0, 1, "failed"), *held_out(0, 2, None))
    assert running["rows"] == [[1, "pending", ["failed"]], [2, "pending", ["pending"]]]
    # a seed whose episode crashed in the evaluation under way is errored before that evaluation's result
    crash = results(*first, started("trial"), *held_out(1, 1, "succeeded"), *held_out(1, 2, None), crashed(2, 1))
    assert [2, "error", ["succeeded", "errored"]] in crash["rows"]


def test_with_three_checkpoints_the_mark_compares_the_first_and_the_latest_and_lists_every_one() -> None:
    out = results(
        *evaluated(0, (1, "failed"), (2, "succeeded")),
        *evaluated(1, (1, "succeeded"), (2, "failed")),
        *evaluated(2, (1, "failed"), (2, "succeeded")),
    )
    assert out["rows"] == [
        [1, "fail", ["failed", "succeeded", "failed"]],
        [2, "pass", ["succeeded", "failed", "succeeded"]],
    ]
    assert [c["name"] for c in out["checkpoints"]] == [
        "before fine-tuning",
        "after round 1",
        "after round 2",
    ]


def test_seeds_come_in_seed_order_whatever_their_marks() -> None:
    first = evaluated(0, *[(seed, "failed") for seed in (9, 3, 5)], (7, "succeeded"), (8, "errored"), (2, "failed"))
    latest = [
        *held_out(1, 9, "succeeded"),
        *held_out(1, 3, "succeeded"),
        *held_out(1, 5, "failed"),
        *held_out(1, 7, "failed"),
    ]
    out = results(*first, *latest)
    marks = [[2, "pending"], [3, "up"], [5, "fail"], [7, "down"], [8, "error"], [9, "up"]]
    assert [row[:2] for row in out["rows"]] == marks


def test_a_seed_missing_from_a_finished_evaluation_errored_and_one_the_running_evaluation_lacks_is_pending() -> None:
    out = results(
        *evaluated(0, (1, "failed")),
        *evaluated(1, (1, "succeeded"), (2, "succeeded")),
        *held_out(2, 1, "failed"),
    )
    # seed 1's latest is the running evaluation's verdict
    assert out["rows"] == [
        [1, "fail", ["failed", "succeeded", "failed"]],
        [2, "error", ["errored", "succeeded", "pending"]],
    ]
    assert out["ids"][1] == [None, "h1-2", None]


def test_hostile_seed_and_checkpoint_values_never_make_a_row_or_break_the_model() -> None:
    hostile = [held_out(0, seed, "failed") for seed in ("7", None, "<img src=x onerror=alert(1)>", [1], {"n": 1})]
    twice = [
        *held_out(0, 1, "failed"),
        ("episode_started", {**held_out(0, 1, None)[0][1], "episode_id": "again"}),
        verdict(True),
    ]
    bad_round = ("evaluation_result", {"checkpoint": {"<b>": 1}, "round": 0, "successes": 0, "n": 1, "errored": 0})
    out = results(*[e for h in hostile for e in h], *twice, bad_round)
    # the first try of a seed in an evaluation is its episode; a checkpoint that is not text reads as none
    assert out["rows"] == [[1, "pending", ["failed"]]]
    assert out["checkpoints"] == [{"round": 0, "checkpoint": "", "name": "before fine-tuning", "done": True}]
    assert results()["rows"] == [] and results()["checkpoints"] == []


TWO_EVALUATIONS = [
    *evaluated(0, (1, "failed"), (2, "succeeded"), (3, "failed")),
    *trial("t0", diagnosed=True),
    *evaluated(1, (1, "succeeded"), (2, "failed"), (3, "failed")),
]


def test_a_seed_opens_over_the_run_arrows_step_its_neighbours_and_escape_returns_to_where_the_view_was() -> None:
    out = presenter(
        """
        open();
        press("ArrowLeft");
        const was = look().episode;
        S.visit(v, { seed: 1 });
        const seen = [[S.where(v), v.seed, look().episode]];
        for (const key of ["ArrowLeft", "ArrowRight", "ArrowRight", "ArrowRight"]) {
          press(key);
          seen.push([S.where(v), v.seed]);
        }
        S.visit(v, { seed: 1, opened: "h0-1" });
        const episodes = [look().episode];
        for (const key of ["ArrowRight", "ArrowRight", "ArrowLeft"]) {
          press(key);
          episodes.push(look().episode);
        }
        press("Escape");
        const seed = [S.where(v), v.seed];
        press("Escape");
        return { was, seen, episodes, seed, back: [S.where(v), look().episode], claims: press("Escape") };
        """,
        stream(*TWO_EVALUATIONS),
    )
    assert out["was"] == "h0-3"
    assert out["seen"] == [["seed", 1, "h0-3"], ["seed", 1], ["seed", 2], ["seed", 3], ["seed", 3]]
    assert out["episodes"] == ["h0-1", "h1-1", "h1-1", "h0-1"]
    assert out["seed"] == ["seed", 1]
    assert out["back"] == ["run", "h0-3"]
    assert out["claims"] is False


def test_escape_from_a_seed_opened_over_move_returns_to_move_and_m_still_turns_it_off() -> None:
    events = stream(*evaluated(0, (1, "failed")), *trial("t0", moved=True), *evaluated(1, (1, "succeeded")))
    out = presenter(
        """
        open();
        S.visit(v, { selected: "t0" });
        press("m");
        const at = () => [S.where(v), look().episode, look().focus];
        const moved = at();
        S.visit(v, S.showSeed(v, 1));
        const seed = at();
        press("Escape");
        const back = at();
        press("m");
        return { moved, seed, back, off: at() };
        """,
        events,
    )
    assert out["moved"] == ["run", "t0", 1]
    assert out["seed"] == ["seed", "t0", None]
    assert out["back"] == ["run", "t0", 1]
    assert out["off"] == ["run", "t0", None]


def test_the_run_walks_every_episode_in_order_held_out_tests_included_and_move_opens_one_in_place() -> None:
    moved = [*held_out(1, 1, "failed")]
    moved.insert(2, ("perturbation", {"kind": "move_box", "step": 0}))
    events = stream(*evaluated(0, (1, "failed")), *trial("t0", diagnosed=True), *trial("t1"), *moved)
    out = presenter(
        """
        open();
        const seen = [look().episode];
        for (const key of ["ArrowLeft", "ArrowLeft", "ArrowLeft", "ArrowRight"]) {
          press(key);
          seen.push(look().episode);
        }
        press("m");
        const moved = [S.where(v), look().episode, S.stageStep(v)];
        press("m");
        return { seen, moved, back: look().episode };
        """,
        events,
    )
    assert out["seen"] == ["t1", "t0", "h0-1", "h0-1", "t0"]
    assert out["moved"] == ["run", "h1-1", 0]
    assert out["back"] == "t0"


def test_the_strips_switch_shows_a_seeds_first_or_latest_episode_in_full_or_both_and_holds_from_seed_to_seed() -> None:
    out = presenter(
        """
        open();
        const at = () => [S.where(v), v.seed, look().episode];
        S.choose(v, "before");
        const idle = at();
        S.visit(v, S.showSeed(v, 1));
        const seen = [at()];
        press("ArrowRight"); seen.push(at());
        S.choose(v, "after"); seen.push(at());
        press("ArrowRight"); seen.push(at());
        S.choose(v, "both"); seen.push(at());
        press("ArrowLeft"); seen.push(at());
        S.choose(v, "after");
        press("Escape");
        return { idle, seen, back: at() };
        """,
        stream(*TWO_EVALUATIONS),
    )
    assert out["idle"] == ["run", None, "t0"]
    assert out["seen"] == [
        ["episode", 1, "h0-1"],
        ["episode", 2, "h0-2"],
        ["episode", 2, "h1-2"],
        ["episode", 3, "h1-3"],
        ["seed", 3, "t0"],
        ["seed", 2, "t0"],
    ]
    assert out["back"] == ["run", None, "t0"]


def test_the_switch_is_dimmed_until_a_seed_is_chosen() -> None:
    script = """
    const buttons = ["before", "both", "after"].map((side) => {
      const b = { dataset: { side }, attrs: {}, setAttribute(k, x) { this.attrs[k] = x; } };
      return b;
    });
    doc.getElementById("sides").querySelectorAll = () => buttons;
    const look = () => buttons.map((b) => [b.attrs["aria-disabled"], b.attrs["aria-pressed"], b.dataset.tip]);
    vm.runInContext("render()", context);
    const idle = look();
    vm.runInContext("MomState.visit(view, MomState.showSeed(view, 1)); render();", context);
    return { idle, open: look() };
    """
    out = app(REPLAY + script, stream(*TWO_EVALUATIONS))
    # the harness's stage is narrow, so the switch defaults to After
    assert out["idle"] == [
        ["true", "false", "Choose a seed to compare"],
        ["true", "false", "Choose a seed to compare"],
        ["true", "true", "Choose a seed to compare"],
    ]
    assert [x[:2] for x in out["open"]] == [["false", "false"], ["false", "false"], ["false", "true"]]
    assert out["open"][2][2] == "After: the seed's latest checkpoint"


def test_the_switch_keeps_its_side_while_a_seed_is_open_when_another_checkpoint_starts() -> None:
    third = held_out(2, 1, None)
    script = f"""
    // wide enough for two checkpoints side by side, not three
    doc.querySelector(".camera").clientWidth = 620;
    const later = events.splice(events.length - {len(third)});
    {REPLAY}
    const at = () => vm.runInContext("[view.side, MomState.where(view)]", context);
    const idle = at();
    vm.runInContext("MomState.visit(view, MomState.showSeed(view, 1)); render();", context);
    const open = at();
    for (const e of later) sources[0].onmessage({{ data: JSON.stringify(e) }});
    flush();
    return {{ idle, open, later: at() }};
    """
    out = app(script, stream(*TWO_EVALUATIONS, *third))
    assert out == {"idle": ["both", "run"], "open": ["both", "seed"], "later": ["both", "seed"]}


def test_the_step_keys_tooltips_name_what_they_step_through() -> None:
    script = """
    const tip = (fields) => {
      vm.runInContext(`MomState.visit(view, ${JSON.stringify(fields)}); render();`, context);
      return [doc.getElementById("b-prev").dataset.tip, doc.getElementById("b-next").dataset.tip];
    };
    const seed = tip({ seed: 1, opened: "h1-1" });
    vm.runInContext(`chosen = true; MomState.choose(view, "both");`, context);
    return { seed, checkpoint: tip({ seed: 1, opened: "h0-1" }), episode: tip({ seed: null, opened: null }) };
    """
    out = app(REPLAY + script, stream(*TWO_EVALUATIONS))
    # the harness's narrow stage defaults the switch to After, so the arrows step from seed to seed
    assert out == {
        "seed": ["Previous seed · ←", "Next seed · →"],
        "checkpoint": ["Previous checkpoint · ←", "Next checkpoint · →"],
        "episode": ["Previous episode · ←", "Next episode · →"],
    }


def test_each_back_button_is_named_by_where_it_goes_and_the_switch_describes_itself() -> None:
    script = """
    const last = (id) => writes.filter(([at]) => at === id).pop()[1];
    const back = (html) => html.match(/class="back"[^>]*aria-label="([^"]*)"/)[1];
    const buttons = ["before", "both", "after"].map((side) => ({
      dataset: { side }, attrs: {}, setAttribute(k, x) { this.attrs[k] = x; },
    }));
    doc.getElementById("sides").querySelectorAll = () => buttons;
    const said = () => buttons.map((b) => b.attrs["aria-description"]);
    vm.runInContext("render()", context);
    const idle = said();
    vm.runInContext("MomState.visit(view, MomState.showSeed(view, 1)); render();", context);
    const [shown, open] = [back(last("control")), said()];
    vm.runInContext(`chosen = true; MomState.choose(view, "both"); render();`, context);
    const seed = back(last("results"));
    vm.runInContext(`MomState.visit(view, { opened: "h0-1" }); render();`, context);
    return { idle, open, shown, seed, card: back(last("control")) };
    """
    out = app(REPLAY + script, stream(*TWO_EVALUATIONS))
    assert out["idle"] == ["Choose a seed to compare"] * 3
    assert out["open"] == [
        "Before: the seed's first checkpoint",
        "Both: every checkpoint side by side",
        "After: the seed's latest checkpoint",
    ]
    # the harness's narrow stage defaults the switch to After, whose episode goes back to the run
    assert (out["shown"], out["seed"], out["card"]) == ("Back to the run", "Back to the run", "Back to seed 1")


def conditioned(round_: int, condition: str, *outcomes: tuple[object, str | None]) -> list[tuple[str, dict]]:
    *episodes, (kind, result) = evaluated(round_, *outcomes)
    return [*episodes, (kind, {**result, "condition": condition})]


def test_an_eval_run_compares_the_first_and_last_checkpoint_on_the_training_task_alone() -> None:
    seeds = range(1, 21)
    blocks = [
        conditioned(0, "training_distribution", *[(s, "succeeded" if s <= 5 else "failed") for s in seeds]),
        conditioned(0, "new_layout", *[(s + 100, "succeeded") for s in seeds]),
        conditioned(0, "move_bin_after_lift", *[(s, "errored") for s in seeds]),
        conditioned(1, "training_distribution", *[(s, "succeeded" if s <= 15 else "failed") for s in seeds]),
        conditioned(1, "new_layout", *[(s + 100, "failed") for s in seeds]),
        conditioned(1, "move_bin_after_lift", *[(s, "succeeded") for s in seeds]),
    ]
    out = results(*[e for block in blocks for e in block])
    assert [c["name"] for c in out["checkpoints"]] == ["ckpt-r0", "ckpt-r1"]
    assert [row[0] for row in out["rows"]] == list(seeds)
    assert out["rows"][2] == [3, "pass", ["succeeded", "succeeded"]]
    assert out["rows"][9] == [10, "up", ["failed", "succeeded"]]
    assert out["rows"][19] == [20, "fail", ["failed", "failed"]]
    assert (out["claim"]["figure"], out["claim"]["counts"], out["claim"]["errored"]) == (
        "+50 pts",
        "5 → 15 of 20",
        None,
    )
    # a loop run's results carry no condition, so every one of them counts
    loop = results(*TWO_EVALUATIONS)
    assert len(loop["checkpoints"]) == 2 and loop["claim"]["counts"] == "1 → 1 of 3"


def test_an_eval_runs_checkpoints_are_named_by_their_ids_and_a_loop_runs_by_round() -> None:
    # `mom eval --ckpt r/ckpt-r3` tests that checkpoint second, so its round is 1, which is no fine-tuning round
    base = conditioned(0, "training_distribution", (1, "failed"))
    *tuned, (kind, result) = conditioned(1, "training_distribution", (1, "succeeded"))
    events = [*base, *tuned, (kind, {**result, "checkpoint": "r/ckpt-r3"})]
    out = results(*events)
    assert [c["name"] for c in out["checkpoints"]] == ["ckpt-r0", "ckpt-r3"]
    assert (out["claim"]["names"], out["claim"]["ids"]) == ("ckpt-r0 → ckpt-r3", "ckpt-r0 → r/ckpt-r3")
    # a seed's cards head each checkpoint by its id as written, never capitalised
    script = """
    vm.runInContext(`
      chosen = true; MomState.choose(view, "both"); MomState.visit(view, { seed: 1 }); render();
    `, context);
    return writes.filter(([id]) => id === "results").pop()[1].match(/<span class="card-head">.*?<\\/span><\\/span>/g);
    """
    assert app(REPLAY + script, stream(*events)) == [
        '<span class="card-head"><span class="what">ckpt-r0</span></span>',
        '<span class="card-head"><span class="what">ckpt-r3</span></span>',
    ]
    loop = results(*TWO_EVALUATIONS)
    assert [c["name"] for c in loop["checkpoints"]] == ["before fine-tuning", "after round 1"]
    assert loop["claim"]["names"] == "before fine-tuning → after round 1"


def test_a_live_eval_run_shows_no_block_under_way_once_a_closed_result_names_its_condition() -> None:
    # the base's training block closes with seed 1 failed, then new_layout runs seed 1 on the base and it succeeds
    layout = held_out(0, 1, "succeeded")
    layout[0][1]["episode_id"] = "layout-1"
    live = [*conditioned(0, "training_distribution", (1, "failed"), (2, "succeeded")), *layout]
    out = results(*live)
    assert len(out["checkpoints"]) == 1
    assert out["rows"] == [[1, "pending", ["failed"]], [2, "pending", ["succeeded"]]]
    assert run(CLAIM, stream(*live))[1] is None
    script = """
    const columns = writes.filter(([id]) => id === "columns").pop()[1];
    vm.runInContext(`
      chosen = true; MomState.choose(view, "both"); MomState.visit(view, { seed: 1 }); render();
    `, context);
    const seed = writes.filter(([id]) => id === "results").pop()[1];
    return { up: columns.includes("#m-up"), cards: seed.match(/class="card"/g).length };
    """
    assert app(REPLAY + script, stream(*live)) == {"up": False, "cards": 1}


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
        "carried": None,
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


def argon_call(call: int, ok: bool = True, latency: object = 1000.0) -> tuple[str, dict]:
    error = None if ok else "unreachable"
    data = {"purpose": "diagnose", "model": "m", "call": call, "budget": 800, "ok": ok, "error": error}
    return "argon_call", {**data, "latency_ms": latency}


def test_the_argon_strip_counts_calls_used_of_the_budget_the_median_answer_and_the_failures() -> None:
    script = "const s = S.initial(); for (const e of events) S.reduce(s, e); return S.argon(s);"
    # The task generator's call comes first; the run's own calls count on from it (SPEC §3.5, rule 5).
    calls = [argon_call(1, latency=1200.0), argon_call(2, ok=False, latency=30000.0), argon_call(3, latency=800.0)]
    out = run(script, stream(*calls, argon_call(4, latency="slow"), argon_call(5)))
    assert out == {"used": 5, "budget": 800, "answer": {"median": 1000.0, "n": 3}, "failed": 1}
    assert run(script, stream(started("a"))) is None


def test_each_fine_tune_names_the_round_whose_training_set_summary_the_page_reads() -> None:
    out = fold(
        ("finetune_started", {"round": 1, "training_set": "runs/r/training_r1.jsonl"}),
        ("finetune_completed", {"round": 1}),
        ("finetune_started", {"round": 2}),
        ("finetune_started", {"round": 2}),
        ("finetune_started", {"round": "3"}),
    )
    assert out["state"]["trained"] == [1, 2]


def test_saved_run_files_are_read_only_when_they_have_the_contracts_shape() -> None:
    summary = {
        "round": 1,
        "episodes_scanned": 30,
        "demonstrations": 6,
        "accepted": 5,
        "rejected": 1,
        "unfinished": 0,
        "rejected_by": {"past_bin": 1},
        "empty_dev_episodes": 0,
        "examples": 43,
    }
    report = {"replayed": 5, "reproduced": 4, "failures": [{"episode_id": "e1", "reason": "box 3 cm off"}]}
    files = [
        summary,
        {**summary, "accepted": "5"},
        {**summary, "rejected_by": {"past_bin": -1}},
        None,
        report,
        {**report, "failures": "none"},
        {**report, "failures": [{"episode_id": 7}]},
        [],
    ]
    script = """
    const [ok, text, negative, missing, report, words, badFailure, list] = events;
    const summaries = [ok, text, negative, missing].map((f) => S.trainingSummary(f));
    return [summaries, [report, words, badFailure, list].map((f) => S.replayReport(f))];
    """
    summaries, reports = run(script, files)
    assert summaries == [summary, None, None, None]
    assert reports == [report, None, None, None]


def test_the_runs_list_is_read_only_in_the_contracts_shape_and_keeps_the_servers_order() -> None:
    live = {
        "run_id": "teach-1",
        "instruction": "Put the box in the bin",
        "status": "live",
        "reason": None,
        "started_t": 2.0,
    }
    refused = {
        "run_id": "teach-0",
        "instruction": "Juggle",
        "status": "not_started",
        "reason": "No such skill",
        "started_t": 1.0,
    }
    old = {"run_id": "shot", "instruction": None, "status": "recorded", "reason": None, "started_t": 0.0}
    bad = [
        {**old, "run_id": 7},
        {**old, "status": "paused"},
        {**old, "reason": 3},
        {**old, "started_t": "yesterday"},
        {**old, "instruction": ["x"]},
        None,
    ]
    script = "return [S.runsOf(events), S.runsOf({ runs: events }), S.runsOf(null)];"
    assert run(script, [live, *bad, refused, old]) == [[live, refused, old], [], []]


def test_a_page_whose_url_names_no_run_opens_the_first_listed() -> None:
    runs = [{"run_id": "live"}, {"run_id": "older"}]
    assert run("const [runs] = events; return [S.runToOpen(runs), S.runToOpen([])];", [runs]) == ["live", None]


def test_an_instruction_is_1_to_200_characters_once_trimmed_and_the_count_shows_near_the_limit() -> None:
    texts = ["   ", "x", "y" * 200, "z" * 201, "  " + "w" * 200 + "\n", "a" * 159, "a" * 160]
    script = "return events.map((t) => [S.instructionProblem(t), S.teachCount(t), S.tooLong(t)]);"
    out = run(script, texts)
    empty, fits = "Type what the robot should learn.", None
    too_long = "At most 200 characters."
    assert [problem for problem, _, _ in out] == [empty, fits, fits, too_long, fits, fits, fits]
    assert [shown for _, shown, _ in out] == [None, None, "200/200", "201/200", "200/200", None, "160/200"]
    assert [over for _, _, over in out] == [False, False, False, True, False, False, False]


@pytest.mark.parametrize(("n", "problem"), [(199, None), (200, None), (201, "At most 200 characters.")])
def test_an_emoji_counts_as_one_character_as_the_server_counts_it(n: int, problem: str | None) -> None:
    script = "return events.map((t) => [S.instructionProblem(t), S.teachCount(t), S.tooLong(t)]);"
    assert run(script, ["\N{ROBOT FACE}" * n]) == [[problem, f"{n}/200", n > 200]]


def test_a_teach_answer_is_started_not_started_with_its_reasons_or_a_failure_in_words() -> None:
    answers = [
        [201, {"run_id": "teach-1", "started": True}],
        [200, {"run_id": "teach-2", "started": False, "reasons": ["Not yet: the loop teaches one task"]}],
        [200, {"run_id": "teach-3", "started": False, "reasons": ['<img src=x onerror="alert(1)">']}],
        [200, {"run_id": "teach-4", "started": False, "reasons": "unreachable"}],
        [403, {"detail": "Forbidden"}],
        [400, {"detail": "1 to 200 characters"}],
        [500, None],
        [0, None],
    ]
    out = run("return events.map(([status, body]) => S.teachAnswer(status, body));", answers)
    assert out[0] == {"state": "started", "runId": "teach-1"}
    assert out[1] == {"state": "refused", "runId": "teach-2", "reasons": ["Not yet: the loop teaches one task"]}
    # Typed and generated text stays text: the page escapes it where it is shown.
    assert out[2]["reasons"] == ['<img src=x onerror="alert(1)">']
    assert [o["state"] for o in out[3:]] == ["failed"] * 5
    assert out[4]["message"] == "The dashboard takes instructions only from its own page on this computer."
    assert out[5]["message"] == "An instruction is 1 to 200 characters."
    assert out[6]["message"] == "The dashboard server could not take it (HTTP 500)."
    assert out[7]["message"] == "Could not reach the dashboard server."


def test_a_runs_instruction_and_why_it_did_not_start_are_read_from_its_events() -> None:
    refused = fold(
        ("instruction", {"text": "Juggle three boxes", "source": "teach"}),
        argon_call(1),
        ("instruction", {"text": "a second one is not the run's", "source": "cli"}),
        ("error", {"error": "not started: No such skill; Not yet: the loop teaches one task"}),
    )["state"]
    assert refused["instruction"] == {"text": "Juggle three boxes", "source": "teach"}
    assert refused["refused"] == "No such skill; Not yet: the loop teaches one task"
    taught = fold(("instruction", {"text": "Put the box in the bin", "source": "cli"}), started("a"))["state"]
    assert taught["refused"] is None and taught["instruction"]["source"] == "cli"
    assert fold(started("a"))["state"]["instruction"] is None


APP_JS = STATE_JS.parent / "app.js"
# app.js under a DOM stub that records every write; the test drives the event stream and the animation frames.
APP_DRIVER = r"""
const vm = require("vm");
const fs = require("fs");
const [statePath, appPath] = process.argv.slice(1);
const { events, runs, script } = JSON.parse(fs.readFileSync(0, "utf8"));
const writes = [];
const elements = new Map();
class FakeElement {
  constructor(id = "", tagName = "DIV") {
    Object.assign(this, { id, tagName, dataset: {}, hidden: false, disabled: false, textContent: "", value: "" });
    Object.assign(this, { children: [], scrollTop: 0, scrollHeight: 0, clientHeight: 0, isConnected: true });
    Object.assign(this, { open: false, listeners: {} });
    this.style = { setProperty() {} };
    this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
    this.lastElementChild = { textContent: "" };
  }
  get innerText() { return ""; }
  setAttribute() {}
  removeAttribute() {}
  toggleAttribute() {}
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  querySelector(selector) { return Object.assign(new FakeElement(), { selector }); }
  querySelectorAll() { return []; }
  getBoundingClientRect() { return { top: 0 }; }
  checkVisibility() { return true; }
  contains() { return false; }
  focus() { doc.activeElement = this; }
}
Object.defineProperty(FakeElement.prototype, "innerHTML", {
  get() { return this.html ?? ""; },
  set(html) { this.html = html; writes.push([this.id, html]); },
  configurable: true,
});
const IDS = ["status", "run-meta", "announce", "control", "stage", "strip", "log", "chip-log", "p-rounds",
  "p-diagnosis", "p-training", "models", "side", "results", "main", "columns", "sides", "runs", "run-name", "run-list",
  "teach", "instruction", "teach-count", "teach-go", "teach-said", "teach-status", "b-prev", "b-next", "b-end",
  "b-d", "b-t", "b-m", "b-pause"];
for (const id of IDS) elements.set(id, new FakeElement(id, id === "instruction" ? "INPUT" : "DIV"));
const body = new FakeElement("", "BODY");
const doc = {
  body,
  activeElement: body,
  listeners: {},
  getElementById: (id) => elements.get(id) || null,
  // The run picker stands for the page's popovers.
  querySelectorAll: (selector) =>
    selector === "[id]" ? [...elements.values()] : selector.includes("details.runs") ? [elements.get("runs")] : [],
  // One element per selector, so a listener app.js adds to it can be driven.
  found: new Map(),
  querySelector(selector) {
    if (!this.found.has(selector)) this.found.set(selector, Object.assign(new FakeElement(), { selector }));
    return this.found.get(selector);
  },
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
  getAnimations: () => [],
};
const sources = [];
class EventSource {
  constructor(url) { this.url = url; this.listeners = {}; sources.push(this); }
  addEventListener(type, fn) { this.listeners[type] = fn; }
}
const frames = [];
const fetched = [];
const intervals = [];
// A body-less answer's json() throws, as a 204's does.
const answer = (status, body) => ({
  ok: status < 300, status, headers: new Map(), json: async () => body ?? JSON.parse(""),
});
const refused = { run_id: "t", started: false, reasons: ["Not yet"] };
const context = vm.createContext({
  document: doc, Element: FakeElement, EventSource, console, structuredClone, URLSearchParams,
  location: { search: "?run=r", replace() {}, assign() {} },
  history: { replaceState() {} },
  fetch: async (url) => {
    fetched.push(url);
    return url === "/runs" ? answer(200, runs) : url === "/teach" ? answer(200, refused) : answer(204);
  },
  requestAnimationFrame: (fn) => frames.push(fn),
  setInterval: (fn, ms) => intervals.push([fn.name, ms]),
  addEventListener() {},
  innerWidth: 1280,
  CSS: { escape: (s) => s },
});
vm.runInContext(fs.readFileSync(statePath, "utf8"), context);
vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
const flush = () => { while (frames.length) frames.shift()(); };
const settled = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  for (let i = 0; i < 5; i++) await settled();
  const AsyncFunction = (async () => {}).constructor;
  const names = ["vm", "context", "sources", "writes", "frames", "flush", "doc", "events", "fetched", "intervals"];
  const run = new AsyncFunction(...names, script);
  const out = await run(vm, context, sources, writes, frames, flush, doc, events, fetched, intervals);
  console.log(JSON.stringify(out));
})();
"""

HOSTILE = "<img src=x onerror=alert(1)>"


def app(script: str, events: list[dict], runs: list[dict] | None = None) -> Any:
    """Run `script` (an async JS function body) against app.js loaded under the DOM stub, and return its result."""
    listed = (
        runs
        if runs is not None
        else [{"run_id": "r", "instruction": None, "status": "recorded", "reason": None, "started_t": 0}]
    )
    stdin = json.dumps({"events": events, "runs": listed, "script": script})
    command = ["node", "-e", APP_DRIVER, str(STATE_JS), str(APP_JS)]
    out = subprocess.run(command, input=stdin, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def hostile_run() -> list[dict]:
    """A run whose every text an event carries is markup: instruction, ids, reasons, checkpoints, frames, errors."""
    x = HOSTILE
    say = {"hypothesis": x, "evidence": x, "lesson": x, "confidence": 0.5}
    start = {"seed": 7, "split": "heldout", "sim_version": f"mujoco {x}", "robot_config": x}
    events = [
        ("instruction", {"text": x, "source": "teach"}),
        ("episode_started", {**start, "episode_id": f"h0{x}", "round": 0}),
        (
            "decision",
            {"controller": "student", "checkpoint": x, "step": 0, "skill": {"skill": x}, "reason": x, "latency_ms": 5},
        ),
        ("action_executed", {"step": 0, "ok": False, "error": x, "frame": x}),
        ("validation", {"task_success": False, "physics_valid": True, "reason": x}),
        (
            "evaluation_result",
            {
                "checkpoint": x,
                "round": 0,
                "successes": 0,
                "n": 1,
                "errored": 0,
                "from_run": {"run": x, "checkpoint": x, "carried": 2},
            },
        ),
        ("episode_started", {**start, "split": "dev", "seed": 1000, "episode_id": f"t{x}", "round": 1}),
        ("decision", {"controller": "student", "checkpoint": x, "step": 0, "skill": {"skill": x}, "reason": x}),
        ("perturbation", {"kind": x, "step": 0, "before_frame": x}),
        ("validation", {"task_success": False, "physics_valid": False, "reason": x}),
        ("diagnosis", {"episode_id": f"t{x}", **say}),
        ("episode_started", {**start, "split": "dev", "seed": 1000, "episode_id": f"d{x}", "acting_on": f"t{x}"}),
        ("decision", {"controller": "teacher", "checkpoint": None, "step": 0, "skill": {"skill": x}, "reason": x}),
        ("error", {"error": x}),
        ("validation", {"task_success": True, "physics_valid": True, "reason": x}),
        ("episode_started", {**start, "episode_id": f"h1{x}", "round": 1}),
        ("decision", {"controller": "student", "checkpoint": x, "step": 0, "skill": {"skill": x}, "reason": x}),
        ("validation", {"task_success": True, "physics_valid": True, "reason": x}),
        ("evaluation_result", {"checkpoint": x, "round": 1, "successes": 1, "n": 1, "errored": 0}),
    ]
    return [{**e, "run_id": f"r{x}"} for e in stream(*events)]


REPLAY = """
const [source] = sources;
source.onopen();
for (const e of events) source.onmessage({ data: JSON.stringify(e) });
source.listeners.replayed();
flush();
"""


# A tag's quoted values are escaped text, so only what lies outside them could be injected markup.
MARKUP = """
const tags = writes.flatMap(([, html]) => html.replace(/"[^"]*"/g, '""').match(/<[^>]+>/g) || []);
return {
  injected: tags.filter((t) => /onerror|^<img src=x/i.test(t)),
  escaped: writes.filter(([, html]) => html.includes("&lt;img src=x onerror=alert(1)&gt;")).map(([id]) => id),
};
"""


def test_no_value_from_an_event_or_the_runs_list_becomes_markup_in_any_view() -> None:
    script = """
    vm.runInContext(`
      MomState.visit(view, { seed: 7 }); render();
      MomState.visit(view, { opened: MomState.results(MomState.shown(view)).seeds[0].episodes[0].id }); render();
      MomState.visit(view, { seed: null, opened: null, selected: null }); render();
      MomState.keydown(view, { key: "m" }, Date.now() / 1000); render();
      MomState.keydown(view, { key: "d" }, Date.now() / 1000); render();
      const reasons = [${JSON.stringify(events[0].data.text)}];
      teaching = MomState.teachAnswer(200, { run_id: "r", started: false, reasons });
      renderTeach();
    `, context);
    """
    runs = [{"run_id": "r", "instruction": HOSTILE, "status": "not_started", "reason": HOSTILE, "started_t": 0}]
    out = app(REPLAY + script + MARKUP, hostile_run(), runs)
    assert out["injected"] == []
    # the markup reached every panel as text
    assert {"run-list", "results", "control", "stage", "log", "p-rounds", "p-diagnosis", "teach-said"} <= set(
        out["escaped"]
    )
    # A run that did not start shows why on the stage, and Argon's answer can steer that text.
    events = stream(
        ("instruction", {"text": HOSTILE, "source": "teach"}), ("error", {"error": f"not started: {HOSTILE}"})
    )
    out = app(REPLAY + MARKUP, [{**e, "run_id": f"r{HOSTILE}"} for e in events])
    assert out["injected"] == []
    assert "stage" in out["escaped"]


def test_nothing_renders_or_announces_until_the_replay_of_the_written_events_settles() -> None:
    script = """
    const [source] = sources;
    const before = writes.map(([id]) => id);
    source.onopen();
    for (const e of events) source.onmessage({ data: JSON.stringify(e) });
    const replaying = { frames: frames.length, written: writes.map(([id]) => id).slice(before.length) };
    const announced = doc.getElementById("announce").textContent;
    source.listeners.replayed();
    flush();
    const heard = doc.getElementById("announce").textContent;
    return { before, replaying, announced, after: new Set(writes.map(([id]) => id)).size, heard };
    """
    events = stream(started("a"), decision(0), executed(0), verdict(False), started("b"), decision(0))
    out = app(script, events)
    # only the runs list, which is not the event stream's, is written before the replay settles
    assert set(out["before"]) <= {"run-list"}
    assert out["replaying"] == {"frames": 0, "written": []}
    assert out["announced"] == ""
    assert out["after"] > 5 and out["heard"]


def test_after_a_move_the_stage_marks_it_on_every_later_frame_and_only_icons_carry_the_results_mark_class() -> None:
    script = """
    // `.mark` sizes the strip's icons, so anything else in that class is squeezed into an icon's box.
    const marked = writes.flatMap(([, html]) => html.match(/<\\w+[^>]*class="[^"]*\\bmark\\b[^"]*"/g) || []);
    const stage = writes.filter(([id]) => id === "stage").pop()[1];
    return { stage, notIcons: marked.filter((t) => !t.startsWith("<svg")) };
    """
    out = app(REPLAY + script, stream(*TAKE))
    assert "Box moved at step 1" in out["stage"]
    assert out["notIcons"] == []


STYLE_CSS = STATE_JS.parent / "style.css"


def test_no_rule_redeclares_a_root_token_so_the_strips_hover_and_open_seed_take_their_tint() -> None:
    rules = re.findall(r"([^{}]*)\{([^{}]*)\}", STYLE_CSS.read_text())
    declared = [(selector.strip(), set(re.findall(r"(--[\w-]+)\s*:", body))) for selector, body in rules]
    tokens = set().union(*(names for selector, names in declared if ":root" in selector))
    shadowed = {selector: names & tokens for selector, names in declared if ":root" not in selector and names & tokens}
    assert shadowed == {}


STRIP = r"""
const html = writes.filter(([id]) => id === "columns").pop()[1];
const button =
  /<button[^>]*data-column="(\d+)"[^>]*tabindex="(-?\d)"[^>]*aria-label="([^"]*)" title="([^"]*)"[^>]*>(.*?)<\/b/g;
const icon = /<svg class="i ([^"]*)"[^>]*><use href="#m-(\w+)"/g;
return [...html.matchAll(button)].map((m) => ({
  seed: Number(m[1]), tab: m[2], name: m[3], tip: m[4],
  cells: [...m[5].matchAll(icon)].map((x) => `${x[1].split(" ")[0]} ${x[2]}`),
}));
"""


def strip(events: list[dict]) -> list[dict]:
    return app(REPLAY + STRIP, events)


def test_the_strip_has_a_column_per_seed_in_seed_order_its_coloured_mark_over_its_verdict_before_and_after() -> None:
    columns = strip(stream(*TWO_EVALUATIONS))
    assert [(c["seed"], c["name"], c["cells"]) for c in columns] == [
        (1, "Seed 1: improved, failed before, succeeded after", ["mark up", "plain fail", "plain pass"]),
        (2, "Seed 2: regressed, succeeded before, failed after", ["mark down", "plain pass", "plain fail"]),
        (3, "Seed 3: still fails, failed before, failed after", ["mark fail", "plain fail", "plain fail"]),
    ]
    assert all(c["tip"] == c["name"] for c in columns)
    assert [c["tab"] for c in columns] == ["0", "-1", "-1"]


def test_more_than_20_seeds_keep_seed_order_and_an_after_not_run_reads_as_not_run() -> None:
    first = [(seed, "errored" if seed == 5 else "failed" if seed % 2 else "succeeded") for seed in range(30, 0, -1)]
    events = [*evaluated(0, *first), started("trial"), *held_out(1, 1, "succeeded"), *held_out(1, 2, None)]
    columns = strip(stream(*events))
    assert [c["seed"] for c in columns] == list(range(1, 31))
    by = {c["seed"]: (c["name"], c["cells"]) for c in columns}
    assert by[1] == ("Seed 1: improved, failed before, succeeded after", ["mark up", "plain fail", "plain pass"])
    assert by[2] == (
        "Seed 2: pending, succeeded before, not run after",
        ["mark pending", "plain pass", "plain pending"],
    )
    assert by[5] == ("Seed 5: errored, errored before, not run after", ["mark error", "plain error", "plain pending"])
    assert by[30] == (
        "Seed 30: pending, succeeded before, not run after",
        ["mark pending", "plain pass", "plain pending"],
    )
    assert strip(stream(*evaluated(0, (1, "failed"))))[0]["name"] == "Seed 1: pending, failed before, not run after"


def test_arrows_in_the_strip_move_between_its_columns_and_enter_on_one_opens_its_seed() -> None:
    script = """
    const at = (seed) => ({ closest: (s) => (s === "[data-column]" ? { dataset: { column: String(seed) } } : null) });
    const key = (k, seed) => {
      let claimed = false;
      doc.listeners.keydown[0]({ key: k, target: at(seed), preventDefault() { claimed = true; } });
      flush();
      return [doc.activeElement.selector || null, claimed];
    };
    const seen = [key("ArrowRight", 1), key("ArrowRight", 2), key("ArrowRight", 3), key("ArrowLeft", 1)];
    const tab = writes.filter(([id]) => id === "columns").pop()[1].match(/data-column="(\d+)"[^>]*tabindex="0"/)[1];
    const run = vm.runInContext("[view.selected, view.seed]", context);
    // Enter on a focused button is its click
    doc.getElementById("columns").listeners.click[0]({ target: { closest: () => ({ dataset: { column: "2" } }) } });
    flush();
    return { seen, tab, run, open: vm.runInContext("[MomState.where(view), view.seed]", context) };
    """
    out = app(REPLAY + script, stream(*TWO_EVALUATIONS))
    assert out["seen"] == [
        ['[data-column="2"]', True],
        ['[data-column="3"]', True],
        ['[data-column="3"]', True],
        ['[data-column="3"]', True],
    ]
    assert out["tab"] == "3"
    assert out["run"] == [None, None]
    # the harness's narrow stage defaults the switch to After, so the seed shows its latest episode in full
    assert out["open"] == ["episode", 2]


def test_a_run_that_did_not_start_says_why_once_on_the_stage() -> None:
    script = """
    const last = (id) => writes.filter(([at]) => at === id).pop()[1];
    return { control: last("control"), stage: last("stage"), strip: last("columns") };
    """
    refused = ("error", {"error": "not started: Not yet: the loop teaches one task"})
    out = app(REPLAY + script, stream(("instruction", {"text": "juggle", "source": "teach"}), refused))
    assert "Not started" in out["control"] and "Not yet" not in out["control"]
    assert "Not yet: the loop teaches one task" in out["stage"]
    assert "Not started" in out["strip"] and "Not yet" not in out["strip"]


def test_a_real_run_before_its_first_episode_reads_live_in_the_header_not_placeholder() -> None:
    script = 'return writes.filter(([id]) => id === "status").pop()[1];'
    fresh = "for (const e of events) e.t = Date.now() / 1000;"
    taught = stream(("instruction", {"text": "put the box in the bin", "source": "teach"}), argon_call(1))
    status = app(fresh + REPLAY + script, taught)
    assert 'class="chip live"' in status and "Placeholder" not in status


HEARD = """
const heard = [];
const said = { get: () => heard.at(-1) || "", set: (t) => heard.push(t) };
Object.defineProperty(doc.getElementById("announce"), "textContent", said);
"""


def test_a_new_episode_elsewhere_in_the_run_never_re_announces_the_episode_shown() -> None:
    later = [e for seed in (5, 6, 7) for e in held_out(1, seed, "failed")]
    script = f"""
    const later = events.splice(events.length - {len(later)});
    {REPLAY}
    vm.runInContext(`MomState.visit(view, {{ tab: "teaching" }}); render();`, context);
    const shown = heard.length;
    for (const e of later) {{ sources[0].onmessage({{ data: JSON.stringify(e) }}); flush(); }}
    return heard.slice(shown - 1);
    """
    out = app(HEARD + script, stream(started("a"), decision(0), executed(0), verdict(False), *later))
    assert len(out) == 1 and out[0].startswith("Episode 1 of 1, student trial.")


def test_a_seed_opened_from_the_strip_is_announced_by_its_mark_and_going_back_by_the_episode_returned_to() -> None:
    script = """
    vm.runInContext(`
      chosen = true; MomState.choose(view, "both");
      MomState.visit(view, { seed: 1 }); render();
      MomState.keydown(view, { key: "ArrowRight" }, Date.now() / 1000); render();
      MomState.keydown(view, { key: "Escape" }, Date.now() / 1000); render();
    `, context);
    return heard;
    """
    heard = app(HEARD + REPLAY + script, stream(*TWO_EVALUATIONS))
    assert heard[1:3] == ["Seed 1, improved: Failed → Succeeded.", "Seed 2, regressed: Succeeded → Failed."]
    assert heard[0] == heard[3] and heard[3].startswith("Episode 4 of 7, student trial.")


@pytest.mark.parametrize("n", [199, 200, 201])
def test_past_200_characters_the_count_says_so_and_teach_stays_off_until_the_text_is_shortened(n: int) -> None:
    script = f"""
    const field = doc.getElementById("instruction");
    field.value = "\N{ROBOT FACE}".repeat({n});
    field.listeners.input[0]();
    const [count, go, status] = ["teach-count", "teach-go", "teach-status"].map((id) => doc.getElementById(id));
    return [count.textContent, go.disabled, status.textContent];
    """
    status = "At most 200 characters." if n > 200 else ""
    assert app(REPLAY + script, stream(started("a"))) == [f"{n}/200", n > 200, status]


def test_the_run_picker_closes_when_the_focus_moves_past_it_or_on_escape_and_not_for_a_click_on_its_own_text() -> None:
    script = """
    const picker = doc.getElementById("runs");
    const link = {};
    picker.contains = (el) => el === link;
    const left = (relatedTarget) => {
      picker.open = true;
      picker.listeners.focusout[0]({ relatedTarget });
      return picker.open;
    };
    picker.open = true;
    doc.listeners.keydown[0]({ key: "Escape", target: doc.body, preventDefault() {} });
    return { escape: picker.open, link: left(link), text: left(null), past: left(doc.getElementById("columns")) };
    """
    assert app(REPLAY + script, stream(started("a"))) == {"escape": False, "link": True, "text": True, "past": False}


def test_a_panel_reads_back_the_html_written_to_it() -> None:
    script = 'return doc.getElementById("p-rounds").innerHTML;'
    assert "Held-out success" in app(REPLAY + script, stream(started("a")))


def test_the_runs_list_is_asked_for_every_3_s_and_again_right_after_a_teach() -> None:
    script = """
    const every = intervals.filter(([name]) => name === "loadRuns").map(([, ms]) => ms);
    const field = doc.getElementById("instruction");
    field.value = "juggle three boxes";
    const asked = fetched.length;
    doc.getElementById("teach").listeners.submit[0]({ preventDefault() {} });
    for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
    return { every, after: fetched.slice(asked) };
    """
    assert app(REPLAY + script, stream(started("a"))) == {"every": [3000], "after": ["/teach", "/runs"]}


def test_a_run_file_the_server_says_is_not_written_yet_reads_as_not_in_the_runs_files() -> None:
    script = """
    for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
    flush();
    const shown = writes.filter(([id]) => id === "p-training").pop()[1];
    return { asked: fetched.filter((url) => url.startsWith("/run-files/")), shown };
    """
    out = app(REPLAY + script, stream(started("a"), ("finetune_started", {"round": 1})))
    assert out["asked"] == ["/run-files/r/training_r1.summary.json", "/run-files/r/replay_report.json"]
    assert out["shown"].count("Not in the run's files.") == 2


def test_an_errored_held_out_episode_reads_errored_not_counted_in_its_own_view_never_failed() -> None:
    script = """
    const last = (id) => writes.filter(([at]) => at === id).pop()[1];
    return { control: last("control"), log: last("log"), heard: heard.at(-1) };
    """
    episode = ("episode_started", {**started("h")[1], "split": "heldout"})
    out = app(HEARD + REPLAY + script, stream(episode, decision(0), executed(0), RECORDING_REFUSED))
    for html in (out["control"], out["log"]):
        assert "Errored, not counted" in html and "#m-error" in html and "Failed" not in html
    assert 'class="verdict warn"' in out["control"]
    assert 'class="event warn"' in out["log"]
    assert out["heard"].endswith("Verdict: Errored, not counted.")


def test_a_fine_tuned_checkpoint_shows_by_its_short_id_and_the_names_tooltip_keeps_the_full_ids() -> None:
    # wide enough for every checkpoint side by side before the seed opens
    wide = 'doc.getElementById("results").clientWidth = 1280;'
    script = """
    vm.runInContext(`
      MomState.visit(view, { seed: 1 }); render();
      MomState.visit(view, { seed: null, selected: MomState.current(MomState.shown(view)).id }); render();
    `, context);
    const last = (id) => writes.filter(([at]) => at === id).pop()[1];
    // A tooltip may carry the full id; nothing else on the page does.
    const shown = writes.map(([, html]) => html.replace(/title="[^"]*"/g, "")).filter((h) => h.includes("r/ckpt-r"));
    return { shown, rounds: last("p-rounds"), seed: last("results"), control: last("control") };
    """
    events = [*evaluated(0, (1, "failed")), started("trial"), *evaluated(1, (1, "succeeded")), *held_out(2, 1, None)]
    out = app(wide + REPLAY + script, stream(*events, executed(0)))
    assert out["shown"] == []
    assert 'title="ckpt-r0 → r/ckpt-r1"' in out["rounds"]
    assert 'Testing <span class="mono">ckpt-r2</span>' in out["rounds"]
    assert (
        '<span class="mono note">ckpt-r1</span>' in out["seed"] and '<span class="mono">ckpt-r2</span>' in out["seed"]
    )
    assert 'Student <span class="mono">ckpt-r2</span>' in out["control"]


def test_a_run_reads_live_on_the_servers_clock_when_the_viewers_runs_30_s_ahead() -> None:
    script = """
    // The server's clock, as its /runs answer's Date header gives it, is 30 s behind this one.
    const behind = new Date(Date.now() - 30000).toUTCString();
    context.fetch = async () => ({ ok: true, status: 200, headers: new Map([["date", behind]]), json: async () => [] });
    await vm.runInContext("loadRuns()", context);
    for (const e of events) e.t = Date.now() / 1000 - 30;
    """
    script += REPLAY + 'return doc.getElementById("status").innerHTML;'
    assert 'class="chip live"' in app(script, stream(started("a"), decision(0)))


def test_a_crash_carrying_another_seed_never_lands_in_the_current_episodes_log() -> None:
    crash = ("error", {"seed": 105, "split": "heldout", "round": 0, "error": "RuntimeError: reset failed"})
    own = ("error", {"seed": 104, "split": "heldout", "round": 0, "error": "recorder refused step 1"})
    stopped = ("error", {"error": "mom run stopped (exit 1)"})
    out = fold(*held_out(0, 104, "failed"), crash, own, stopped)
    assert [r.get("text") or r["kind"] for r in out["ep"]["rows"]] == [
        "decision",
        "verdict",
        "recorder refused step 1",
        "mom run stopped (exit 1)",
    ]
    assert [c["seed"] for c in out["state"]["crashes"]] == [105, 104]


def test_a_paused_view_stays_put_when_the_run_stops_while_it_is_paused() -> None:
    stopped = ("error", {"error": "mom run stopped (exit 1)"})
    events = stream(started("a"), verdict(True), ("finetune_started", {"round": 1}), stopped)
    out = presenter(
        """
        feed(3);
        press(" ");
        const paused = [S.finetuning(v, now), S.following(v, now)];
        feed(1);
        const stopped = [S.finetuning(v, now), S.following(v, now)];
        press(" ");
        return [paused, stopped, [S.finetuning(v, now), S.following(v, now)]];
        """,
        events,
    )
    # finetune_started was written at t=2 and the view paused as it arrived, at t=3
    assert out == [[1, True], [1, True], [None, False]]


def test_the_filmstrip_ends_on_the_stage_frame_so_a_strip_that_drops_its_earliest_keeps_it() -> None:
    steps = [x for step in range(8) for x in (decision(step), executed(step, frame=f"/f/{step}.jpg"))]
    script = """
    const s = S.initial();
    for (const e of events) S.reduce(s, e);
    const strip = (focus) => S.filmstrip(S.current(s), focus).map((f) => (f.now ? `[${f.step}]` : f.step)).join(" ");
    return [null, 0, 1, 4].map(strip);
    """
    assert run(script, stream(started("a"), *steps)) == ["2 3 4 5 6 [7]", "[0]", "0 [1]", "0 1 2 3 [4]"]


def tick(round_: object, step: object, total: object) -> tuple[str, dict]:
    return "finetune_progress", {"round": round_, "step": step, "total": total}


def progress(*events: tuple[str, dict]) -> object:
    return fold(("finetune_started", {"round": 1}), *events)["state"]["finetune"]["pending"]["progress"]


def test_the_finetune_progress_is_the_latest_report_of_the_round_under_way_in_whole_percent() -> None:
    assert progress() is None
    assert progress(tick(1, 0, 300)) == 0
    assert progress(tick(1, 29, 100)) == 29  # whole steps, never a float that rounds 29 down to 28
    assert progress(tick(1, 100, 300)) == 33
    assert progress(tick(1, 300, 300)) == 100
    assert progress(tick(1, 30, 300), tick(1, 150, 300)) == 50
    assert progress(tick(1, 150, 300), tick(0, 299, 300), tick(2, 1, 300), ("finetune_progress", {})) == 50


@pytest.mark.parametrize(
    "step,total", [(None, 300), (1.5, 3), ("1", 3), (1, "3"), (1, 0), (1, -3), (4, 3), (-1, 3), (True, 3)]
)
def test_a_malformed_finetune_progress_report_reads_starting_never_a_guess(step: object, total: object) -> None:
    assert progress(tick(1, 1, 3), tick(1, step, total)) is None


def test_a_finetune_progress_ends_with_its_finetune_and_a_new_round_starts_from_starting() -> None:
    done = fold(("finetune_started", {"round": 1}), tick(1, 3, 3), ("finetune_completed", {"round": 1}))
    assert done["state"]["finetune"]["pending"] is None
    assert progress(tick(1, 3, 3), ("finetune_completed", {"round": 1}), ("finetune_started", {"round": 2})) is None


def test_the_finetune_stage_shows_a_progressbar_with_its_percentage_or_starting() -> None:
    script = 'return writes.filter(([id]) => id === "stage").pop()[1];'
    fresh = "for (const e of events) e.t = Date.now() / 1000;"
    stage = app(fresh + REPLAY + script, stream(started("a"), ("finetune_started", {"round": 1}), tick(1, 120, 400)))
    assert re.search(r'role="progressbar"[^>]*aria-valuenow="30"', stage) and ">30%<" in stage
    stage = app(fresh + REPLAY + script, stream(started("a"), ("finetune_started", {"round": 1})))
    assert 'role="progressbar"' in stage and "aria-valuenow" not in stage and ">Starting<" in stage


INDEX_HTML = STATE_JS.parent / "index.html"
LOADING = ("control", "columns", "stage", "p-rounds", "p-diagnosis", "p-training", "log")


class Skeleton(HTMLParser):
    VOID = {"meta", "link", "input", "br", "img"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, str | None, bool]] = []
        self.blocks: list[tuple[str | None, bool]] = []
        self.loading = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        _, container, hidden = self.stack[-1] if self.stack else ("", None, False)
        here = (a.get("id") if a.get("id") in LOADING else container, hidden or a.get("aria-hidden") == "true")
        if "skel" in (a.get("class") or "").split():
            self.blocks.append(here)
        if tag not in self.VOID:
            self.stack.append((tag, *here))

    def handle_endtag(self, tag: str) -> None:
        while self.stack and self.stack.pop()[0] != tag:
            pass

    def handle_data(self, data: str) -> None:
        if data.strip() == "Loading" and not (self.stack and self.stack[-1][2]):
            self.loading += 1


def test_a_run_shows_hidden_skeletons_and_says_loading_once_until_its_replay_settles_then_its_content() -> None:
    page = Skeleton()
    page.feed(INDEX_HTML.read_text())
    assert {container for container, _ in page.blocks} == set(LOADING)
    assert all(hidden for _, hidden in page.blocks)
    assert page.loading == 1
    script = """
    const [source] = sources;
    source.onopen();
    for (const e of events) source.onmessage({ data: JSON.stringify(e) });
    const replaying = writes.map(([id]) => id);
    source.listeners.replayed();
    flush();
    return { replaying, after: Object.fromEntries(writes) };
    """
    out = app(script, stream(started("a"), decision(0), executed(0)))
    assert not set(out["replaying"]) & set(LOADING)
    for id_ in LOADING:
        assert id_ in out["after"] and not re.search(r'class="[^"]*\bskel', out["after"][id_]), id_
