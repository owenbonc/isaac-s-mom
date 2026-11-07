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
    skill = {"skill": "push_forward", "params": {}}
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


def test_median_latency_carries_its_sample_size_per_controller() -> None:
    out = fold(
        ("episode_started", {"episode_id": "e"}),
        decision(0, latency=10.0),
        decision(1, latency=30.0),
        decision(2, latency=20.0),
        decision(3, controller="teacher", latency=500.0),
        ("argon_call", {"latency_ms": 900.0}),
        ("argon_call", {"latency_ms": 1100.0}),
    )
    assert out["summary"]["student"] == {"median": 20.0, "n": 3}
    assert out["summary"]["teacher"] == {"median": 500.0, "n": 1}
    assert out["summary"]["argon"] == {"median": 1000.0, "n": 2}
    assert out["summary"]["finetune"] is None


def test_missing_or_bad_latency_is_not_counted() -> None:
    out = fold(("episode_started", {"episode_id": "e"}), decision(0, latency=None), decision(1, latency="fast"))
    assert out["summary"]["student"] is None


def test_the_first_frame_after_the_perturbation_is_flagged() -> None:
    out = fold(
        ("episode_started", {"episode_id": "e"}),
        decision(0),
        executed(0),
        decision(1),
        ("perturbation", {"kind": "move_block", "step": 2}),
        decision(2),
        executed(1),
        executed(2),
    )
    assert [f["moved"] for f in out["filmstrip"]] == [False, False, True]


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
        ("argon_call", {"latency_ms": 900.0}),
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
    assert out["summary"]["argon"]["n"] == 1


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


def provenance(events: list[dict], now: float, episode: int | None = None, paused: bool = False) -> str:
    script = f"""
    const v = S.view();
    for (const e of events) S.reduce(v.live, e);
    if ({"true" if paused else "false"}) S.keydown(v, {{ key: " " }});
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


def test_a_run_stops_being_live_20_s_after_its_last_event() -> None:
    events = stream(started("a"))  # last event at t=0
    assert provenance(events, now=19.9) == "live"
    assert provenance(events, now=20.1) == "recorded"


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
const feed = (n) => { for (const e of events.splice(0, n)) S.reduce(v.live, e); };
const press = (key) => S.keydown(v, { key });
const look = () => {
  const ep = S.episode(v);
  return { shown: S.shown(v).eventCount, held: S.held(v), episode: ep && ep.id, focus: v.compare && v.compare.step };
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


def test_p_jumps_to_the_perturbation_frame_and_pins_the_episode_until_pressed_again() -> None:
    events = stream(
        started("a"),
        *[x for step in range(6) for x in (decision(step), executed(step, frame=f"/f/{step}.jpg"))],
        ("perturbation", {"kind": "move_block", "step": 2}),
        started("b"),
    )
    events.insert(5, events.pop(-2))  # the perturbation arrives before decision 2
    out = presenter(
        """
        feed(events.length - 1);
        press("p");
        const jumped = look();
        const strip = S.filmstrip(S.episode(v), v.compare.step).map((f) => [f.step, f.moved, f.now]);
        feed(1);
        const pinned = look();
        press("P");
        return { jumped, strip, pinned, again: look() };
        """,
        events,
    )
    assert out["jumped"]["focus"] == 2 and out["jumped"]["episode"] == "a"
    assert [f for f in out["strip"] if f[2]] == [[2, True, True]]
    assert out["pinned"]["episode"] == "a"
    # pressing P again returns to where the view was: following the run, now on "b"
    assert out["again"] == {"shown": len(events), "held": 0, "episode": "b", "focus": None}


def test_p_pressed_again_returns_to_a_pinned_episode() -> None:
    events = stream(
        started("a"),
        started("b"),
        decision(0),
        executed(0),
        ("perturbation", {"kind": "move_block", "step": 1}),
        decision(1),
        executed(1),
        started("c"),
    )
    out = presenter(
        """
        feed(events.length);
        press("ArrowLeft");
        const before = look();
        press("p");
        const during = look();
        press("p");
        return [before, during, look()];
        """,
        events,
    )
    assert [o["episode"] for o in out] == ["b", "b", "b"]
    assert [o["focus"] for o in out] == [None, 1, None]


def test_p_does_nothing_on_an_episode_without_a_perturbation() -> None:
    out = presenter(
        "feed(3); press('p'); return look();",
        stream(started("a"), decision(0), executed(0)),
    )
    assert out == {"shown": 3, "held": 0, "episode": "a", "focus": None}


def test_space_pauses_from_any_focus_claims_the_key_and_acts_once_while_held() -> None:
    out = presenter(
        """
        feed(events.length);
        const button = { tagName: "BUTTON" };  // a focused presenter button sends the key too
        const claimed = S.keydown(v, { key: " ", target: button });
        const paused = Boolean(v.frozen);
        const held = S.keydown(v, { key: " ", repeat: true, target: button });
        const stillPaused = Boolean(v.frozen);
        const ctrl = S.keydown(v, { key: " ", ctrlKey: true });
        const other = S.keydown(v, { key: "q" });
        S.keydown(v, { key: " " });
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
