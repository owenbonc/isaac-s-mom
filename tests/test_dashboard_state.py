"""Behaviour of the dashboard's event reducer (static/state.js), run under node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATE_JS = Path(__file__).parent.parent / "isaacs_mom" / "orchestrator" / "dashboard" / "static" / "state.js"
DRIVER = """
const S = require(process.argv[1]);
const events = JSON.parse(require("fs").readFileSync(0, "utf8"));
const state = S.initial();
for (const e of events) S.reduce(state, e);
console.log(JSON.stringify({ state, summary: S.summary(state), filmstrip: S.filmstrip(state.ep) }));
"""

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def fold(*events: tuple[str, dict]) -> dict:
    stream = [{"t": float(i), "run_id": "r", "type": type_, "data": data} for i, (type_, data) in enumerate(events)]
    out = subprocess.run(
        ["node", "-e", DRIVER, str(STATE_JS)], input=json.dumps(stream), capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


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
    (row,) = out["state"]["ep"]["rows"]
    assert (row["reason"], row["ok"], row["error"]) == (reason, False, "joint limit")
    assert out["state"]["ep"]["frames"][0]["url"] == "/frames/a.jpg"


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
    assert out["state"]["ep"]["id"] == "b" and out["state"]["ep"]["rows"] == []
    assert out["state"]["episodeCount"] == 2


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
    kinds = [r["kind"] for r in out["state"]["ep"]["rows"]]
    assert kinds == ["takeover", "error", "verdict"]
    assert out["state"]["ep"]["verdict"] == {"task": False, "physics": True, "reason": "far"}
