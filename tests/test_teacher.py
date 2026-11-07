"""Workstream 2: the Argon teacher (SPEC §3.3, rules 1 and 5). A fake Argon client; no network calls."""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import isaacs_mom.teacher.argon as argon_module
from isaacs_mom.contracts import (
    SKILLS,
    Diagnosis,
    EpisodeManifest,
    Observation,
    Skill,
    SkillCall,
    StepRecord,
    Validation,
)
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.seeds import SealedSeedError, heldout_seeds
from isaacs_mom.teacher import OFFLINE_REOBSERVES, PARSE_ATTEMPTS, ArgonPolicy, choose_scenes, diagnose
from isaacs_mom.teacher.argon import Argon, ArgonUnavailable, MissingKey, Part, env_value, gemini_generate

MOVE: Skill = next(s for s in SKILLS if s != "reobserve")
STEP = json.dumps({"skill": MOVE, "params": {"distance_cm": 5}, "reason": "box is left of centre"})
LOOK = json.dumps({"skill": "reobserve", "reason": "the box may have moved"})
DONE = json.dumps({"done": True, "reason": "box is in the bin"})


class FakeArgon:
    """Answers in order and records every prompt it was sent."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.prompts: list[list[Part]] = []

    def __call__(self, parts: list[Part]) -> str:
        self.prompts.append(parts)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def argon(tmp_path: Path, *answers: str | Exception, budget: int = 50) -> tuple[Argon, FakeArgon]:
    fake = FakeArgon(*answers)
    return Argon(EventLog(tmp_path, "run"), budget=budget, generate=fake), fake


def obs(step: int = 0, frame: bytes = b"jpeg-0") -> Observation:
    return Observation(episode_id="ep", step=step, sim_time=0.0, head_rgb_jpeg=frame, joint_state=[])


def events(tmp_path: Path, type_: str) -> list[dict[str, object]]:
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    return [e["data"] for e in map(json.loads, lines) if e["type"] == type_]


def demo(policy: ArgonPolicy, steps: int = 20) -> list[str]:
    """Skills the policy chose until it stopped."""
    skills: list[str] = []
    for i in range(steps):
        call = policy.act(obs(i), "task", "ok")
        if call is None:
            return skills
        skills.append(call.skill)
    raise AssertionError("the demo never ended")


# --- deciding from the camera ---


def test_act_parses_a_skill_call_and_sends_the_latest_frame(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, STEP)
    call = ArgonPolicy(None, a).act(obs(frame=b"frame-7"), "do the task", "start")
    assert call == SkillCall(skill=MOVE, params={"distance_cm": 5}, reason="box is left of centre")
    assert fake.prompts[0][-1] == b"frame-7"


def test_a_new_frame_and_the_last_result_reach_the_next_decision(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, STEP, LOOK)
    policy = ArgonPolicy(None, a)
    policy.act(obs(0, b"before"), "task", "start")
    policy.act(obs(1, b"after the box moved"), "task", f"{MOVE}: ok")
    prompt, frame = fake.prompts[1]
    assert frame == b"after the box moved" and f"{MOVE}: ok" in str(prompt)


# --- the done rule: exactly one closing reobserve, then None ---


def test_done_ends_with_exactly_one_reobserve_then_none(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, STEP, DONE)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]


def test_done_right_after_a_look_adds_no_second_reobserve(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, STEP, LOOK, DONE)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]


def test_a_second_look_in_a_row_ends_the_demo_instead_of_recording_it(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, STEP, LOOK, LOOK)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"skill": "fly"}',
        '{"skill": "reobserve", "params": {"speed": 1}}',
        json.dumps({"skill": MOVE, "params": {"distance_cm": "7"}}),  # no "7" for 7
        json.dumps({"skill": MOVE, "params": {"distance_cm": True}}),  # no true for 1
        json.dumps({"skill": MOVE, "extra": 1}),  # no extra keys
        json.dumps({"done": True, "skill": MOVE}),  # done alongside a skill
        json.dumps({"done": "yes"}),
        "[" * 100_000 + "]" * 100_000,  # RecursionError in json.loads
    ],
)
def test_unparseable_answers_are_asked_again_not_recorded_and_logged_raw(tmp_path: Path, bad: str) -> None:
    a, fake = argon(tmp_path, bad, STEP)
    call = ArgonPolicy(None, a).act(obs(), "task", "start")
    assert call is not None and call.skill == MOVE
    assert len(fake.prompts) == 2
    assert [e["raw"] for e in events(tmp_path, "error")] == [bad[:500]]


def test_an_answer_that_never_parses_ends_the_demo_with_one_reobserve(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, *["nonsense"] * PARSE_ATTEMPTS)
    assert demo(ArgonPolicy(None, a)) == ["reobserve"]
    assert len(fake.prompts) == PARSE_ATTEMPTS


# --- unreachable, missing key, our own bugs ---


def test_unreachable_argon_holds_and_reobserves_then_stops(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, *[ArgonUnavailable("offline")] * (OFFLINE_REOBSERVES + 1))
    assert demo(ArgonPolicy(None, a)) == ["reobserve"] * OFFLINE_REOBSERVES


def test_an_empty_answer_is_a_failed_call_not_ok(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, *[""] * (OFFLINE_REOBSERVES + 1))
    assert demo(ArgonPolicy(None, a)) == ["reobserve"] * OFFLINE_REOBSERVES
    assert all(not e["ok"] and e["error"] == "empty answer" for e in events(tmp_path, "argon_call"))


def test_a_missing_key_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(argon_module, "ENV_PATH", tmp_path / "no.env")
    with pytest.raises(MissingKey):
        Argon(EventLog(tmp_path, "run"), budget=5).ask("act", ["hi"])


def test_a_bug_in_our_code_is_not_mistaken_for_unreachable(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, KeyError("our bug"))
    with pytest.raises(KeyError):
        ArgonPolicy(None, a).act(obs(), "task", "start")


def test_the_sdk_client_has_a_timeout_and_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai

    made: list[dict[str, object]] = []

    class Client:
        def __init__(self, **kwargs: object) -> None:
            made.append(kwargs)
            self.models = SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=STEP))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", Client)
    generate = gemini_generate()
    assert generate(["a"]) == generate(["b"]) == STEP
    assert len(made) == 1
    assert made[0]["http_options"].timeout == argon_module.TIMEOUT_S * 1000  # type: ignore[union-attr]


def test_only_sdk_and_network_errors_count_as_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx2
    from google import genai
    from google.genai import errors

    def client_raising(error: Exception) -> Callable[..., object]:
        def boom(**_: object) -> None:
            raise error

        return lambda **_: SimpleNamespace(models=SimpleNamespace(generate_content=boom))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    for error in (errors.ClientError(429, {"error": {"message": "quota"}}), httpx2.ConnectError("down")):
        monkeypatch.setattr(genai, "Client", client_raising(error))
        with pytest.raises(ArgonUnavailable):
            gemini_generate()(["hi"])
    monkeypatch.setattr(genai, "Client", client_raising(KeyError("our bug")))
    with pytest.raises(KeyError):
        gemini_generate()(["hi"])


# --- rule 5: the budget and the call log ---


def test_every_call_counts_and_is_logged_even_when_it_fails_and_the_cap_stops_calls(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, ArgonUnavailable("offline"), "", STEP, budget=2)
    policy = ArgonPolicy(None, a)
    for i in range(3):
        policy.act(obs(i), "task", "x")
    assert len(fake.prompts) == 2
    logged = events(tmp_path, "argon_call")
    assert [e["ok"] for e in logged] == [False, False]
    assert all(isinstance(e["latency_ms"], float) for e in logged)
    assert any("budget" in str(e["error"]) for e in events(tmp_path, "error"))


# --- diagnose ---


def failed_episode(seed: int = 1001, accepted: bool = False) -> EpisodeManifest:
    step = StepRecord(
        step=0,
        sim_time=0.0,
        frame="frames/0000.jpg",
        joint_state=[],
        skill_call=SkillCall(skill="reobserve"),
        controller="student",
        checkpoint="ckpt-r0",
        actions="actions/0000.npy",
        timestamps=[0.0],
        ok=True,
    )
    return EpisodeManifest(
        episode_id=f"ep-{seed}",
        run_id="run",
        seed=seed,
        split="dev",
        round=0,
        instruction="do the task",
        sim_version="fake",
        robot_config="fake",
        steps=[step],
        validation=Validation(task_success=accepted, physics_valid=True, reason="box short of the bin"),
    )


def test_diagnose_sends_the_validator_reason_and_bounded_frames_and_parses_a_diagnosis(tmp_path: Path) -> None:
    answer = {"hypothesis": "never reached", "evidence": "hands stay open", "confidence": 0.7, "lesson": "line up"}
    a, fake = argon(tmp_path, json.dumps(answer))
    assert diagnose(failed_episode(), [bytes([i]) for i in range(10)], a) == Diagnosis(**answer)
    prompt, *frames = fake.prompts[0]
    assert "box short of the bin" in str(prompt)
    assert frames == [b"\x00", b"\x07", b"\x08", b"\x09"]


def test_diagnose_reports_an_unparseable_or_missing_answer_instead_of_guessing(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, '{"hypothesis": "x", "confidence": "high"}', budget=1)
    assert diagnose(failed_episode(), [b"f"], a).confidence == 0.0
    assert events(tmp_path, "error")[0]["raw"]
    assert "unavailable" in diagnose(failed_episode(), [b"f"], a).hypothesis  # over budget now


# --- choose_scenes: Owen's option D ---


def test_choose_scenes_re_teaches_rejected_dev_seeds_first_then_fresh_ones() -> None:
    failures = [failed_episode(1003), failed_episode(1001), failed_episode(1007, accepted=True)]
    seeds = choose_scenes(failures, 4, start=1500)
    assert seeds[:2] == [1003, 1001]
    assert 1007 not in seeds and len(seeds) == len(set(seeds)) == 4


def test_choose_scenes_never_returns_a_heldout_seed() -> None:
    sealed = min(heldout_seeds())
    leaked = failed_episode(sealed)
    assert sealed not in choose_scenes([leaked], 30, start=sealed)


def test_assert_not_heldout_still_guards_choose_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    import isaacs_mom.teacher as teacher

    monkeypatch.setattr(teacher, "heldout_seeds", lambda: frozenset())  # a bug that forgets the filter
    with pytest.raises(SealedSeedError):
        teacher.choose_scenes([], 1, start=min(heldout_seeds()))


# --- rule 1 and config ---


def test_the_teacher_never_loads_the_simulator() -> None:
    probe = "import sys, isaacs_mom.teacher; print([m for m in sys.modules if m.startswith('isaacs_mom.sim')])"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize(
    "line", ["GEMINI_API_KEY=abc", "export GEMINI_API_KEY=abc", "GEMINI_API_KEY=abc # note", "GEMINI_API_KEY='abc'"]
)
def test_the_key_is_read_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# comment\n{line}\n")
    assert env_value("GEMINI_API_KEY", env) == "abc"
    assert env_value("GEMINI_API_KEY", tmp_path / "missing") is None
