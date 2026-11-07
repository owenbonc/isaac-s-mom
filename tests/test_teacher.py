"""Workstream 2: the Argon teacher (SPEC §3.3, rules 1 and 5). A fake Argon client; no network calls."""

import ast
import json
import subprocess
import sys
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
from isaacs_mom.teacher import ATTEMPTS, ArgonPolicy, choose_scenes, diagnose
from isaacs_mom.teacher.argon import (
    RETRY_WAITS_S,
    Argon,
    ArgonUnavailable,
    MissingKey,
    Part,
    env_value,
    gemini_generate,
    retry_after_s,
    stub_generate,
)

MOVE: Skill = next(s for s in SKILLS if s != "reobserve")
STEP = json.dumps({"skill": MOVE, "params": {"distance_cm": 5}, "reason": "box is left of centre"})
LOOK = json.dumps({"skill": "reobserve", "reason": "the box may have moved"})
DONE = json.dumps({"done": True, "reason": "box is in the bin"})


class FakeArgon:
    """Answers in order and records every prompt it was sent."""

    def __init__(self, *answers: str | Exception) -> None:
        self.answers = list(answers)
        self.prompts: list[list[Part]] = []

    def __call__(self, purpose: str, parts: list[Part]) -> str:
        self.prompts.append(parts)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def argon(
    tmp_path: Path, *answers: str | Exception, budget: int = 50, waits: list[float] | None = None
) -> tuple[Argon, FakeArgon]:
    fake = FakeArgon(*answers)
    sleep = (waits if waits is not None else []).append
    return Argon(EventLog(tmp_path, "run"), budget=budget, generate=fake, sleep=sleep), fake


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
        json.dumps({"skill": MOVE}).replace("}", ', "params": {"distance_cm": ' + "9" * 5000 + "}}"),  # ValueError
    ],
)
def test_unparseable_answers_are_asked_again_not_recorded_and_logged_raw(tmp_path: Path, bad: str) -> None:
    a, fake = argon(tmp_path, bad, STEP)
    call = ArgonPolicy(None, a).act(obs(), "task", "start")
    assert call is not None and call.skill == MOVE
    assert len(fake.prompts) == 2
    assert [e["raw"] for e in events(tmp_path, "error")] == [bad[:500]]


def test_an_answer_that_never_parses_ends_the_demo_with_one_reobserve(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, *["nonsense"] * ATTEMPTS)
    assert demo(ArgonPolicy(None, a)) == ["reobserve"]
    assert len(fake.prompts) == ATTEMPTS


def test_a_huge_integer_in_a_diagnosis_is_unparseable_not_a_crash(tmp_path: Path) -> None:
    huge = '{"hypothesis": "x", "evidence": "y", "confidence": ' + "9" * 5000 + ', "lesson": "z"}'
    a, _ = argon(tmp_path, huge, budget=1)
    assert diagnose(failed_episode(), [b"f"], a).confidence == 0.0


def test_an_unreachable_argon_is_retried_after_a_wait_without_recording_a_step(tmp_path: Path) -> None:
    down = ArgonUnavailable("offline")
    waits: list[float] = []
    a, fake = argon(tmp_path, down, down, STEP, DONE, waits=waits)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]
    assert len(fake.prompts) == 4
    assert waits == list(RETRY_WAITS_S[:2])


def test_a_rate_limit_waits_as_long_as_the_server_says_capped(tmp_path: Path) -> None:
    waits: list[float] = []
    limited = [ArgonUnavailable("429", retry_after_s=37.0), ArgonUnavailable("429", retry_after_s=999.0)]
    a, _ = argon(tmp_path, *limited, STEP, waits=waits)
    call = ArgonPolicy(None, a).act(obs(), "task", "start")
    assert call is not None and call.skill == MOVE
    assert waits == [37.0, 120.0]


def test_a_minute_long_rate_limit_does_not_end_the_demo(tmp_path: Path) -> None:
    waits: list[float] = []
    a, _ = argon(tmp_path, *[ArgonUnavailable("429")] * (ATTEMPTS - 1), STEP, DONE, waits=waits)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]
    assert sum(waits) >= 60


def test_argon_down_for_a_whole_step_ends_the_demo_with_one_reobserve(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, STEP, LOOK, *[ArgonUnavailable("offline")] * ATTEMPTS)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]


def test_no_two_reobserves_in_a_row_before_the_end_whatever_argon_does(tmp_path: Path) -> None:
    down = ArgonUnavailable("offline")
    a, _ = argon(tmp_path, LOOK, down, "junk", LOOK, STEP, LOOK, "", LOOK, DONE, budget=50)
    skills = demo(ArgonPolicy(None, a))
    assert not any(x == y == "reobserve" for x, y in zip(skills[:-1], skills[1:], strict=False))
    assert skills[-1] == "reobserve"


def test_an_empty_answer_is_a_failed_call_not_ok(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, *[""] * ATTEMPTS)
    assert demo(ArgonPolicy(None, a)) == ["reobserve"]
    assert all(not e["ok"] and e["error"] == "empty answer" for e in events(tmp_path, "argon_call"))


def test_over_budget_ends_the_demo_without_another_call(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, STEP, budget=1)
    assert demo(ArgonPolicy(None, a)) == [MOVE, "reobserve"]
    assert len(fake.prompts) == 1
    assert not events(tmp_path, "error")  # it stopped, rather than knocking on a closed budget three times


def test_a_missing_key_fails_when_the_generator_is_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(argon_module, "ENV_PATH", tmp_path / "no.env")
    with pytest.raises(MissingKey):
        Argon(EventLog(tmp_path, "run"), budget=5)


def test_a_bug_in_our_code_is_not_mistaken_for_unreachable(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, KeyError("our bug"))
    with pytest.raises(KeyError):
        ArgonPolicy(None, a).act(obs(), "task", "start")


def fake_client(
    monkeypatch: pytest.MonkeyPatch, made: list[dict[str, object]], answer: object, preflight: Exception | None = None
) -> None:
    from google import genai

    def get(**_: object) -> SimpleNamespace:
        if preflight:
            raise preflight
        return SimpleNamespace(name="models/gemini-4-argon")

    def generate_content(**_: object) -> SimpleNamespace:
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(text=answer)

    def client(**kwargs: object) -> SimpleNamespace:
        made.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(get=get, generate_content=generate_content))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", client)


def test_the_sdk_client_has_a_timeout_and_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[dict[str, object]] = []
    fake_client(monkeypatch, made, STEP)
    generate = gemini_generate()
    assert generate("act", ["a"]) == generate("act", ["b"]) == STEP
    assert len(made) == 1
    assert made[0]["http_options"].timeout == argon_module.TIMEOUT_S * 1000  # type: ignore[union-attr]


def unreachable_errors() -> list[Exception]:
    import httpx2
    from google.genai import errors

    return [
        errors.ServerError(503, {"error": {"message": "overloaded"}}),
        errors.ClientError(429, {"error": {"message": "rate limited"}}),
        errors.ClientError(408, {"error": {"message": "request timeout"}}),
        errors.UnknownApiResponseError("not json"),
        httpx2.ConnectError("down"),
        httpx2.ReadTimeout("slow"),
        httpx2.DecodingError("bad gzip"),
    ]


@pytest.mark.parametrize("error", unreachable_errors(), ids=lambda e: type(e).__name__)
def test_server_rate_limit_transport_and_non_json_errors_count_as_unreachable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake_client(monkeypatch, [], error)
    with pytest.raises(ArgonUnavailable):
        gemini_generate()("act", ["hi"])


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_other_4xx_errors_fail_fast(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    from google.genai import errors

    fake_client(monkeypatch, [], errors.ClientError(code, {"error": {"message": "bad request"}}))
    with pytest.raises(errors.ClientError):
        gemini_generate()("act", ["hi"])


def test_a_redirect_loop_is_a_setup_fault_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx2

    fake_client(monkeypatch, [], httpx2.TooManyRedirects("loop"))
    with pytest.raises(httpx2.TooManyRedirects):
        gemini_generate()("act", ["hi"])


def test_our_own_bug_inside_the_sdk_call_is_not_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client(monkeypatch, [], KeyError("our bug"))
    with pytest.raises(KeyError):
        gemini_generate()("act", ["hi"])


def test_the_startup_check_is_one_counted_logged_call(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, '{"done": true}')
    a.check()
    assert len(fake.prompts) == 1 and a.calls == 1
    assert [e["purpose"] for e in events(tmp_path, "argon_call")] == ["startup"]


def test_a_startup_check_that_gets_no_answer_stops_the_run(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, ArgonUnavailable("down"))
    with pytest.raises(ArgonUnavailable, match="startup"):
        a.check()


@pytest.mark.parametrize("code", [401, 403, 404])
def test_a_bad_key_or_a_wrong_model_stops_the_run_at_the_startup_check(tmp_path: Path, code: int) -> None:
    from google.genai import errors

    a, _ = argon(tmp_path, errors.ClientError(code, {"error": {"message": "no"}}))
    with pytest.raises(errors.ClientError):
        a.check()


def test_the_server_wait_is_read_from_retry_info_or_the_header() -> None:
    from google.genai import errors

    info = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}
    assert retry_after_s(errors.ClientError(429, {"error": {"message": "q", "details": [info]}})) == 37.0
    with_header = errors.ClientError(429, {"error": {"message": "q"}})
    with_header.response = SimpleNamespace(headers={"retry-after": "12"})  # type: ignore[assignment]
    assert retry_after_s(with_header) == 12.0
    assert retry_after_s(errors.ClientError(429, {"error": {"message": "q"}})) is None


def test_tests_cannot_build_a_live_client_or_send_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import models

    monkeypatch.setenv("GEMINI_API_KEY", "a-real-looking-key")
    with pytest.raises(RuntimeError, match="live Gemini"):
        gemini_generate()
    with pytest.raises(RuntimeError, match="live Gemini"):
        models.Models.generate_content(object(), model="m", contents="hi")  # type: ignore[arg-type]


def test_the_stub_teacher_needs_no_key_and_ends_each_demo_at_once(tmp_path: Path) -> None:
    a = Argon(EventLog(tmp_path, "run"), budget=5, generate=stub_generate)
    assert demo(ArgonPolicy(None, a)) == ["reobserve"]
    assert diagnose(failed_episode(), [b"f"], a).confidence == 0.0
    assert "argon: 2/5 calls, 0 unavailable" in a.summary()


def test_the_summary_reports_failures_and_a_used_up_budget(tmp_path: Path) -> None:
    a, _ = argon(tmp_path, ArgonUnavailable("offline"), budget=1)
    a.ask("act", ["x"])
    assert a.summary() == "argon: 1/1 calls, 1 unavailable or empty, budget used up"


def test_every_call_counts_and_is_logged_even_when_it_fails_and_the_cap_stops_calls(tmp_path: Path) -> None:
    a, fake = argon(tmp_path, ArgonUnavailable("offline"), "", STEP, budget=2)
    assert [a.ask("act", ["x"]) for _ in range(3)] == [None, None, None]
    assert len(fake.prompts) == 2
    logged = events(tmp_path, "argon_call")
    assert [e["ok"] for e in logged] == [False, False]
    assert all(isinstance(e["latency_ms"], float) for e in logged)
    assert any("budget" in str(e["error"]) for e in events(tmp_path, "error"))


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
    assert "over budget" in diagnose(failed_episode(), [b"f"], a).hypothesis


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


def test_the_teacher_never_loads_the_simulator() -> None:
    probe = "import sys, isaacs_mom.teacher; print([m for m in sys.modules if m.startswith('isaacs_mom.sim')])"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_no_teacher_code_imports_the_simulator_even_inside_a_function() -> None:
    folder = Path(__file__).resolve().parent.parent / "isaacs_mom" / "teacher"
    for path in folder.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("isaacs_mom.sim") for a in node.names), path
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
                assert not any(n.split(".")[-1] == "sim" or ".sim." in f"{n}." for n in names), path
            # A computed module name can't be checked statically, so dynamic imports are banned outright here.
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "importlib" for a in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "importlib", path
            if isinstance(node, ast.Name | ast.Attribute):
                assert getattr(node, "id", getattr(node, "attr", "")) not in ("__import__", "import_module"), path


@pytest.mark.parametrize(
    "line",
    [
        "GEMINI_API_KEY=abc",
        "export GEMINI_API_KEY=abc",
        "GEMINI_API_KEY=abc # note",
        "GEMINI_API_KEY='abc'",
        'GEMINI_API_KEY="abc',  # unmatched quote
        "GEMINI_API_KEY=abc\t# note after a tab",
        "\ufeffGEMINI_API_KEY=abc",  # a BOM on line 1
        "GEMINI_API_KEY=old\nGEMINI_API_KEY=abc",  # the last definition wins
        "GEMINI_API_KEY=\nGEMINI_API_KEY=abc",  # an empty one doesn't stop the scan
    ],
)
def test_the_key_is_read_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{line}\n# comment\n" if line.startswith("\ufeff") else f"# comment\n{line}\n", encoding="utf-8")
    assert env_value("GEMINI_API_KEY", env) == "abc"
    assert env_value("GEMINI_API_KEY", tmp_path / "missing") is None


def test_the_documented_fake_run_needs_no_key_and_asking_for_argon_without_one_stops_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from isaacs_mom.orchestrator import cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(argon_module, "ENV_PATH", tmp_path / "no.env")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["mom", "run", "--sim", "fake", "--student", "stub", "--rounds", "0", "--run-id", "doc"]
    )
    cli.main()
    out = capsys.readouterr().out
    assert "(stub teacher)" in out and "argon: 0/" in out  # the stub spends nothing, not even a startup check
    monkeypatch.setattr(sys, "argv", ["mom", "run", "--sim", "fake", "--teacher", "argon", "--rounds", "0"])
    with pytest.raises(MissingKey):
        cli.main()


def test_the_mujoco_default_needs_a_key_and_stops_before_building_the_sim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from isaacs_mom.orchestrator import cli

    built: list[str] = []
    monkeypatch.setattr(cli, "make_sim", lambda name: built.append(name))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(argon_module, "ENV_PATH", tmp_path / "no.env")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mom", "run", "--sim", "mujoco", "--rounds", "0"])
    with pytest.raises(MissingKey):
        cli.main()
    assert built == []


def test_a_bad_key_stops_the_cli_at_the_startup_check_before_the_sim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.genai import errors

    from isaacs_mom.orchestrator import cli

    def bad_key(purpose: str, parts: list[Part]) -> str:
        raise errors.ClientError(401, {"error": {"message": "API key not valid"}})

    built: list[str] = []
    monkeypatch.setattr(cli, "make_sim", lambda name: built.append(name))
    monkeypatch.setattr(argon_module, "gemini_generate", lambda: bad_key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mom", "run", "--sim", "mujoco", "--rounds", "0"])
    with pytest.raises(errors.ClientError):
        cli.main()
    assert built == []


@pytest.mark.parametrize("bad", ["-3s", "nans", "infs", "-1e9s"])
def test_a_negative_or_non_finite_retry_delay_is_ignored(bad: str) -> None:
    from google.genai import errors

    info = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": bad}
    assert retry_after_s(errors.ClientError(429, {"error": {"message": "q", "details": [info]}})) is None
    with_header = errors.ClientError(429, {"error": {"message": "q"}})
    with_header.response = SimpleNamespace(headers={"retry-after": bad[:-1]})  # type: ignore[assignment]
    assert retry_after_s(with_header) is None


@pytest.mark.parametrize("server_wait", [-3.0, float("nan"), float("-inf")])
def test_a_bad_server_wait_never_reaches_sleep_as_negative_or_nan(tmp_path: Path, server_wait: float) -> None:
    waits: list[float] = []
    a, _ = argon(tmp_path, ArgonUnavailable("429", retry_after_s=server_wait), STEP, waits=waits)
    ArgonPolicy(None, a).act(obs(), "task", "start")
    assert len(waits) == 1 and 0.0 <= waits[0] <= 120.0


def test_a_fatal_4xx_is_still_logged_as_a_call(tmp_path: Path) -> None:
    from google.genai import errors

    a, _ = argon(tmp_path, errors.ClientError(401, {"error": {"message": "API key not valid"}}))
    with pytest.raises(errors.ClientError):
        a.check()
    [call] = events(tmp_path, "argon_call")
    assert call["purpose"] == "startup" and call["ok"] is False and "ClientError" in str(call["error"])
    assert a.calls == 1


def test_failing_the_last_budgeted_call_ends_at_once_without_a_wait(tmp_path: Path) -> None:
    waits: list[float] = []
    a, _ = argon(tmp_path, ArgonUnavailable("down"), budget=1, waits=waits)
    assert demo(ArgonPolicy(None, a)) == ["reobserve"]
    assert waits == []


def test_a_budget_spent_on_the_last_attempt_is_reported_as_over_budget(tmp_path: Path) -> None:
    down = ArgonUnavailable("down")
    a, _ = argon(tmp_path, down, down, down, budget=ATTEMPTS)
    call = ArgonPolicy(None, a).act(obs(), "task", "start")
    assert call is not None and call.skill == "reobserve" and "budget" in call.reason
    a2, _ = argon(tmp_path / "d", down, down, down, budget=ATTEMPTS)
    assert "over budget" in diagnose(failed_episode(), [b"f"], a2).hypothesis
