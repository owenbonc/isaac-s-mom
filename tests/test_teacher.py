"""Workstream 2: the Argon teacher (SPEC §3.3, rules 1 and 5). A fake Argon client; no network calls."""

import ast
import json
from pathlib import Path

import pytest

from isaacs_mom.contracts import SKILLS, Diagnosis, EpisodeManifest, Observation, SkillCall, StepRecord, Validation
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.seeds import SealedSeedError, heldout_seeds
from isaacs_mom.teacher import OFFLINE_REOBSERVES, ArgonPolicy, choose_scenes, diagnose
from isaacs_mom.teacher.argon import Argon, Part, env_value


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


def obs(step: int = 0, frame: bytes = b"jpeg-0") -> Observation:
    return Observation(episode_id="ep", step=step, sim_time=0.0, head_rgb_jpeg=frame, joint_state=[])


def events(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def first_skill_and_param() -> tuple[str, str]:
    """A whitelisted moving skill, read from the contract so the tests follow a skill-list change."""
    return next(s for s in SKILLS if s != "reobserve"), "distance_cm"


def test_act_parses_a_skill_call_and_sends_the_latest_frame(tmp_path: Path) -> None:
    skill, param = first_skill_and_param()
    fake = FakeArgon(json.dumps({"skill": skill, "params": {param: 5}, "reason": "box is left of centre"}))
    call = ArgonPolicy(None, Argon(budget=5, generate=fake)).act(obs(frame=b"frame-7"), "do the task", "start")
    assert call == SkillCall(skill=skill, params={param: 5}, reason="box is left of centre")  # type: ignore[arg-type]
    assert fake.prompts[0][-1] == b"frame-7"


def test_a_new_frame_and_the_last_result_reach_the_next_decision() -> None:
    skill, _ = first_skill_and_param()
    fake = FakeArgon(json.dumps({"skill": skill}), json.dumps({"skill": "reobserve"}))
    policy = ArgonPolicy(None, Argon(budget=5, generate=fake))
    policy.act(obs(0, b"before"), "task", "start")
    policy.act(obs(1, b"after the box moved"), "task", f"{skill}: ok")
    second = fake.prompts[1]
    assert second[-1] == b"after the box moved"
    assert f"{skill}: ok" in str(second[0])


def test_done_ends_the_demonstration() -> None:
    policy = ArgonPolicy(None, Argon(budget=5, generate=FakeArgon('{"done": true}')))
    assert policy.act(obs(), "task", "start") is None


@pytest.mark.parametrize("answer", ["not json", '{"skill": "fly"}', '{"skill": "reobserve", "params": {"speed": 1}}'])
def test_unparseable_or_non_whitelisted_answers_become_reobserve(answer: str) -> None:
    call = ArgonPolicy(None, Argon(budget=5, generate=FakeArgon(answer))).act(obs(), "task", "start")
    assert call is not None and call.skill == "reobserve"


def test_unreachable_argon_holds_and_reobserves_then_stops() -> None:
    fake = FakeArgon(*[ConnectionError("offline")] * (OFFLINE_REOBSERVES + 1))
    policy = ArgonPolicy(None, Argon(budget=50, generate=fake))
    calls = [policy.act(obs(i), "task", "x") for i in range(OFFLINE_REOBSERVES + 1)]
    assert [c.skill for c in calls[:-1] if c] == ["reobserve"] * OFFLINE_REOBSERVES
    assert calls[-1] is None


def test_budget_stops_calls_at_the_cap_and_every_call_is_logged_with_latency(tmp_path: Path) -> None:
    fake = FakeArgon(*['{"skill": "reobserve"}'] * 5)
    argon = Argon(EventLog(tmp_path, "run"), budget=2, generate=fake)
    policy = ArgonPolicy(None, argon)
    for i in range(4):
        policy.act(obs(i), "task", "x")
    assert len(fake.prompts) == 2
    logged = events(tmp_path)
    calls = [e for e in logged if e["type"] == "argon_call"]
    assert len(calls) == 2 and all("latency_ms" in e["data"] and e["data"]["ok"] for e in calls)  # type: ignore[operator]
    assert any(e["type"] == "error" and "budget" in str(e["data"]["error"]) for e in logged)  # type: ignore[index]


def failed_episode() -> EpisodeManifest:
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
        episode_id="ep",
        run_id="run",
        seed=1001,
        split="dev",
        round=0,
        instruction="do the task",
        sim_version="fake",
        robot_config="fake",
        steps=[step],
        validation=Validation(task_success=False, physics_valid=True, reason="box not in bin"),
    )


def test_diagnose_sends_the_validator_reason_and_bounded_frames_and_parses_a_diagnosis() -> None:
    answer = {"hypothesis": "never reached", "evidence": "hands stay open", "confidence": 0.7, "lesson": "line up"}
    fake = FakeArgon(json.dumps(answer))
    diagnosis = diagnose(failed_episode(), [bytes([i]) for i in range(10)], Argon(budget=5, generate=fake))
    assert diagnosis == Diagnosis(**answer)
    prompt, *frames = fake.prompts[0]
    assert "box not in bin" in str(prompt)
    assert frames == [b"\x00", b"\x07", b"\x08", b"\x09"]


def test_diagnose_without_argon_reports_it_instead_of_guessing() -> None:
    diagnosis = diagnose(failed_episode(), [b"f"], Argon(budget=0, generate=FakeArgon()))
    assert diagnosis.confidence == 0.0 and "unavailable" in diagnosis.hypothesis


def test_choose_scenes_never_returns_a_heldout_seed() -> None:
    sealed = min(heldout_seeds())
    assert sealed not in choose_scenes([], 30, start=sealed)


def test_assert_not_heldout_still_guards_choose_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    import isaacs_mom.teacher as teacher

    monkeypatch.setattr(teacher, "heldout_seeds", lambda: frozenset())  # a bug that forgets the filter
    with pytest.raises(SealedSeedError):
        teacher.choose_scenes([], 1, start=min(heldout_seeds()))


def test_the_teacher_never_imports_the_simulator() -> None:
    folder = Path(__file__).resolve().parent.parent / "isaacs_mom" / "teacher"
    for path in folder.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            names += [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith("isaacs_mom.sim") for n in names), path


def test_the_key_is_read_from_dotenv_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY='abc'\n")
    assert env_value("GEMINI_API_KEY", env) == "abc"
    assert env_value("GEMINI_API_KEY", tmp_path / "missing") is None
