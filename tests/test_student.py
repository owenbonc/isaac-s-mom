import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from isaacs_mom.contracts import SKILLS, Observation, SkillCall, SkillTarget
from isaacs_mom.student.checkpoints import ADAPTER_FILES, BASE_CHECKPOINT, resolve_adapter
from isaacs_mom.student.policy import StudentPolicy
from isaacs_mom.student.prompt import UnparseableReply, format_target, parse_reply, prompt_text

PUSH_5 = SkillCall(skill="push_forward", params={"distance_cm": 5})


@pytest.mark.parametrize(
    "text",
    [
        '{"skill": "push_forward", "params": {"distance_cm": 5}}',
        '```json\n{"skill": "push_forward", "params": {"distance_cm": 5}}\n```',
        'Plan {not json} then {"skill": "push_forward", "params": {"distance_cm": 5}} since the block is ahead.',
    ],
)
def test_a_reply_containing_a_skill_call_parses(text: str) -> None:
    assert parse_reply(text) == PUSH_5


@pytest.mark.parametrize(
    "text",
    [
        "",
        "push forward a bit",
        '{"skill": "fly"}',
        '{"skill": "push_forward", "params": {"torque": 1}}',
        '{"skill": ',
        "[1, 2]",
    ],
)
def test_a_reply_without_a_whitelisted_skill_call_is_unparseable(text: str) -> None:
    with pytest.raises(UnparseableReply):
        parse_reply(text)


def test_reply_distance_is_clamped_to_the_skill_limit() -> None:
    assert parse_reply('{"skill": "push_forward", "params": {"distance_cm": 90}}').params == {"distance_cm": 20.0}


def test_training_target_round_trips_through_the_parser() -> None:
    call = SkillCall(skill="approach_left", params={"distance_cm": 7.26})
    assert parse_reply(format_target(call)) == SkillCall(skill="approach_left", params={"distance_cm": 7.3})


def test_training_target_has_no_reason() -> None:
    assert format_target(SkillTarget(skill="retract")) == '{"skill":"retract","params":{}}'


def test_prompt_names_the_task_the_last_result_and_every_skill() -> None:
    text = prompt_text("Push the red block into the green target zone.", "push_forward: ok")
    assert "Push the red block into the green target zone." in text
    assert "push_forward: ok" in text
    assert all(skill in text for skill in SKILLS)


def test_base_checkpoint_is_the_plain_model(tmp_path: Path) -> None:
    assert resolve_adapter(BASE_CHECKPOINT, tmp_path) is None


def test_fine_tuned_checkpoint_needs_a_complete_adapter(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ckpt-r1"):
        resolve_adapter("ckpt-r1", tmp_path)
    (tmp_path / "ckpt-r1").mkdir()
    (tmp_path / "ckpt-r1" / ADAPTER_FILES[0]).write_text("{}")
    with pytest.raises(FileNotFoundError, match=ADAPTER_FILES[1]):
        resolve_adapter("ckpt-r1", tmp_path)
    (tmp_path / "ckpt-r1" / ADAPTER_FILES[1]).write_bytes(b"weights")
    assert resolve_adapter("ckpt-r1", tmp_path) == tmp_path / "ckpt-r1"


REOBSERVE = '{"skill": "reobserve", "params": {}}'


def frame(step: int = 0, episode_id: str = "ep-1") -> Observation:
    return Observation(episode_id=episode_id, step=step, sim_time=0.0, head_rgb_jpeg=b"jpeg", joint_state=[])


def scripted(*replies: str) -> Callable[[bytes, str, str], str]:
    queue = list(replies)
    return lambda jpeg, instruction, feedback: queue.pop(0)


def test_reply_becomes_the_skill_call_and_its_latency_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted('{"skill": "approach_left", "params": {"distance_cm": 4}}'))
    with caplog.at_level(logging.INFO, logger="isaacs_mom.student"):
        call = policy.act(frame(), "push", "start")
    assert call == SkillCall(skill="approach_left", params={"distance_cm": 4})
    assert policy.last_latency_ms >= 0
    assert " ms" in caplog.text


def test_unparseable_reply_reobserves_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = StudentPolicy(generate=scripted("I would push the block"))
    with caplog.at_level(logging.WARNING, logger="isaacs_mom.student"):
        assert policy.act(frame(), "push", "start") == SkillCall(skill="reobserve")
    assert "unparseable" in caplog.text


def test_second_reobserve_in_a_row_means_done() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, REOBSERVE))
    assert policy.act(frame(0), "push", "start") == SkillCall(skill="reobserve")
    assert policy.act(frame(1), "push", "reobserve: ok") is None


def test_unparseable_replies_never_end_the_episode() -> None:
    policy = StudentPolicy(generate=scripted("??", "??", "??"))
    assert all(policy.act(frame(i), "push", "start") == SkillCall(skill="reobserve") for i in range(3))


def test_reobserve_streak_starts_over_in_a_new_episode() -> None:
    policy = StudentPolicy(generate=scripted(REOBSERVE, REOBSERVE))
    policy.act(frame(0, "ep-1"), "push", "start")
    assert policy.act(frame(0, "ep-2"), "push", "start") == SkillCall(skill="reobserve")


def test_missing_checkpoint_fails_before_any_model_call(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ckpt-r2"):
        StudentPolicy("ckpt-r2", checkpoint_dir=tmp_path, generate=scripted())
