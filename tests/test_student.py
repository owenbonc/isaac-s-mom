from pathlib import Path

import pytest

from isaacs_mom.contracts import SKILLS, SkillCall
from isaacs_mom.student.checkpoints import ADAPTER_FILES, BASE_CHECKPOINT, resolve_adapter
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
