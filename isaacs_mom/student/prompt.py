"""The student's prompt and reply format, shared by local inference and Modal training (SPEC §3.2)."""

import json

from pydantic import ValidationError

from isaacs_mom.contracts import MAX_DISTANCE_CM, SKILLS, SkillCall, SkillTarget


class UnparseableReply(ValueError):
    pass


def prompt_text(instruction: str, feedback: str) -> str:
    return (
        "You control a robot arm and see the table through its head camera.\n"
        f"Task: {instruction}\n"
        f"Result of your last skill: {feedback}\n"
        f"Choose the next skill from: {', '.join(SKILLS)}. distance_cm is optional, 0 to {MAX_DISTANCE_CM:g}.\n"
        'Answer with one JSON object only, for example {"skill": "push_forward", "params": {"distance_cm": 5}}. '
        'When the block is inside the target zone, answer {"skill": "reobserve", "params": {}}.'
    )


def build_messages(instruction: str, feedback: str) -> list[dict[str, object]]:
    text = prompt_text(instruction, feedback)
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def format_target(target: SkillTarget) -> str:
    """The reply the student is trained to give."""
    return SkillTarget(skill=target.skill, params={k: round(v, 1) for k, v in target.params.items()}).model_dump_json()


def parse_reply(text: str) -> SkillCall:
    """The first JSON object in the reply, checked against the skill whitelist."""
    decoder = json.JSONDecoder()
    for start in (i for i, char in enumerate(text) if char == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        try:
            return SkillCall.model_validate(value)
        except ValidationError as error:
            raise UnparseableReply(str(error)) from error
    raise UnparseableReply("no JSON object in reply")
