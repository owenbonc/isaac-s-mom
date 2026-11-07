"""Workstream 2: the Argon teacher (SPEC §3.3).

Argon sees only head-camera JPEGs, the instruction, the skill history and the validator's reason, never simulator
ground truth (rule 1), so nothing here imports `isaacs_mom.sim`.

Done rule: a demo ends with exactly one `reobserve`, then None, and the student reads two `reobserve`s in a row as
done. So the teacher never records a second `reobserve` in a row before the end: an unreachable Argon or an unusable
answer is asked again on the same frame after a wait, and neither records a step.
"""

import json
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ValidationError

from isaacs_mom.contracts import Controller, Diagnosis, EpisodeManifest, Observation, SkillCall
from isaacs_mom.seeds import DEV_SEED_START, assert_not_heldout, heldout_seeds
from isaacs_mom.teacher.argon import Argon, Part, Purpose

ATTEMPTS = 3  # per step, on the same frame, waiting between them (SPEC §3.3); then the demo ends
DIAGNOSIS_FRAMES = 4  # the first frame and the last three: the start and how it went wrong, at a bounded cost
RAW_LOG_CHARS = 500

SKILL_SCHEMA = json.dumps(SkillCall.model_json_schema())

ROBOT = """You control a simulated Unitree G1 robot. Your only sensor is the head RGB camera image you are given.
You act by choosing one whitelisted skill at a time; a controller turns it into joint motion.
Skills are refused with a reason if they would go out of reach; the next feedback tells you."""

Done = Literal["done"]
OverBudget = Literal["over budget"]


def _json(answer: str) -> object:
    """Parsed JSON, or None. ValueError covers a 4,300+-digit integer as well as malformed text."""
    try:
        return json.loads(answer)
    except (ValueError, RecursionError):
        return None


def _strict(model: type[BaseModel], answer: str) -> BaseModel | None:
    """The answer as `model`, or None if it is not exactly that object: no extra keys, no "7" for 7."""
    raw = _json(answer)
    if not isinstance(raw, dict) or set(raw) - set(model.model_fields):
        return None
    try:
        return model.model_validate(raw, strict=True)
    except ValidationError:
        return None


def parse_step(answer: str) -> SkillCall | Done | None:
    """A skill call, "done", or None for anything else (including "done" alongside a skill)."""
    raw = _json(answer)
    if isinstance(raw, dict) and "done" in raw:
        ok = raw["done"] is True and set(raw) <= {"done", "reason"} and isinstance(raw.get("reason", ""), str)
        return "done" if ok else None
    call = _strict(SkillCall, answer)
    return call if isinstance(call, SkillCall) else None


def ask_parsed[T](
    argon: Argon, purpose: Purpose, parts: list[Part], parse: Callable[[str], T | None]
) -> T | OverBudget | None:
    """Up to ATTEMPTS calls on the same input, waiting between them: the parsed answer, "over budget", or None when
    Argon stayed unreachable or unusable."""
    for attempt in range(ATTEMPTS):
        if argon.exhausted:
            return "over budget"
        answer = argon.ask(purpose, parts)
        if answer is not None:
            parsed = parse(answer)
            if parsed is not None:
                return parsed
            argon.error(purpose, "unparseable or non-whitelisted Argon answer", raw=answer[:RAW_LOG_CHARS])
        if attempt < ATTEMPTS - 1 and not argon.exhausted:  # over budget ends at once, without a last wait
            argon.wait(attempt)
    return "over budget" if argon.exhausted else None  # the last attempt may have spent the budget


def _diagnosis(answer: str) -> Diagnosis | None:
    parsed = _strict(Diagnosis, answer)
    return parsed if isinstance(parsed, Diagnosis) else None


def diagnose(failed: EpisodeManifest, frames: list[bytes], argon: Argon) -> Diagnosis:
    """From a failed episode's frames, skill calls and validator reason, a failure hypothesis."""
    reason = failed.validation.reason if failed.validation else "no validation"
    steps = [
        f"step {s.step} [{s.controller}] {s.skill_call.model_dump_json(exclude={'reason'})} -> "
        + ("ok" if s.ok else f"refused: {s.error}")
        for s in failed.steps
    ]
    prompt = f"""{ROBOT}

A student policy failed this task: {failed.instruction}
Its skill calls, in order:
{chr(10).join(steps) or "(none)"}
The independent validator's verdict: {reason}
The images are head-camera frames from the start and the end of the episode.

Diagnose the most likely cause of the failure from this evidence only. Answer with exactly this JSON object:
{{"hypothesis": str, "evidence": str, "confidence": number 0-1, "lesson": str}}
`lesson` is one sentence the teacher should apply when demonstrating this task."""
    shown = frames if len(frames) <= DIAGNOSIS_FRAMES else [frames[0], *frames[-(DIAGNOSIS_FRAMES - 1) :]]
    diagnosis = ask_parsed(argon, "diagnose", [prompt, *shown], _diagnosis)
    if isinstance(diagnosis, Diagnosis):
        return diagnosis
    why = "Argon over budget" if diagnosis == "over budget" else "Argon unavailable or no usable answer"
    return Diagnosis(hypothesis=f"not diagnosed: {why}", evidence=reason, confidence=0.0, lesson="")


def choose_scenes(failures: list[EpisodeManifest], n: int, start: int = DEV_SEED_START) -> list[int]:
    """The dev seeds whose trials the validator rejected, in order, then fresh dev seeds up to n. Pass or fail is
    validator output, so no layout or ground truth is read (rule 1)."""
    rejected = [m.seed for m in failures if m.split == "dev" and m.validation and not m.validation.accepted]
    seeds = list(dict.fromkeys(s for s in rejected if s not in heldout_seeds()))[:n]
    fresh = (s for s in range(start, start + 10 * n + len(heldout_seeds())) if s not in heldout_seeds())
    seeds += [s for s in fresh if s not in seeds][: n - len(seeds)]
    for seed in seeds:
        assert_not_heldout(seed)
    return seeds


class ArgonPolicy:
    """Drives the robot skill by skill, deciding afresh from every new frame, so a mid-run change alters the next
    decision."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, diagnosis: Diagnosis | None, argon: Argon) -> None:
        self.diagnosis = diagnosis
        self.argon = argon
        self.history: list[SkillCall] = []
        self.results: list[str] = []
        self.finished = False

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if self.history:
            self.results.append(feedback)
        if self.finished:
            return None
        step = ask_parsed(self.argon, "act", [self._prompt(obs, instruction), obs.head_rgb_jpeg], parse_step)
        if step == "over budget":
            return self._finish("ended: Argon call budget used up")
        if step is None:
            return self._finish("ended: Argon unreachable or no usable answer")
        if step == "done":
            return self._finish("Argon: the task looks complete")
        if step.skill == "reobserve" and self._last_was_reobserve():
            return self._finish(step.reason)  # a second look in a row would read as "done" to the student
        self.history.append(step)
        return step

    def _last_was_reobserve(self) -> bool:
        return bool(self.history) and self.history[-1].skill == "reobserve"

    def _finish(self, reason: str) -> SkillCall | None:
        """The one closing `reobserve` of the done rule, unless the last step already was one."""
        self.finished = True
        if self._last_was_reobserve():
            return None
        call = SkillCall(skill="reobserve", reason=reason or "done")
        self.history.append(call)
        return call

    def _prompt(self, obs: Observation, instruction: str) -> str:
        lesson = (
            f"\nA diagnosis of an earlier student failure: {self.diagnosis.hypothesis}. Lesson: {self.diagnosis.lesson}"
            if self.diagnosis and self.diagnosis.lesson
            else ""
        )
        done = [
            f"{c.model_dump_json(exclude={'reason'})} -> {r}" for c, r in zip(self.history, self.results, strict=False)
        ]
        return f"""{ROBOT}

Task: {instruction}{lesson}
Skills you have used so far, each with its result:
{chr(10).join(done) or "(none yet)"}
This is step {obs.step}. Look at the new camera image: if the scene changed, re-plan from what you see now.
Use `reobserve` only to look again after something may have moved, never twice in a row.

Answer with exactly one JSON object and no other keys. Either the next skill, matching this schema, with a short
`reason` saying what in the image led to it:
{SKILL_SCHEMA}
or, only if the image shows the task is complete: {{"done": true, "reason": str}}"""
