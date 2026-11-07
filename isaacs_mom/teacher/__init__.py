"""Workstream 2: the Argon teacher (SPEC §3.3).

Argon sees only head-camera JPEGs, the instruction, the skill history and the validator's reason, never simulator
ground truth (rule 1), so nothing here imports `isaacs_mom.sim`.
"""

import json

from pydantic import ValidationError

from isaacs_mom.contracts import Controller, Diagnosis, EpisodeManifest, Observation, SkillCall
from isaacs_mom.seeds import DEV_SEED_START, assert_not_heldout, heldout_seeds
from isaacs_mom.teacher.argon import Argon

OFFLINE_REOBSERVES = 3  # holds this many times when Argon is unreachable or over budget, then gives up (SPEC §3.3)
DIAGNOSIS_FRAMES = 4  # the first frame and the last three: the start and how it went wrong, at a bounded cost

SKILL_SCHEMA = json.dumps(SkillCall.model_json_schema())

ROBOT = """You control a simulated Unitree G1 robot. Your only sensor is the head RGB camera image you are given.
You act by choosing one whitelisted skill at a time; a controller turns it into joint motion.
Skills are refused with a reason if they would go out of reach; the next feedback tells you."""


def diagnose(failed: EpisodeManifest, frames: list[bytes], argon: Argon | None = None) -> Diagnosis:
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

Diagnose the most likely cause of the failure from this evidence only. Answer with JSON:
{{"hypothesis": str, "evidence": str, "confidence": number 0-1, "lesson": str}}
`lesson` is one sentence the teacher should apply when demonstrating this task."""
    shown = frames if len(frames) <= DIAGNOSIS_FRAMES else [frames[0], *frames[-(DIAGNOSIS_FRAMES - 1) :]]
    answer = (argon or Argon()).ask("diagnose", [prompt, *shown])
    if answer is None:
        return Diagnosis(hypothesis="not diagnosed: Argon unavailable", evidence=reason, confidence=0.0, lesson="")
    try:
        return Diagnosis.model_validate_json(answer)
    except ValidationError as e:
        return Diagnosis(
            hypothesis="not diagnosed: unparseable Argon answer",
            evidence=f"{reason}; {e.error_count()} validation errors",
            confidence=0.0,
            lesson="",
        )


def choose_scenes(diagnoses: list[Diagnosis], n: int, start: int = DEV_SEED_START) -> list[int]:
    """Development seeds for the next demonstrations. A seed's layout is ground truth the teacher may not see
    (rule 1), so for now these are the next n dev seeds; each diagnosis still shapes how they are demonstrated."""
    seeds = [s for s in range(start, start + n + len(heldout_seeds())) if s not in heldout_seeds()][:n]
    for seed in seeds:
        assert_not_heldout(seed)
    return seeds


class ArgonPolicy:
    """Drives the robot skill by skill, deciding afresh from every new frame, so a mid-run change alters the next
    decision."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, diagnosis: Diagnosis | None, argon: Argon | None = None) -> None:
        self.diagnosis = diagnosis
        self.argon = argon or Argon()
        self.history: list[str] = []
        self.offline = 0

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if self.history:
            self.history[-1] += f" -> {feedback}"
        answer = self.argon.ask("act", [self._prompt(obs, instruction), obs.head_rgb_jpeg])
        if answer is None:
            self.offline += 1
            if self.offline > OFFLINE_REOBSERVES:
                return None
            return self._record(SkillCall(skill="reobserve", reason="Argon unavailable: holding and re-observing"))
        self.offline = 0
        try:
            raw = json.loads(answer)
            if isinstance(raw, dict) and raw.get("done") is True:
                return None
            call = SkillCall.model_validate(raw)
        except (json.JSONDecodeError, ValidationError):
            call = SkillCall(skill="reobserve", reason="unparseable or non-whitelisted Argon answer")
        return self._record(call)

    def _record(self, call: SkillCall) -> SkillCall:
        self.history.append(call.model_dump_json(exclude={"reason"}))
        return call

    def _prompt(self, obs: Observation, instruction: str) -> str:
        lesson = (
            f"\nA diagnosis of an earlier student failure: {self.diagnosis.hypothesis}. Lesson: {self.diagnosis.lesson}"
            if self.diagnosis and self.diagnosis.lesson
            else ""
        )
        return f"""{ROBOT}

Task: {instruction}{lesson}
Skills you have used so far, each with its result:
{chr(10).join(self.history) or "(none yet)"}
This is step {obs.step}. Look at the new camera image: if the scene changed, re-plan from what you see now.

Answer with exactly one JSON object. Either the next skill, matching this schema, with a short `reason` saying
what in the image led to it:
{SKILL_SCHEMA}
or, only if the image shows the task is complete: {{"done": true}}"""
