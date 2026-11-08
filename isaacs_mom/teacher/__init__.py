"""Workstream 2: the Argon teacher (SPEC §3.3).

Argon sees only head-camera JPEGs, the instruction, the skill history and the validator's reason, never simulator
ground truth (rule 1), so nothing here imports `isaacs_mom.sim`.

Done rule: a demo ends with exactly one `reobserve`, then None, and the student reads two `reobserve`s in a row as
done. So the teacher never records a second `reobserve` in a row before the end: an unreachable Argon or an unusable
answer is asked again on the same frame after a wait, and neither records a step.
"""

import json
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from isaacs_mom.contracts import (
    MAX_ANGLE_DEG,
    MAX_DISTANCE_CM,
    Controller,
    Diagnosis,
    EpisodeManifest,
    Observation,
    SkillCall,
)
from isaacs_mom.seeds import DEV_SEED_START, assert_not_heldout, heldout_seeds
from isaacs_mom.teacher.argon import Argon, Part, Purpose

ATTEMPTS = 3  # per step, on the same frame, waiting between them (SPEC §3.3); then the demo ends
DIAGNOSIS_FRAMES = 4  # the first frame and the last three: the start and how it went wrong, at a bounded cost
RAW_LOG_CHARS = 500
ROUND_FRAMES = 8  # decision points across a whole round, at a bounded cost
ROUND_MODES = 3
EPISODE_ID = re.compile(r"ep-\d+-[0-9a-f]+")
EXPLAINS = re.compile(r" Explains: (ep-[^.]+)\.$")

SKILL_SCHEMA = json.dumps(SkillCall.model_json_schema())

# What the schema alone doesn't tell Argon: what each skill does, its parameters, and that up close the carried
# box hides the bin. Without it, Argon walked and turned only by the defaults and let go short of the bin.
ROBOT = f"""You control a simulated Unitree G1 humanoid. Your only sensor is the head RGB camera image you are
given. It looks down and ahead, so once the box is lifted it fills the lower middle of the image and hides the floor
just in front of you: when the bin is close enough to place into, the box hides it.
You act by choosing one skill at a time; a controller turns it into motion. The skills and their parameters:
- reach_left / reach_right {{"distance_cm": default 5}}: slide both open hands sideways, at most 12 cm off-centre
  in all, to line them up with the box on the table.
- hug: close both arms on the box between the hands. It only works on a box on the table: a box that falls to the
  floor can't be picked up again, and the task has failed.
- lift: raise the hugged box.
- walk_forward / walk_back {{"distance_cm": 0-{MAX_DISTANCE_CM:.0f}, default 30}}: walk straight ahead or back,
  the whole distance in one call. Refused when your feet stay put in this task.
- turn_left / turn_right {{"angle_deg": 0-{MAX_ANGLE_DEG:.0f}, default 30}}: turn the whole body in place by that
  angle, in one call. When your feet stay put, the waist twists instead: at most 45 degrees at a time and 150 in all.
- place: lower the box about 40 cm in front of your chest and let go. Only when the bin is straight ahead and right
  in front of you; anywhere else the box falls on the floor.
- reobserve: look again without moving.
A skill that would reach too far, or walk into the table or the bin, is refused with a reason and nothing moves.
A good plan: line the hands up with the box, hug and lift it. If you can walk: walk_back 30 cm from the table, then
turn to find the bin (it is beside or behind you, never ahead; turn 90 degrees at a time while it is out of view).
Line it up only while the whole bin shows above the box: turn until its centre is straight above the box's centre.
A bin at the box's top corner, partly behind it, is about 20 degrees to that side; one fully beside the box, about 40.
If it shows only beside the box or partly behind it, it is too close to line up: walk_back 50 cm and look again.
Judge its distance (from you to its centre) by where it sits on the lifted box: its grey stand showing with a gap above
the box, 1.6 m or more; the whole bin resting on the box's top edge, 1.3 m; only its green inside showing over the
edge, 1.1 m; only its far rim peeking over, 0.9 m; hidden behind the box, under 0.8 m. You place from 45 cm away, so
walk_forward the distance minus 45 cm in one call (if that is refused, try 10 cm less). Then walk_forward 10: if it is
refused because it would walk into the bin, you are right up to it, so place; if not, walk_forward 10 again.
If the bin shows beside you instead of ahead, you have walked past it: walk_back 50 cm and line it up again.
If your feet stay put, the bin is beside you within reach: twist toward it until it is straight ahead, then place."""

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


def _steps(episode: EpisodeManifest) -> list[str]:
    return [
        f"step {s.step} [{s.controller}] {s.skill_call.model_dump_json(exclude={'reason'})} -> "
        + ("ok" if s.ok else f"refused: {s.error}")
        for s in episode.steps
    ]


def diagnose(failed: EpisodeManifest, frames: list[bytes], argon: Argon) -> Diagnosis:
    """From a failed episode's frames, skill calls and validator reason, a failure hypothesis."""
    reason = failed.validation.reason if failed.validation else "no validation"
    steps = _steps(failed)
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


class _Mode(BaseModel):
    hypothesis: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    lesson: str
    explains: list[str] = Field(min_length=1)


class _Modes(BaseModel):
    modes: list[_Mode] = Field(min_length=1, max_length=ROUND_MODES)


def explained_by(mode: Diagnosis) -> list[str]:
    """The failed trials a failure mode from `diagnose_round` names, read back from the end of its evidence."""
    match = EXPLAINS.search(mode.evidence)
    return match.group(1).split(", ") if match else []


def _accepted(episode: EpisodeManifest) -> bool:
    return episode.validation is not None and episode.validation.accepted


def _decision_frames(trials: list[tuple[EpisodeManifest, list[bytes]]]) -> list[Part]:
    """Up to ROUND_FRAMES labelled frames where a trial decided (before its first refused call, and before `place`),
    failed trials first, one frame per trial in turn."""
    queues = []
    for episode, frames in sorted(trials, key=lambda trial: _accepted(trial[0])):
        points, refused = [], False
        for step, frame in zip(episode.steps, frames, strict=True):
            first_refusal = not step.ok and not refused
            refused = refused or not step.ok
            if first_refusal or step.skill_call.skill == "place":
                outcome = "ok" if step.ok else "refused"
                label = f"trial {episode.episode_id}, step {step.step}, before {step.skill_call.skill} ({outcome}):"
                points.append((label, frame))
        queues.append(points)
    parts: list[Part] = []
    while len(parts) < 2 * ROUND_FRAMES and any(queues):
        for queue in queues:
            if queue and len(parts) < 2 * ROUND_FRAMES:
                label, frame = queue.pop(0)
                parts += [label, frame]
    return parts


def _round_modes(answer: str, trial_ids: set[str], failed_ids: set[str]) -> list[Diagnosis] | None:
    parsed = _strict(_Modes, answer)
    if not isinstance(parsed, _Modes):
        return None
    modes = []
    for mode in parsed.modes:
        explains = list(dict.fromkeys(mode.explains))
        cited = set(EPISODE_ID.findall(f"{mode.hypothesis} {mode.evidence} {mode.lesson}"))
        if not set(explains) <= failed_ids or not cited <= trial_ids:  # only this round's trials, only its failures
            return None
        evidence = f"{mode.evidence} Explains: {', '.join(explains)}."
        modes.append(Diagnosis(hypothesis=mode.hypothesis, evidence=evidence, confidence=mode.confidence,
                               lesson=mode.lesson))  # fmt: skip
    return modes


def diagnose_round(trials: list[tuple[EpisodeManifest, list[bytes]]], argon: Argon) -> list[Diagnosis]:
    """The round's failure modes, from every student trial of it, failed and successful, in one Argon call: each
    mode names the failed trials it explains. Empty when Argon is unreachable, over budget or unusable, so the
    per-episode lessons stand."""
    failed = [episode for episode, _ in trials if not _accepted(episode)]
    if not failed:
        return []
    blocks = []
    for episode, _ in trials:
        outcome = "succeeded" if _accepted(episode) else "failed"
        verdict = episode.validation.reason if episode.validation else "no validation"
        lines = "\n".join(f"  {line}" for line in _steps(episode)) or "  (no skill calls)"
        blocks.append(f"trial {episode.episode_id} ({outcome}), the validator's verdict: {verdict}\n{lines}")
    prompt = f"""{ROBOT}

A student policy tried this task {len(trials)} times this round, and {len(failed)} of them failed.
The task: {trials[0][0].instruction}
Every trial's skill calls, in order, and the independent validator's verdict:
{chr(10).join(blocks)}
The images are moments where trials decided, each labelled with its trial, step and the call it made next.

Group the failures by cause: compare them with each other and with the trials that succeeded. Answer with exactly
this JSON object, 1 to {ROUND_MODES} modes, the one explaining the most failures first:
{{"modes": [{{"hypothesis": str, "evidence": str, "confidence": number 0-1, "lesson": str, "explains": [str]}}]}}
`explains` lists the ids of the failed trials the mode accounts for. Cite only the trials above. `lesson` is one
sentence the teacher should apply when demonstrating this task."""
    answer = argon.ask("diagnose", [prompt, *_decision_frames(trials)])  # one call per round (rule 5)
    if answer is None:
        return []
    ids = {episode.episode_id for episode, _ in trials}
    modes = _round_modes(answer, ids, {episode.episode_id for episode in failed})
    if modes is None:
        argon.error("diagnose", "unusable round diagnosis", raw=answer[:RAW_LOG_CHARS])
        return []
    return modes


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
