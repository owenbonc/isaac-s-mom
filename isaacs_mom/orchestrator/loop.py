"""The teach -> fine-tune loop (SPEC §3.6)."""

import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from isaacs_mom.contracts import Diagnosis, EpisodeManifest, EvaluationResult
from isaacs_mom.orchestrator.episode import HEARTBEAT_SECONDS, heartbeat, run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.seeds import DEV_SEED_START, heldout_seeds
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.student import BASE_CHECKPOINT, StudentPolicy, adapter_dir, finetune
from isaacs_mom.teacher import ArgonPolicy, choose_scenes, diagnose
from isaacs_mom.teacher.argon import Argon
from isaacs_mom.validator.rules import RECORDING_ERROR
from isaacs_mom.validator.training import build_training_set

INSTRUCTION = "Hug the orange box, lift it, and put it in the green bin."
MAX_INSTRUCTION_CHARS = 200  # the Teach box's limit (SPEC §3.5)
TARGET_RATE = 0.8
MAX_ROUNDS = 3
DEMOS_PER_ROUND = 20
DEV_TRIALS = 10
MOVE_BIN_SHARE = 0.25  # of each round's teacher demonstrations, the bin moves right after the lift (§3.5)

StudentFactory = Callable[[str], Policy]
StudentTrainer = Callable[[Path, str, int], str]


def instruction_text(text: str) -> str:
    """A run's instruction, stripped: 1 to 200 characters and no NUL, as it travels as one argument of `mom run`."""
    text = text.strip()
    if not 1 <= len(text) <= MAX_INSTRUCTION_CHARS:
        raise ValueError(f"an instruction is 1 to {MAX_INSTRUCTION_CHARS} characters, not {len(text)}")
    if "\0" in text:
        raise ValueError("an instruction can't hold a NUL character")
    return text


def evaluate(
    sim: TruthSim,
    checkpoint: str,
    round_: int,
    run_dir: Path,
    events: EventLog,
    make_student: StudentFactory = StudentPolicy,
    instruction: str = INSTRUCTION,
) -> EvaluationResult:
    seeds = sorted(heldout_seeds())
    successes = errored = 0
    for seed in seeds:
        student = make_student(checkpoint)  # outside the try: a checkpoint that can't load stops the run
        try:
            manifest = run_episode(
                sim,
                student,
                run_dir=run_dir,
                events=events,
                seed=seed,
                split="heldout",
                round_=round_,
                instruction=instruction,
            )
        except Exception as error:  # a crash is the system's fault, not the student's, so it's counted apart (§3.5)
            errored += 1
            events.emit("error", seed=seed, split="heldout", round=round_, error=f"{type(error).__name__}: {error}")
            continue
        verdict = manifest.validation
        if verdict is None or verdict.reason.startswith(RECORDING_ERROR):
            errored += 1
        elif verdict.accepted:
            successes += 1
    result = EvaluationResult(checkpoint=checkpoint, round=round_, successes=successes, n=len(seeds), errored=errored)
    events.emit("evaluation_result", **result.model_dump(), rate=result.rate)
    return result


def moves_bin(demos: int, share: float) -> set[int]:
    """Which of a round's demonstrations get the lift-triggered bin move: round(share * demos) of them, rounded half
    to even (0.25 of 10 is 2, of 2 is none), spread evenly, so 0.25 of 20 is demonstrations 3, 7, 11, 15 and 19."""
    k = round(share * demos)
    return {i for i in range(demos) if (i + 1) * k // demos > i * k // demos}


def bin_moves(events_path: Path) -> str:
    """How many teacher demonstrations the bin really moved in, from the run's events: a scheduled move applies only
    once the box is lifted, and the sim may refuse it."""
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []

    def episodes(kind: str, holds: Callable[[dict[str, Any]], bool]) -> set[str]:
        return {e["data"]["episode_id"] for e in events if e["type"] == kind and holds(e["data"])}

    taught = episodes("decision", lambda d: d["controller"] == "teacher")
    scheduled = episodes("episode_started", lambda d: bool(d.get("move_bin_after_lift")))
    moved = episodes("perturbation", lambda d: d["kind"] == "move_bin")
    return (
        f"the bin moved right after the lift in {len(moved & taught)} of {len(taught)} teacher demonstrations "
        f"({len(scheduled & taught)} scheduled)"
    )


def run_loop(
    sim: TruthSim,
    run_dir: Path,
    run_id: str,
    *,
    max_rounds: int = MAX_ROUNDS,
    demos: int = DEMOS_PER_ROUND,
    dev_trials: int = DEV_TRIALS,
    argon: Argon,
    make_student: StudentFactory | None = None,
    train_student: StudentTrainer = finetune,
    instruction: str = INSTRUCTION,
    move_bin_share: float = MOVE_BIN_SHARE,
) -> list[EvaluationResult]:
    if not 0 <= move_bin_share <= 1:
        raise ValueError(f"move_bin_share is a share of the demonstrations, not {move_bin_share}")
    # finetune writes checkpoints beside the run dir, so the default student looks there too.
    make_student = make_student or partial(StudentPolicy, runs_dir=run_dir.parent)
    events = EventLog(run_dir, run_id)
    checkpoint = BASE_CHECKPOINT
    results: list[EvaluationResult] = []
    for round_ in range(max_rounds + 1):
        results.append(evaluate(sim, checkpoint, round_, run_dir, events, make_student, instruction))
        if results[-1].rate >= TARGET_RATE or round_ == max_rounds:
            break
        trial_start = DEV_SEED_START + 1000 * round_
        failures: list[EpisodeManifest] = []
        for seed in range(trial_start, trial_start + dev_trials):
            manifest = run_episode(
                sim,
                make_student(checkpoint),
                run_dir=run_dir,
                events=events,
                seed=seed,
                split="dev",
                round_=round_,
                instruction=instruction,
            )
            verdict = manifest.validation
            # A recording error is a simulator bug, never a lesson, so the teacher doesn't diagnose it.
            if verdict and not verdict.accepted and not verdict.reason.startswith(RECORDING_ERROR):
                failures.append(manifest)
        diagnosed: list[tuple[str, Diagnosis]] = []
        for manifest in failures:
            episode_dir = run_dir / "episodes" / manifest.episode_id
            with heartbeat(events, manifest.episode_id, HEARTBEAT_SECONDS):
                diagnosis = diagnose(manifest, [(episode_dir / s.frame).read_bytes() for s in manifest.steps], argon)
            events.emit("diagnosis", episode_id=manifest.episode_id, **diagnosis.model_dump())
            diagnosed.append((manifest.episode_id, diagnosis))
        lessons: list[tuple[str | None, Diagnosis | None]] = [*diagnosed] or [(None, None)]
        moved = moves_bin(demos, move_bin_share)
        # Rejected trials come back first, in the order of their diagnoses; fresh seeds cycle through the lessons.
        for i, seed in enumerate(choose_scenes(failures, demos, start=trial_start + 500)):
            failed_id, diagnosis = lessons[i % len(lessons)]
            run_episode(
                sim,
                ArgonPolicy(diagnosis, argon),
                run_dir=run_dir,
                events=events,
                seed=seed,
                split="dev",
                round_=round_,
                instruction=instruction,
                acting_on=failed_id,
                move_bin_after_lift=i in moved,
            )
        training_set = build_training_set(run_dir, round_ + 1)
        events.emit("finetune_started", round=round_ + 1, training_set=str(training_set))
        started = time.monotonic()
        previous, checkpoint = checkpoint, train_student(training_set, checkpoint, round_ + 1)
        duration_s = round(time.monotonic() - started, 1)
        skipped = checkpoint == previous
        path = None if skipped else adapter_dir(checkpoint, run_dir.parent)
        events.emit(
            "finetune_completed",
            round=round_ + 1,
            checkpoint=checkpoint,
            skipped=skipped,
            examples=len(training_set.read_text().splitlines()),
            duration_s=duration_s,
            checkpoint_path=str(path) if path else None,
        )
    return results
