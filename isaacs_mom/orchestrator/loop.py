"""The teach -> fine-tune loop (SPEC §3.6)."""

from pathlib import Path

from isaacs_mom.contracts import EpisodeManifest, EvaluationResult
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.seeds import DEV_SEED_START, heldout_seeds
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.student import BASE_CHECKPOINT, StudentPolicy, finetune
from isaacs_mom.teacher import ArgonPolicy, choose_scenes, diagnose
from isaacs_mom.teacher.argon import Argon
from isaacs_mom.validator.training import build_training_set

INSTRUCTION = "Hug the orange box, lift it, and put it in the green bin."
TARGET_RATE = 0.8
MAX_ROUNDS = 3
DEMOS_PER_ROUND = 20
DEV_TRIALS = 10


def evaluate(sim: TruthSim, checkpoint: str, round_: int, run_dir: Path, events: EventLog) -> EvaluationResult:
    seeds = sorted(heldout_seeds())
    successes = sum(
        run_episode(
            sim,
            StudentPolicy(checkpoint),
            run_dir=run_dir,
            events=events,
            seed=seed,
            split="heldout",
            round_=round_,
            instruction=INSTRUCTION,
        ).validation.accepted  # type: ignore[union-attr]  # finish() always sets validation
        for seed in seeds
    )
    result = EvaluationResult(checkpoint=checkpoint, round=round_, successes=successes, n=len(seeds))
    events.emit("evaluation_result", **result.model_dump(), rate=result.rate)
    return result


def run_loop(
    sim: TruthSim,
    run_dir: Path,
    run_id: str,
    *,
    max_rounds: int = MAX_ROUNDS,
    demos: int = DEMOS_PER_ROUND,
    dev_trials: int = DEV_TRIALS,
) -> list[EvaluationResult]:
    events = EventLog(run_dir, run_id)
    argon = Argon(events)  # one budget for the whole run (SPEC §2 rule 5)
    checkpoint = BASE_CHECKPOINT
    results: list[EvaluationResult] = []
    for round_ in range(max_rounds + 1):
        results.append(evaluate(sim, checkpoint, round_, run_dir, events))
        if results[-1].rate >= TARGET_RATE or round_ == max_rounds:
            break
        trial_start = DEV_SEED_START + 1000 * round_
        failures: list[EpisodeManifest] = []
        for seed in range(trial_start, trial_start + dev_trials):
            manifest = run_episode(
                sim,
                StudentPolicy(checkpoint),
                run_dir=run_dir,
                events=events,
                seed=seed,
                split="dev",
                round_=round_,
                instruction=INSTRUCTION,
            )
            if manifest.validation and not manifest.validation.accepted:
                failures.append(manifest)
        diagnoses = []
        for manifest in failures:
            episode_dir = run_dir / "episodes" / manifest.episode_id
            diagnosis = diagnose(manifest, [(episode_dir / s.frame).read_bytes() for s in manifest.steps], argon)
            events.emit("diagnosis", episode_id=manifest.episode_id, **diagnosis.model_dump())
            diagnoses.append(diagnosis)
        for seed, diagnosis in zip(
            choose_scenes(diagnoses, demos, start=trial_start + 500),
            (diagnoses or [None]) * demos,
            strict=False,
        ):
            run_episode(
                sim,
                ArgonPolicy(diagnosis, argon),
                run_dir=run_dir,
                events=events,
                seed=seed,
                split="dev",
                round_=round_,
                instruction=INSTRUCTION,
            )
        training_set = build_training_set(run_dir, round_ + 1)
        events.emit("finetune_started", round=round_ + 1, training_set=str(training_set))
        checkpoint = finetune(training_set, checkpoint, round_ + 1)
        events.emit("finetune_completed", round=round_ + 1, checkpoint=checkpoint)
    return results
