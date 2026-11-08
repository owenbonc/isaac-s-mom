"""Learn a task from one sentence, end to end (SPEC §3.7): instruction -> checked TaskSpec -> Argon demonstrations
in a parallel batch (§3.6) -> one fine-tune on the accepted ones -> the fine-tuned student alone on the held-out seeds.

    uv run python -m isaacs_mom.tasks.learn "deliver the box to the bin" --demos 40 --workers 8

For a generated task `tasks.judge` is the validator of rule 3: each demonstration's manifest carries the spec's
verdict, so the project's training set keeps only the demonstrations the spec's rule accepts, and the held-out exam is
judged by the same rule. Every demonstration runs on a development seed (rule 2); each teacher process spends its own
`ARGON_CALL_BUDGET` (rule 5).
"""

import argparse
import json
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from isaacs_mom.contracts import Event, TrainingSetSummary
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.policy import Policy
from isaacs_mom.seeds import DEV_SEED_START, assert_not_heldout, heldout_seeds
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.student import BASE_CHECKPOINT, StudentPolicy, finetune, has_cuda
from isaacs_mom.tasks.generate import Propose, argon_proposer, generate_task
from isaacs_mom.tasks.rules import judge
from isaacs_mom.tasks.spec import Refusal, TaskSpec
from isaacs_mom.teacher import ArgonPolicy
from isaacs_mom.teacher.argon import Argon
from isaacs_mom.validator.recorder import write_atomically, write_manifest
from isaacs_mom.validator.training import build_training_set

DEMO_SEED_START = DEV_SEED_START + 5000  # clear of the loop's trials and choices and of the readiness trial's seeds
ROUND = 1  # a batch fine-tunes once
MakePolicy = Callable[[MujocoSim], Policy]
Train = Callable[[Path, str, int], str]  # (training set, previous checkpoint, round) -> new checkpoint


class LearnReport(BaseModel):
    """One batch-taught task, reported apart from `mom run` rounds (SPEC §3.6)."""

    taught: Literal["batch"] = "batch"
    run_id: str
    spec: TaskSpec
    fingerprint: str
    demos: int
    accepted: int
    examples: int
    argon_calls: int  # every teacher process's, failed calls included
    checkpoint: str
    finetune_metrics: dict[str, object]
    exam_seeds: list[int]
    exam: dict[str, dict[str, int]]  # checkpoint -> {"successes", "errored", "n"}


def teacher_instruction(spec: TaskSpec) -> str:
    """The person's words, plus the one fact the teacher's skill guide needs: its feedback calls the destination
    the bin, whatever the spec calls it."""
    kind = spec.scene.destination.kind
    if kind == "bin":
        return spec.instruction
    return f"{spec.instruction} (the {kind} is the grey stand with a green top; feedback calls it the bin)"


def demonstrate(spec_json: str, seeds: Sequence[int], out: str, make_teacher: MakePolicy | None = None) -> None:
    """One teacher process: a demonstration on each seed, its manifest's verdict the spec's. Argon by default."""
    spec = TaskSpec.model_validate_json(spec_json)
    out_dir = Path(out)
    sim = MujocoSim(spec.task_config())
    events = EventLog(out_dir, out_dir.name)
    if make_teacher is None:
        argon = Argon(events)
        make_teacher = lambda _sim: ArgonPolicy(None, argon)  # noqa: E731
    for seed in seeds:
        assert_not_heldout(seed)
        manifest = run_episode(
            sim,
            make_teacher(sim),
            run_dir=out_dir,
            events=events,
            seed=seed,
            split="dev",
            round_=ROUND,
            instruction=teacher_instruction(spec),
        )
        manifest.validation = judge(spec, sim.ground_truth())
        write_manifest(out_dir / "episodes" / manifest.episode_id / "manifest.json", manifest)


def collect(spec: TaskSpec, run_dir: Path, demos: int, workers: int, make_teacher: MakePolicy | None = None) -> None:
    """`demos` demonstrations over `workers` teacher processes (this one when 1), gathered into run_dir/episodes.
    Across processes `make_teacher` must be picklable: a top-level function or class."""
    seeds = list(range(DEMO_SEED_START, DEMO_SEED_START + demos))
    shares = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    outs = [str(run_dir / "workers" / f"w{i}") for i in range(len(shares))]
    spec_json = spec.model_dump_json()
    if len(shares) == 1:
        demonstrate(spec_json, shares[0], outs[0], make_teacher)
    else:
        with ProcessPoolExecutor(len(shares), mp_context=get_context("spawn")) as pool:
            # Drained, so a worker's exception raises here instead of vanishing with its unread result.
            list(pool.map(demonstrate, [spec_json] * len(shares), shares, outs, [make_teacher] * len(shares)))
    episodes = run_dir / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    for out in outs:
        for episode in sorted((Path(out) / "episodes").glob("*")):
            shutil.move(str(episode), episodes / episode.name)


def argon_calls(run_dir: Path) -> int:
    """The batch's Argon calls: every `argon_call` event in every teacher process's own log."""
    return sum(
        Event.model_validate_json(line).type == "argon_call"
        for log in sorted((run_dir / "workers").glob("*/events.jsonl"))
        for line in log.read_text().splitlines()
    )


def finetune_metrics(run_dir: Path) -> dict[str, object]:
    """What the fine-tune measured, or nothing when it trained no checkpoint."""
    path = run_dir / "checkpoints" / f"ckpt-r{ROUND}" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def exam(
    spec: TaskSpec, run_dir: Path, checkpoint: str, seeds: Sequence[int], make_student: Callable[[str], Policy]
) -> dict[str, int]:
    """The student alone on the held-out seeds, each judged by the spec's rule; a crash counts as errored (§3.5)."""
    sim = MujocoSim(spec.task_config())
    out = run_dir / "exam" / checkpoint.replace("/", "_")
    events = EventLog(out, out.name)
    successes = errored = 0
    for seed in seeds:
        try:
            run_episode(
                sim,
                make_student(checkpoint),
                run_dir=out,
                events=events,
                seed=seed,
                split="heldout",
                round_=ROUND,
                instruction=teacher_instruction(spec),
            )
        except Exception as error:  # the system's fault, not the student's: reported apart
            errored += 1
            events.emit("error", seed=seed, split="heldout", error=f"{type(error).__name__}: {error}")
            continue
        successes += judge(spec, sim.ground_truth()).accepted
    return {"successes": successes, "errored": errored, "n": len(seeds)}


def learn(
    instruction: str,
    propose: Propose,
    *,
    runs_dir: Path,
    demos: int,
    workers: int,
    exam_seeds: int,
    train: Train,
    make_student: Callable[[str], Policy],
    base: str | None = None,
    make_teacher: MakePolicy | None = None,
) -> LearnReport | Refusal:
    spec = generate_task(instruction, propose)
    if isinstance(spec, Refusal):
        return spec
    run_id = f"learn-{spec.name}-{spec.fingerprint()}-{time.strftime('%H%M%S')}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    write_atomically(run_dir / "spec.json", spec.model_dump_json(indent=2))
    collect(spec, run_dir, demos, workers, make_teacher)
    training_set = build_training_set(run_dir, ROUND)
    summary = TrainingSetSummary.model_validate_json(training_set.with_suffix(".summary.json").read_text())
    checkpoint = train(training_set, base or BASE_CHECKPOINT, ROUND)
    seeds = sorted(heldout_seeds())[:exam_seeds]
    report = LearnReport(
        run_id=run_id,
        spec=spec,
        fingerprint=spec.fingerprint(),
        demos=summary.demonstrations,
        accepted=summary.accepted,
        examples=summary.examples,
        argon_calls=argon_calls(run_dir),
        checkpoint=checkpoint,
        finetune_metrics=finetune_metrics(run_dir),
        exam_seeds=seeds,
        exam={c: exam(spec, run_dir, c, seeds, make_student) for c in (base, checkpoint) if c is not None},
    )
    write_atomically(run_dir / "report.json", report.model_dump_json(indent=2))
    return report


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a task from one sentence, end to end (SPEC §3.7)")
    parser.add_argument("instruction")
    parser.add_argument("--demos", type=_positive, default=40)
    parser.add_argument("--workers", type=_positive, default=8, help="teacher processes, each with its own budget")
    parser.add_argument("--exam-seeds", type=int, default=20)
    parser.add_argument("--with-base", action="store_true", help="also examine the base model on the same seeds")
    args = parser.parse_args()
    if not has_cuda():
        parser.error("fine-tuning needs CUDA: run this on the RTX 5070 Ti")
    runs = Path("runs")
    argon = Argon(EventLog(runs / "learn-generator", "learn-generator"))
    argon.check()
    report = learn(
        args.instruction,
        argon_proposer(argon),
        runs_dir=runs,
        demos=args.demos,
        workers=args.workers,
        exam_seeds=args.exam_seeds,
        train=finetune,
        make_student=StudentPolicy,
        base=BASE_CHECKPOINT if args.with_base else None,
    )
    if isinstance(report, Refusal):
        print("refused:\n  " + "\n  ".join(report.reasons))
        raise SystemExit(1)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
