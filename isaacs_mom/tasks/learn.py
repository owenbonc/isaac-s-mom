"""Learn a task from one sentence, end to end (SPEC §3.7): instruction -> checked TaskSpec -> Argon demonstrations
in a parallel batch (§3.6) -> one fine-tune on the accepted ones -> the fine-tuned student alone on the held-out seeds.

    uv run python -m isaacs_mom.tasks.learn "deliver the box, even if someone moves the bin" --demos 40 --workers 8

The batch can also run on its own and be finished later, so the teachers can work while the GPU is busy:

    uv run python -m isaacs_mom.tasks.learn "<instruction>" --demos 40 --workers 8 --collect-only
    uv run python -m isaacs_mom.tasks.learn --resume runs/<run_id>

For a generated task `tasks.judge` is the validator of rule 3: each demonstration's manifest carries the spec's
verdict, so the project's training set keeps only the demonstrations the spec's rule accepts, every accepted one is
replayed against that same rule, and the held-out exam is judged by it. When the spec lets the bin move, the
lift-triggered `move_bin` runs in a stated share of the demonstrations (§3.5) and in a second held-out exam (§6.10).
Every demonstration runs on a development seed (rule 2); each teacher process spends its own `ARGON_CALL_BUDGET`
(rule 5).
"""

import argparse
import json
import shutil
import time
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from isaacs_mom.contracts import EpisodeManifest, Event, TrainingSetSummary
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.evaluation import adaptation_of
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import moves_bin
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
from isaacs_mom.validator.replay import replay
from isaacs_mom.validator.training import build_training_set

DEMO_SEED_START = DEV_SEED_START + 5000  # clear of the loop's trials and choices and of the readiness trial's seeds
ROUND = 1  # a batch fine-tunes once
MOVE_BIN_SHARE = 0.5  # of the demonstrations, when the spec lets the bin move: half show it found again
MakePolicy = Callable[[MujocoSim], Policy]
Train = Callable[[Path, str, int], str]  # (training set, previous checkpoint, round) -> new checkpoint


class Collected(BaseModel):
    """What the batch did, saved beside its episodes so a resumed run reports it."""

    demos: int
    workers: int
    move_bin_share: float  # 0 when the spec doesn't let the bin move
    moved_seeds: list[int]  # where the move was due; it applies only once the box is lifted
    generator_calls: int


class ExamResult(BaseModel):
    successes: int
    errored: int
    n: int
    # The bin-moved exam only: its scored episodes split three ways, and only an applied move tests adaptation.
    adaptation_tests: int | None = None
    adapted: int | None = None
    not_applied: int | None = None
    trigger_not_reached: int | None = None


class LearnReport(BaseModel):
    """One batch-taught task, reported apart from `mom run` rounds (SPEC §3.6)."""

    taught: Literal["batch"] = "batch"
    run_id: str
    spec: TaskSpec
    fingerprint: str
    demos: int
    demos_errored: int  # crashed before a verdict: logged, never trained on
    accepted: int
    examples: int
    move_bin_share: float
    demos_bin_moved: int  # demonstrations where the move applied
    accepted_bin_moved: int
    replayed: int  # accepted demonstrations that reproduce, by the spec's rule
    replay_failures: list[str]  # accepted demonstrations that don't
    argon_calls: int  # every teacher process's, failed calls included
    generator_calls: int
    checkpoint: str
    finetune_metrics: dict[str, object]
    exam_seeds: list[int]
    exam: dict[str, ExamResult]  # checkpoint -> result
    exam_bin_moved: dict[str, ExamResult]  # empty when the spec doesn't let the bin move


def teacher_instruction(spec: TaskSpec) -> str:
    """The person's words, plus the one fact the teacher's skill guide needs: its feedback calls the destination
    the bin, whatever the spec calls it."""
    kind = spec.scene.destination.kind
    if kind == "bin":
        return spec.instruction
    return f"{spec.instruction} (the {kind} is the grey stand with a green top; feedback calls it the bin)"


def demonstrate(
    spec_json: str,
    seeds: Sequence[int],
    out: str,
    make_teacher: MakePolicy | None = None,
    moved: Collection[int] = (),
) -> None:
    """One teacher process: a demonstration on each seed, its manifest's verdict the spec's, and each accepted one
    replayed against that rule. Argon by default. A demonstration that crashes is logged and the rest go on."""
    spec = TaskSpec.model_validate_json(spec_json)
    out_dir = Path(out)
    sim = MujocoSim(spec.task_config())
    events = EventLog(out_dir, out_dir.name)
    if make_teacher is None:
        argon = Argon(events)
        make_teacher = lambda _sim: ArgonPolicy(None, argon)  # noqa: E731
    for seed in seeds:
        assert_not_heldout(seed)
        try:
            manifest = run_episode(
                sim,
                make_teacher(sim),
                run_dir=out_dir,
                events=events,
                seed=seed,
                split="dev",
                round_=ROUND,
                instruction=teacher_instruction(spec),
                move_bin_after_lift=seed in moved,
            )
        except Exception as error:  # one lost demonstration mustn't cost the batch the others
            events.emit("error", seed=seed, split="dev", error=f"{type(error).__name__}: {error}")
            continue
        manifest.validation = judge(spec, sim.ground_truth())
        path = out_dir / "episodes" / manifest.episode_id / "manifest.json"
        write_manifest(path, manifest)
        if manifest.validation.accepted:
            replay(sim, path, verdict=lambda gt: judge(spec, gt))


def collect(
    spec: TaskSpec,
    run_dir: Path,
    demos: int,
    workers: int,
    make_teacher: MakePolicy | None = None,
    move_bin_share: float = MOVE_BIN_SHARE,
    generator_calls: int = 0,
) -> Collected:
    """`demos` demonstrations over `workers` teacher processes (this one when 1), gathered into run_dir/episodes.
    Across processes `make_teacher` must be picklable: a top-level function or class."""
    seeds = list(range(DEMO_SEED_START, DEMO_SEED_START + demos))
    share = move_bin_share if "move_bin" in spec.perturbation_kinds() else 0.0
    moved = sorted(seeds[i] for i in moves_bin(demos, share))
    shares = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    outs = [str(run_dir / "workers" / f"w{i}") for i in range(len(shares))]
    spec_json = spec.model_dump_json()
    if len(shares) == 1:
        demonstrate(spec_json, shares[0], outs[0], make_teacher, moved)
    else:
        n = len(shares)
        with ProcessPoolExecutor(n, mp_context=get_context("spawn")) as pool:
            # Drained, so a worker's exception raises here instead of vanishing with its unread result.
            list(pool.map(demonstrate, [spec_json] * n, shares, outs, [make_teacher] * n, [moved] * n))
    episodes = run_dir / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    for out in outs:
        for episode in sorted((Path(out) / "episodes").glob("*")):
            shutil.move(str(episode), episodes / episode.name)
    collected = Collected(
        demos=demos,
        workers=len(shares),
        move_bin_share=share,
        moved_seeds=moved,
        generator_calls=generator_calls,
    )
    write_atomically(run_dir / "collected.json", collected.model_dump_json(indent=2))
    return collected


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
    spec: TaskSpec,
    run_dir: Path,
    checkpoint: str,
    seeds: Sequence[int],
    make_student: Callable[[str], Policy],
    move_bin: bool = False,
) -> ExamResult:
    """The student alone on the held-out seeds, each judged by the spec's rule; a crash counts as errored (§3.5).
    With `move_bin`, the bin moves once right after the lift, and the scored episodes are split as §6.10 says."""
    sim = MujocoSim(spec.task_config())
    out = run_dir / ("exam-bin-moved" if move_bin else "exam") / checkpoint.replace("/", "_")
    events = EventLog(out, out.name)
    successes = errored = 0
    buckets = {"applied": 0, "adapted": 0, "not applied": 0, "trigger not reached": 0}
    for seed in seeds:
        start = events.path.stat().st_size if events.path.exists() else 0
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
                move_bin_after_lift=move_bin,
            )
        except Exception as error:  # the system's fault, not the student's: reported apart
            errored += 1
            events.emit("error", seed=seed, split="heldout", error=f"{type(error).__name__}: {error}")
            continue
        accepted = judge(spec, sim.ground_truth()).accepted
        successes += accepted
        if move_bin:
            bucket = adaptation_of(events.path, start)
            buckets[bucket] += 1
            buckets["adapted"] += bucket == "applied" and accepted
    result = ExamResult(successes=successes, errored=errored, n=len(seeds))
    if move_bin:
        result = result.model_copy(
            update={
                "adaptation_tests": buckets["applied"],
                "adapted": buckets["adapted"],
                "not_applied": buckets["not applied"],
                "trigger_not_reached": buckets["trigger not reached"],
            }
        )
    return result


def start(
    instruction: str, propose: Propose, runs_dir: Path, generator_calls: Callable[[], int] = lambda: 0
) -> tuple[TaskSpec, Path, int] | Refusal:
    """The spec and a run dir of its own, or Argon's refusal with nothing recorded."""
    spec = generate_task(instruction, propose)
    if isinstance(spec, Refusal):
        return spec
    run_dir = runs_dir / f"learn-{spec.name}-{spec.fingerprint()}-{time.strftime('%H%M%S')}"
    run_dir.mkdir(parents=True)
    write_atomically(run_dir / "spec.json", spec.model_dump_json(indent=2))
    return spec, run_dir, generator_calls()


def finish(
    run_dir: Path,
    *,
    exam_seeds: int,
    train: Train,
    make_student: Callable[[str], Policy],
    base: str | None = None,
) -> LearnReport:
    """Train on a collected batch and examine the result: the second half of `learn`, and all of `--resume`."""
    spec = TaskSpec.model_validate_json((run_dir / "spec.json").read_text())
    collected = Collected.model_validate_json((run_dir / "collected.json").read_text())
    training_set = build_training_set(run_dir, ROUND)
    summary = TrainingSetSummary.model_validate_json(training_set.with_suffix(".summary.json").read_text())
    demos = [EpisodeManifest.model_validate_json(p.read_text()) for p in (run_dir / "episodes").glob("*/manifest.json")]
    accepted = [m for m in demos if m.validation and m.validation.accepted]
    bin_moved = {m.episode_id for m in demos if any(p.perturbation.kind == "move_bin" for p in m.perturbations)}
    checkpoint = train(training_set, base or BASE_CHECKPOINT, ROUND)
    seeds = sorted(heldout_seeds())[:exam_seeds]
    checkpoints = [c for c in (base, checkpoint) if c is not None]
    moves = "move_bin" in spec.perturbation_kinds()
    report = LearnReport(
        run_id=run_dir.name,
        spec=spec,
        fingerprint=spec.fingerprint(),
        demos=summary.demonstrations,
        demos_errored=collected.demos - len(demos) + summary.unfinished,
        accepted=summary.accepted,
        examples=summary.examples,
        move_bin_share=collected.move_bin_share,
        demos_bin_moved=len(bin_moved),
        accepted_bin_moved=sum(m.episode_id in bin_moved for m in accepted),
        replayed=sum(m.replay_valid is True for m in accepted),
        replay_failures=sorted(m.episode_id for m in accepted if m.replay_valid is not True),
        argon_calls=argon_calls(run_dir),
        generator_calls=collected.generator_calls,
        checkpoint=checkpoint,
        finetune_metrics=finetune_metrics(run_dir),
        exam_seeds=seeds,
        exam={c: exam(spec, run_dir, c, seeds, make_student) for c in checkpoints},
        exam_bin_moved={c: exam(spec, run_dir, c, seeds, make_student, move_bin=True) for c in checkpoints if moves},
    )
    write_atomically(run_dir / "report.json", report.model_dump_json(indent=2))
    return report


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
    move_bin_share: float = MOVE_BIN_SHARE,
) -> LearnReport | Refusal:
    started = start(instruction, propose, runs_dir)
    if isinstance(started, Refusal):
        return started
    spec, run_dir, generator_calls = started
    collect(spec, run_dir, demos, workers, make_teacher, move_bin_share, generator_calls)
    return finish(run_dir, exam_seeds=exam_seeds, train=train, make_student=make_student, base=base)


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def _share(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be from 0 to 1, not {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a task from one sentence, end to end (SPEC §3.7)")
    parser.add_argument("instruction", nargs="?")
    parser.add_argument("--demos", type=_positive, default=40)
    parser.add_argument("--workers", type=_positive, default=8, help="teacher processes, each with its own budget")
    parser.add_argument(
        "--move-bin-share",
        type=_share,
        default=MOVE_BIN_SHARE,
        help="the share of demonstrations where the bin moves right after the lift, if the spec lets it move",
    )
    parser.add_argument("--exam-seeds", type=_positive, default=20)
    parser.add_argument("--with-base", action="store_true", help="also examine the base model on the same seeds")
    parser.add_argument("--collect-only", action="store_true", help="stop after the batch; no GPU needed")
    parser.add_argument("--resume", type=Path, help="train on and examine a run dir whose batch finished")
    args = parser.parse_args()
    if (args.instruction is None) == (args.resume is None):
        parser.error("give an instruction, or --resume with a run dir, not both")
    if args.resume is not None and not (args.resume / "collected.json").exists():
        parser.error(f"{args.resume} has no finished batch (collected.json); start the instruction again")
    if not args.collect_only and not has_cuda():
        parser.error("fine-tuning needs CUDA: run this on the RTX 5070 Ti, or pass --collect-only")
    runs = Path("runs")
    base = BASE_CHECKPOINT if args.with_base else None
    if args.resume is None:
        argon = Argon(EventLog(runs / "learn-generator", "learn-generator"))
        argon.check()
        started = start(args.instruction, argon_proposer(argon), runs, lambda: argon.calls)
        if isinstance(started, Refusal):
            print("refused:\n  " + "\n  ".join(started.reasons))
            raise SystemExit(1)
        spec, run_dir, generator_calls = started
        print(f"spec {spec.name} ({spec.fingerprint()}) in {run_dir}", flush=True)
        collected = collect(spec, run_dir, args.demos, args.workers, None, args.move_bin_share, generator_calls)
        print(f"batch done: {collected.model_dump_json()}", flush=True)
        if args.collect_only:
            print(f"resume with: uv run python -m isaacs_mom.tasks.learn --resume {run_dir}")
            return
    else:
        run_dir = args.resume
    report = finish(run_dir, exam_seeds=args.exam_seeds, train=finetune, make_student=StudentPolicy, base=base)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
