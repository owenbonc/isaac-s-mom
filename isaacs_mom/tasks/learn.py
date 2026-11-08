"""Teach a generated task end to end (SPEC §3.7): `uv run python -m isaacs_mom.tasks.learn "<instruction>" --demos N
--workers K`, or the batch alone with `--collect-only`, then `--resume runs/<run_id>` on the GPU."""

import argparse
import json
import shutil
import time
import uuid
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
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
from isaacs_mom.student.checkpoints import RUNS_DIR
from isaacs_mom.tasks.generate import Propose, argon_proposer, generate_task
from isaacs_mom.tasks.rules import judge
from isaacs_mom.tasks.spec import Refusal, TaskSpec
from isaacs_mom.teacher import ArgonPolicy
from isaacs_mom.teacher.argon import Argon, ArgonUnavailable, MissingKey
from isaacs_mom.validator.recorder import write_atomically, write_manifest
from isaacs_mom.validator.replay import REPLAY_ERRORS, replay
from isaacs_mom.validator.training import build_training_set

DEMO_SEED_START = DEV_SEED_START + 5000  # clear of the loop's trials and choices and of the readiness trial's seeds
ROUND = 1
MOVE_BIN_SHARE = 0.5
# One lost demonstration raises one of these; a fatal Argon error (any other 4xx) is none, so it stops the batch (§3.3).
EPISODE_ERRORS = (ArgonUnavailable, *REPLAY_ERRORS)
# The generator runs before its spec names the run, so its log is shared by every start; each run keeps its own calls.
GENERATOR_CALLS = Path("generator") / "events.jsonl"
MakePolicy = Callable[[MujocoSim], Policy]
Train = Callable[[Path, str, int], str]  # (training set, previous checkpoint, round) -> new checkpoint


class Collected(BaseModel):
    """What the batch did, saved beside its episodes so a resumed run reports it."""

    demos: int
    workers: int
    move_bin_share: float  # 0 when the spec doesn't let the bin move
    moved_seeds: list[int]  # where the move was due; it applies only once the box is lifted
    stopped_early: dict[str, str] = {}  # teacher process -> the error that stopped it; what it didn't record errored


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
    reproduced: int
    replay_failures: list[str]  # accepted demonstrations that don't
    argon_calls: int  # every teacher process's, failed calls included
    generator_calls: int
    checkpoint: str
    finetune_metrics: dict[str, object]
    exam_seeds: list[int]
    exam: dict[str, ExamResult]  # checkpoint -> result
    exam_bin_moved: dict[str, ExamResult]  # empty when the spec doesn't let the bin move


def demonstrate(
    spec_json: str,
    seeds: Sequence[int],
    out: str,
    run_id: str,
    make_teacher: MakePolicy | None = None,
    moved: Collection[int] = (),
) -> None:
    """One teacher process (Argon by default): a demonstration on each seed and a replay of each accepted one, both
    by the spec's rule. One lost is logged and the rest go on; a missing key or a fatal Argon error raises."""
    spec = TaskSpec.model_validate_json(spec_json)
    rule = partial(judge, spec)
    out_dir = Path(out)
    sim = MujocoSim(spec.task_config())
    events = EventLog(out_dir, run_id)
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
                instruction=spec.instruction,
                move_bin_after_lift=seed in moved,
                verdict=rule,
            )
        except MissingKey:
            raise  # a RuntimeError, but every later demonstration would need the key too
        except EPISODE_ERRORS as error:
            events.emit("error", seed=seed, split="dev", error=f"{type(error).__name__}: {error}")
            continue
        if manifest.validation and manifest.validation.accepted:
            path = out_dir / "episodes" / manifest.episode_id / "manifest.json"
            try:
                replay(sim, path, verdict=rule)
            except REPLAY_ERRORS as error:
                manifest.replay_valid = False
                write_manifest(path, manifest)
                reason = f"replay raised {type(error).__name__}: {error}"
                events.emit("error", episode_id=manifest.episode_id, seed=seed, split="dev", error=reason)


def collect(
    spec: TaskSpec,
    run_dir: Path,
    demos: int,
    workers: int,
    make_teacher: MakePolicy | None = None,
    move_bin_share: float = MOVE_BIN_SHARE,
) -> Collected:
    """`demos` demonstrations over `workers` teacher processes (this one when 1), gathered into run_dir/episodes;
    across processes `make_teacher` must be picklable. If one stops early, what every process recorded is still
    gathered and `collected.json` written, so `--resume` can finish the run, and then its error raises."""
    seeds = list(range(DEMO_SEED_START, DEMO_SEED_START + demos))
    share = move_bin_share if "move_bin" in spec.perturbation_kinds() else 0.0
    moved = sorted(seeds[i] for i in moves_bin(demos, share))
    shares = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    outs = [str(run_dir / "workers" / f"w{i}") for i in range(len(shares))]
    spec_json = spec.model_dump_json()
    errors: dict[str, BaseException] = {}
    if len(shares) == 1:
        try:
            demonstrate(spec_json, shares[0], outs[0], run_dir.name, make_teacher, moved)
        except Exception as error:
            errors[Path(outs[0]).name] = error
    else:
        with ProcessPoolExecutor(len(shares), mp_context=get_context("spawn")) as pool:
            futures = {
                Path(out).name: pool.submit(demonstrate, spec_json, share_seeds, out, run_dir.name, make_teacher, moved)
                for share_seeds, out in zip(shares, outs, strict=True)
            }
        errors = {name: error for name, future in futures.items() if (error := future.exception()) is not None}
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
        stopped_early={name: f"{type(error).__name__}: {error}" for name, error in errors.items()},
    )
    write_atomically(run_dir / "collected.json", collected.model_dump_json(indent=2))
    if errors:
        raise next(iter(errors.values()))
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
    """The student alone on the held-out seeds, each counted by the spec's verdict its manifest was saved with; a
    crash counts as errored (§3.5). With `move_bin`, the bin moves once right after the lift, and the scored episodes
    are split as §6.10 says."""
    sim = MujocoSim(spec.task_config())
    out = run_dir / ("exam-bin-moved" if move_bin else "exam") / checkpoint.replace("/", "_")
    events = EventLog(out, run_dir.name)
    successes = errored = 0
    buckets = {"applied": 0, "adapted": 0, "not applied": 0, "trigger not reached": 0}
    for seed in seeds:
        start = events.path.stat().st_size if events.path.exists() else 0
        try:
            manifest = run_episode(
                sim,
                make_student(checkpoint),
                run_dir=out,
                events=events,
                seed=seed,
                split="heldout",
                round_=ROUND,
                instruction=spec.instruction,
                move_bin_after_lift=move_bin,
                verdict=partial(judge, spec),
            )
        except Exception as error:  # the system's fault, not the student's: reported apart
            errored += 1
            events.emit("error", seed=seed, split="heldout", error=f"{type(error).__name__}: {error}")
            continue
        accepted = manifest.validation is not None and manifest.validation.accepted
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
    instruction: str, propose: Propose, runs_dir: Path, generator: EventLog | None = None
) -> tuple[TaskSpec, Path] | Refusal:
    """The spec and a run dir of its own, holding the spec and the `argon_call` events `generator` logged (none
    without it), or Argon's refusal with nothing recorded."""
    spec = generate_task(instruction, propose)
    if isinstance(spec, Refusal):
        return spec
    run_dir = runs_dir / f"learn-{spec.name}-{spec.fingerprint()}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    write_atomically(run_dir / "spec.json", spec.model_dump_json(indent=2))
    calls: list[Event] = []
    if generator is not None and generator.path.exists():
        events = map(Event.model_validate_json, generator.path.read_text().splitlines())
        calls = [e for e in events if e.type == "argon_call" and e.run_id == generator.run_id]
    (run_dir / GENERATOR_CALLS).parent.mkdir()
    write_atomically(run_dir / GENERATOR_CALLS, "".join(e.model_dump_json() + "\n" for e in calls))
    return spec, run_dir


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
    generator_calls = len((run_dir / GENERATOR_CALLS).read_text().splitlines())
    teacher_calls = argon_calls(run_dir)
    training_set = build_training_set(run_dir, ROUND)
    summary = TrainingSetSummary.model_validate_json(training_set.with_suffix(".summary.json").read_text())
    demos = [EpisodeManifest.model_validate_json(p.read_text()) for p in (run_dir / "episodes").glob("*/manifest.json")]
    accepted = [m for m in demos if m.validation and m.validation.accepted]
    bin_moved = {m.episode_id for m in demos if any(p.perturbation.kind == "move_bin" for p in m.perturbations)}
    checkpoint = train(training_set, base or BASE_CHECKPOINT, ROUND)
    seeds = sorted(heldout_seeds())[:exam_seeds]
    checkpoints = [checkpoint] if base in (None, checkpoint) else [base, checkpoint]
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
        reproduced=sum(m.replay_valid is True for m in accepted),
        replay_failures=sorted(m.episode_id for m in accepted if m.replay_valid is not True),
        argon_calls=teacher_calls,
        generator_calls=generator_calls,
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
    spec, run_dir = started
    collect(spec, run_dir, demos, workers, make_teacher, move_bin_share)
    return finish(run_dir, exam_seeds=exam_seeds, train=train, make_student=make_student, base=base)


def summary(report: LearnReport) -> str:
    """The report in a few lines, each count with its n (rule 6)."""
    spec, metrics = report.spec, report.finetune_metrics
    moves = "the bin may move" if report.exam_bin_moved else "nothing moves"
    seeds = report.exam_seeds
    lines = [
        f'"{spec.instruction}" -> {spec.name} ({report.fingerprint}), {spec.stance.replace("_", " ")}, {moves}',
        f"demonstrations: {report.accepted}/{report.demos} accepted ({report.demos_errored} errored), "
        f"{report.accepted_bin_moved} of them with the bin moved; {report.examples} training examples; "
        f"{report.reproduced}/{report.accepted} accepted replay by the spec's rule",
        f"Argon: {report.argon_calls} teacher calls and {report.generator_calls} for the spec",
        f"fine-tune: {metrics.get('seconds', '-')} s, peak {metrics.get('peak_gpu_memory_gb', '-')} GiB"
        if metrics
        else "fine-tune: nothing to train on, so no new checkpoint",
        f"held-out seeds {seeds[0]}-{seeds[-1]}, student alone, successes / (n - errored):",
    ]
    for checkpoint, result in report.exam.items():
        line = f"  {checkpoint}: {result.successes}/{result.n - result.errored}"
        if (moved := report.exam_bin_moved.get(checkpoint)) is not None:
            line += (
                f"; bin moved after the lift {moved.successes}/{moved.n - moved.errored}, adapted "
                f"{moved.adapted}/{moved.adaptation_tests} (not applied {moved.not_applied}, "
                f"no lift {moved.trigger_not_reached})"
            )
        lines.append(line)
    return "\n".join(lines)


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
    if args.resume is not None:
        if args.collect_only:
            parser.error("--collect-only stops after the batch and --resume starts after it: give one")
        if args.resume.resolve().parent != RUNS_DIR.resolve():
            parser.error(f"--resume takes a dir in ./{RUNS_DIR}, where the exam finds its adapter, not {args.resume}")
        if not (args.resume / "collected.json").exists():
            parser.error(f"{args.resume} has no finished batch (collected.json); start the instruction again")
    if not args.collect_only and not has_cuda():
        parser.error("fine-tuning needs CUDA: run this on the Windows RTX 5070, or pass --collect-only")
    base = BASE_CHECKPOINT if args.with_base else None
    if args.resume is None:
        generator = EventLog(RUNS_DIR / "learn-generator", f"learn-generator-{uuid.uuid4().hex[:8]}")
        argon = Argon(generator)
        argon.check()
        started = start(args.instruction, argon_proposer(argon), RUNS_DIR, generator)
        if isinstance(started, Refusal):
            print("refused:\n  " + "\n  ".join(started.reasons))
            raise SystemExit(1)
        spec, run_dir = started
        print(f"spec {spec.name} ({spec.fingerprint()}) in {run_dir}", flush=True)
        collected = collect(spec, run_dir, args.demos, args.workers, None, args.move_bin_share)
        print(f"batch done: {collected.model_dump_json()}", flush=True)
        if args.collect_only:
            print(f"resume with: uv run python -m isaacs_mom.tasks.learn --resume {run_dir}")
            return
    else:
        run_dir = args.resume
    report = finish(run_dir, exam_seeds=args.exam_seeds, train=finetune, make_student=StudentPolicy, base=base)
    print(summary(report))
    print(f"full report: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
