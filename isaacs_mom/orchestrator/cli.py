import argparse
import json
import time
from pathlib import Path

from isaacs_mom.orchestrator import evaluation
from isaacs_mom.orchestrator.events import EventLog, read_events
from isaacs_mom.orchestrator.loop import (
    DEMOS_PER_ROUND,
    DEV_TRIALS,
    INSTRUCTION,
    MOVE_BIN_SHARE,
    bin_moves,
    check_carry_over,
    instruction_text,
    run_loop,
)
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.sim.mujoco.task import DEFAULT_TASK, TASKS
from isaacs_mom.student import (
    NO_CUDA,
    StubStudent,
    StudentPolicy,
    finetune,
    has_cuda,
    stub_finetune,
    valid_run_id,
)
from isaacs_mom.teacher.argon import Argon, stub_generate


def run_id(text: str) -> str:
    """Checked up front, because a bad id would only fail after a round's fine-tune."""
    if not valid_run_id(text):
        raise argparse.ArgumentTypeError(
            f"{text!r}: use ASCII letters, digits, '.', '_' or '-', not starting or ending with '.'"
        )
    return text


def instruction(text: str) -> str:
    try:
        return instruction_text(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def share(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text}: must be from 0 to 1")
    return value


def at_least_one(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value}: must be at least 1")
    return value


def make_sim(name: str, task: str = DEFAULT_TASK.name) -> TruthSim:
    if name == "fake":
        from isaacs_mom.sim.fake import FakeSim

        return FakeSim()
    from isaacs_mom.sim.mujoco import MujocoSim

    return MujocoSim(TASKS[task])


def main() -> None:
    parser = argparse.ArgumentParser(prog="mom")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="the teach -> fine-tune loop (SPEC §3.6)")
    run.add_argument("--sim", choices=["fake", "mujoco"], default="mujoco")
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--demos", type=at_least_one, default=DEMOS_PER_ROUND, help="teacher demonstrations per round")
    run.add_argument("--dev-trials", type=at_least_one, default=DEV_TRIALS, help="student trials the teacher diagnoses")
    run.add_argument("--run-id", type=run_id, default=time.strftime("run-%Y%m%d-%H%M%S"))
    run.add_argument(
        "--instruction", type=instruction, default=INSTRUCTION, help="what the robot learns; held-out tests use it too"
    )
    run.add_argument(
        "--from-run",
        type=Path,
        help="start from an earlier run in the same runs dir: its development episodes and its last checkpoint",
    )
    run.add_argument(
        "--move-bin-share",
        type=share,
        default=MOVE_BIN_SHARE,
        help="the share of teacher demonstrations where the bin moves right after the lift (SPEC §3.5)",
    )
    dashboard = commands.add_parser("dashboard", help="serve the live dashboard over a runs directory")
    dashboard.add_argument("--runs", type=Path, required=True, help="a directory of run directories, e.g. runs/")
    dashboard.add_argument("--port", type=int, default=8200)
    dashboard.add_argument(
        "--task", choices=sorted(TASKS), default=DEFAULT_TASK.name, help="the task preset a Teach-started run trains on"
    )
    evaluate = commands.add_parser("eval", help="the student evaluation report in three conditions (SPEC §6.10)")
    evaluation.add_arguments(evaluate)
    demo = commands.add_parser("demo", help="the judged single take, recorded to an MP4")
    demo.add_argument("--sim", choices=["fake", "mujoco"], default="mujoco")
    demo.add_argument("--ckpt", required=True)
    demo.add_argument("--seed", type=int, required=True, help="an unseen scene; use >= 900000")
    demo.add_argument("--perturb-at", type=int, required=True)
    demo.add_argument("--out", type=Path, default=Path("demo.mp4"))
    demo.add_argument("--run-id", type=run_id, default=time.strftime("demo-%Y%m%d-%H%M%S"))
    for command in (run, demo):
        command.add_argument(
            "--student", choices=["smolvlm", "stub"], default="smolvlm", help="stub: no model, no training"
        )
        command.add_argument(
            "--teacher",
            choices=["argon", "stub"],
            help="stub: no Argon calls and no key; the default on --sim fake, so the skeleton run needs no key",
        )
        command.add_argument(
            "--task", choices=sorted(TASKS), default=DEFAULT_TASK.name, help="the task preset (SPEC §3.1)"
        )
        command.add_argument("--runs", type=Path, default=Path("runs"), help="the directory the run directory goes in")
    args = parser.parse_args()
    if args.command == "eval":
        evaluation.run_from_args(evaluate, args)
        return
    if args.command == "dashboard":
        from isaacs_mom.orchestrator.dashboard.server import serve

        serve(args.runs, args.port, args.task)
        return
    if args.sim == "fake" and args.task != DEFAULT_TASK.name:
        parser.error(f"--sim fake plays only the default task, {DEFAULT_TASK.name}")
    if args.command == "run" and args.student == "smolvlm" and args.rounds > 0 and not has_cuda():
        run.error(NO_CUDA)
    make_student, train_student = (StubStudent, stub_finetune) if args.student == "stub" else (StudentPolicy, finetune)
    teacher = args.teacher or ("stub" if args.sim == "fake" else "argon")
    run_dir = args.runs / args.run_id

    def checked_argon(events: EventLog) -> Argon:
        # One budget per run (SPEC §2 rule 5), a Teach's generator calls too; checked first so a bad key fails fast.
        argon = Argon(events, generate=stub_generate if teacher == "stub" else None)
        argon.calls = sum(e.type == "argon_call" for e in read_events(run_dir))
        if teacher == "argon":
            argon.check()
        return argon

    if args.command == "demo":
        from isaacs_mom.orchestrator.demo import refuse_seen_take, run_demo

        try:
            refuse_seen_take(args.seed, run_dir)
        except ValueError as refused:
            demo.error(str(refused))
        manifest = run_demo(
            make_sim(args.sim, args.task),
            ckpt=args.ckpt,
            seed=args.seed,
            perturb_at=args.perturb_at,
            out=args.out,
            run_dir=run_dir,
            run_id=args.run_id,
            make_argon=checked_argon,
            make_student=make_student,
        )
        verdict = manifest.validation.reason if manifest.validation else "no verdict"
        print(f"{len(manifest.steps)} steps, validator: {verdict}\nvideo {args.out}, run saved to {run_dir}")
        return
    sim = None
    if args.from_run is not None:
        sim = make_sim(args.sim, args.task)
        try:  # before anything is written, so a refused source leaves no run behind
            check_carry_over(args.from_run, run_dir, sim.sim_version)
        except ValueError as refused:
            run.error(f"--from-run: {refused}")
    events = EventLog(run_dir, args.run_id)
    if not any(e.type == "instruction" for e in read_events(run_dir)):  # a Teach-started run already holds its own
        events.emit("instruction", text=args.instruction, source="cli")
    argon = checked_argon(events)
    results = run_loop(
        sim or make_sim(args.sim, args.task),
        run_dir,
        args.run_id,
        max_rounds=args.rounds,
        demos=args.demos,
        dev_trials=args.dev_trials,
        argon=argon,
        make_student=make_student,
        train_student=train_student,
        instruction=args.instruction,
        move_bin_share=args.move_bin_share,
        from_run=args.from_run,
    )
    if args.from_run is not None:
        record = json.loads((run_dir / "from_run.json").read_text())
        carried, left = len(record["episodes"]), len(record["left_behind"])
        print(
            f"round 0 started from {record['checkpoint']}: {carried} demonstrations carried, {left} exams left behind"
        )
    print(bin_moves(run_dir / "events.jsonl"))
    for r in results:
        scored = f"{r.successes}/{r.n - r.errored} held-out successes, {r.errored} errored"
        print(f"round {r.round} {r.checkpoint}: {scored}")
    print(argon.summary() + (" (stub teacher)" if teacher == "stub" else ""))
    print(f"run saved to {run_dir}")


if __name__ == "__main__":  # the Teach box starts a run as `python -m isaacs_mom.orchestrator.cli run ...`
    main()
