import argparse
import time
from pathlib import Path

from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import DEMOS_PER_ROUND, DEV_TRIALS, run_loop
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
    run.add_argument("--student", choices=["smolvlm", "stub"], default="smolvlm", help="stub: no model, no training")
    run.add_argument(
        "--teacher",
        choices=["argon", "stub"],
        help="stub: no Argon calls and no key; the default on --sim fake, so the skeleton run needs no key",
    )
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--demos", type=at_least_one, default=DEMOS_PER_ROUND, help="teacher demonstrations per round")
    run.add_argument("--dev-trials", type=at_least_one, default=DEV_TRIALS, help="student trials the teacher diagnoses")
    run.add_argument("--run-id", type=run_id, default=time.strftime("run-%Y%m%d-%H%M%S"))
    run.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK.name, help="the task preset (SPEC §3.1)")
    args = parser.parse_args()
    if args.sim == "fake" and args.task != DEFAULT_TASK.name:
        parser.error(f"--sim fake plays only the default task, {DEFAULT_TASK.name}")

    if args.student == "smolvlm" and args.rounds > 0 and not has_cuda():
        run.error(NO_CUDA)
    make_student, train_student = (StubStudent, stub_finetune) if args.student == "stub" else (StudentPolicy, finetune)
    teacher = args.teacher or ("stub" if args.sim == "fake" else "argon")
    run_dir = Path("runs") / args.run_id
    # One budget for the whole run (SPEC §2 rule 5); checked before the sim, so a bad key stops the run in seconds.
    argon = Argon(EventLog(run_dir, args.run_id), generate=stub_generate if teacher == "stub" else None)
    if teacher == "argon":
        argon.check()  # the stub has no key or model to check, so it spends nothing
    results = run_loop(
        make_sim(args.sim, args.task),
        run_dir,
        args.run_id,
        max_rounds=args.rounds,
        demos=args.demos,
        dev_trials=args.dev_trials,
        argon=argon,
        make_student=make_student,
        train_student=train_student,
    )
    for r in results:
        scored = f"{r.successes}/{r.n - r.errored} held-out successes, {r.errored} errored"
        print(f"round {r.round} {r.checkpoint}: {scored}")
    print(argon.summary() + (" (stub teacher)" if teacher == "stub" else ""))
    print(f"run saved to {run_dir}")
