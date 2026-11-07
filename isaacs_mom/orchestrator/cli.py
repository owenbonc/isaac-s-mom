import argparse
import time
from pathlib import Path

from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.student import StubStudent, StudentPolicy, finetune, stub_finetune, valid_run_id
from isaacs_mom.teacher.argon import Argon, stub_generate


def run_id(text: str) -> str:
    """Checked up front, because a bad id would only fail after a paid fine-tune."""
    if not valid_run_id(text):
        raise argparse.ArgumentTypeError(
            f"{text!r}: use ASCII letters, digits, '.', '_' or '-', not starting or ending with '.'"
        )
    return text


def make_sim(name: str) -> TruthSim:
    if name == "fake":
        from isaacs_mom.sim.fake import FakeSim

        return FakeSim()
    from isaacs_mom.sim.mujoco import MujocoSim

    return MujocoSim()


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
    run.add_argument("--run-id", type=run_id, default=time.strftime("run-%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    make_student, train_student = (StubStudent, stub_finetune) if args.student == "stub" else (StudentPolicy, finetune)
    teacher = args.teacher or ("stub" if args.sim == "fake" else "argon")
    argons: list[Argon] = []

    def make_argon(events: EventLog) -> Argon:
        argons.append(Argon(events, generate=stub_generate if teacher == "stub" else None))
        return argons[-1]

    run_dir = Path("runs") / args.run_id
    results = run_loop(
        make_sim(args.sim),
        run_dir,
        args.run_id,
        max_rounds=args.rounds,
        make_student=make_student,
        train_student=train_student,
        make_argon=make_argon,
    )
    for r in results:
        print(f"round {r.round} {r.checkpoint}: {r.successes}/{r.n} held-out successes")
    print(argons[0].summary() + (" (stub teacher)" if teacher == "stub" else ""))
    print(f"run saved to {run_dir}")
