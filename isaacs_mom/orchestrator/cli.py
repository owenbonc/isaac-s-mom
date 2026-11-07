import argparse
import time
from functools import partial
from pathlib import Path

from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.student import (
    NO_CUDA,
    TRAINERS,
    StubStudent,
    StudentPolicy,
    finetune,
    has_cuda,
    stub_finetune,
    valid_run_id,
)


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
        "--trainer", choices=list(TRAINERS), default="local", help="local: this machine's CUDA GPU; modal: an A10G"
    )
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--run-id", type=run_id, default=time.strftime("run-%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    if args.student == "stub":
        make_student, train_student = StubStudent, stub_finetune
    else:
        if args.trainer == "local" and not has_cuda():
            run.error(NO_CUDA)
        make_student, train_student = StudentPolicy, partial(finetune, trainer=TRAINERS[args.trainer])
    run_dir = Path("runs") / args.run_id
    results = run_loop(
        make_sim(args.sim),
        run_dir,
        args.run_id,
        max_rounds=args.rounds,
        make_student=make_student,
        train_student=train_student,
    )
    for r in results:
        print(f"round {r.round} {r.checkpoint}: {r.successes}/{r.n} held-out successes")
    print(f"run saved to {run_dir}")
