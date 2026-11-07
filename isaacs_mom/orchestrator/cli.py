import argparse
import time
from pathlib import Path

from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.teacher.argon import Argon, stub_generate


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
    run.add_argument(
        "--teacher",
        choices=["argon", "stub"],
        help="stub: no Argon calls and no key; the default on --sim fake, so the skeleton run needs no key",
    )
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    teacher = args.teacher or ("stub" if args.sim == "fake" else "argon")
    run_dir = Path("runs") / args.run_id
    # One budget for the whole run (SPEC §2 rule 5); checked before the sim, so a bad key stops the run in seconds.
    argon = Argon(EventLog(run_dir, args.run_id), generate=stub_generate if teacher == "stub" else None)
    if teacher == "argon":
        argon.check()  # the stub has no key or model to check, so it spends nothing
    results = run_loop(make_sim(args.sim), run_dir, args.run_id, max_rounds=args.rounds, argon=argon)
    for r in results:
        print(f"round {r.round} {r.checkpoint}: {r.successes}/{r.n} held-out successes")
    print(argon.summary() + (" (stub teacher)" if teacher == "stub" else ""))
    print(f"run saved to {run_dir}")
