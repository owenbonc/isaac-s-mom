import argparse
import time
from pathlib import Path

from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.sim.mujoco.task import DEFAULT_TASK, TASKS
from isaacs_mom.teacher.argon import Argon, stub_generate


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
    run.add_argument(
        "--teacher",
        choices=["argon", "stub"],
        help="stub: no Argon calls and no key; the default on --sim fake, so the skeleton run needs no key",
    )
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    run.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK.name, help="the task preset (SPEC §3.1)")
    args = parser.parse_args()
    if args.sim == "fake" and args.task != DEFAULT_TASK.name:
        parser.error(f"--sim fake plays only the default task, {DEFAULT_TASK.name}")

    teacher = args.teacher or ("stub" if args.sim == "fake" else "argon")
    argons: list[Argon] = []

    def make_argon(events: EventLog) -> Argon:
        argons.append(Argon(events, generate=stub_generate if teacher == "stub" else None))
        return argons[-1]

    run_dir = Path("runs") / args.run_id
    results = run_loop(
        make_sim(args.sim, args.task), run_dir, args.run_id, max_rounds=args.rounds, make_argon=make_argon
    )
    for r in results:
        print(f"round {r.round} {r.checkpoint}: {r.successes}/{r.n} held-out successes")
    print(argons[0].summary() + (" (stub teacher)" if teacher == "stub" else ""))
    print(f"run saved to {run_dir}")
