import argparse
import time
from pathlib import Path

from isaacs_mom.orchestrator.loop import run_loop
from isaacs_mom.sim.base import TruthSim


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
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    dashboard = commands.add_parser("dashboard", help="serve the live dashboard over a run directory")
    dashboard.add_argument("--run", type=Path, required=True)
    dashboard.add_argument("--port", type=int, default=8200)
    demo = commands.add_parser("demo", help="the judged single take, recorded to an MP4")
    demo.add_argument("--sim", choices=["fake", "mujoco"], default="mujoco")
    demo.add_argument("--ckpt", required=True)
    demo.add_argument("--seed", type=int, required=True, help="an unseen scene; use >= 900000")
    demo.add_argument("--perturb-at", type=int, required=True)
    demo.add_argument("--out", type=Path, default=Path("demo.mp4"))
    demo.add_argument("--run-id", default=time.strftime("demo-%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    if args.command == "dashboard":
        from isaacs_mom.orchestrator.dashboard.server import serve

        serve(args.run, args.port)
        return
    if args.command == "demo":
        from isaacs_mom.orchestrator.demo import run_demo

        run_dir = Path("runs") / args.run_id
        manifest = run_demo(
            make_sim(args.sim),
            ckpt=args.ckpt,
            seed=args.seed,
            perturb_at=args.perturb_at,
            out=args.out,
            run_dir=run_dir,
            run_id=args.run_id,
        )
        verdict = manifest.validation.reason if manifest.validation else "no verdict"
        print(f"{len(manifest.steps)} steps, validator: {verdict}\nvideo {args.out}, run saved to {run_dir}")
        return

    run_dir = Path("runs") / args.run_id
    results = run_loop(make_sim(args.sim), run_dir, args.run_id, max_rounds=args.rounds)
    for r in results:
        print(f"round {r.round} {r.checkpoint}: {r.successes}/{r.n} held-out successes")
    print(f"run saved to {run_dir}")
