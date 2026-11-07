"""Gate B (SPEC §6.2): the student reads a real frame, returns a whitelisted skill, the simulator executes it.

uv run python -m isaacs_mom.student.smoke --sim mujoco --steps 5
"""

import argparse
import logging
import statistics

from isaacs_mom.orchestrator.cli import make_sim
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.sim.base import RobotSim
from isaacs_mom.student import vlm
from isaacs_mom.student.checkpoints import BASE_CHECKPOINT
from isaacs_mom.student.policy import StudentPolicy


def main() -> None:
    parser = argparse.ArgumentParser(prog="student-smoke")
    parser.add_argument("--sim", choices=["fake", "mujoco"], default="mujoco")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ckpt", default=BASE_CHECKPOINT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot: RobotSim = make_sim(args.sim)
    robot.reset(args.seed)
    policy = StudentPolicy(args.ckpt)
    feedback = "start"
    latencies: list[float] = []
    for _ in range(args.steps):
        obs = robot.observe()
        call = policy.act(obs, INSTRUCTION, feedback)
        latencies.append(policy.last_latency_ms)
        if call is None:
            print("student: done")
            break
        result = robot.execute(call)
        actions = len(result.executed_actions)
        print(f"step {obs.step}: {call.model_dump_json()} ok={result.ok} actions={actions} error={result.error}")
        feedback = f"{call.skill}: {'ok' if result.ok else result.error}"
    median = statistics.median(latencies)
    print(f"{args.ckpt} on {vlm.pick_device()}: median {median:.0f} ms over {len(latencies)} replies")


if __name__ == "__main__":
    main()
