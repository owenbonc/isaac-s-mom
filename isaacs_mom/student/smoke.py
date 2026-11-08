"""Gate B (SPEC §6.2): the student reads a real frame, returns a whitelisted skill, the simulator executes it.

uv run python -m isaacs_mom.student.smoke --sim mujoco --steps 5
"""

import argparse
import logging
import statistics
import sys

import torch

from isaacs_mom.orchestrator.cli import make_sim
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.sim.base import RobotSim
from isaacs_mom.student import vlm
from isaacs_mom.student.checkpoints import BASE_CHECKPOINT
from isaacs_mom.student.policy import StudentPolicy

GB = 2**30


def device_memory_gb(device: str) -> float | None:
    """What the accelerator holds for the model; None on CPU, where the process peak is the whole story."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / GB
    if device == "mps":
        return torch.mps.driver_allocated_memory() / GB
    return None


def process_peak_gb() -> float | None:
    """Peak resident memory of this process; on a Mac's unified memory this is the student's real footprint."""
    if sys.platform == "win32":
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / GB if sys.platform == "darwin" else peak * 1024 / GB  # bytes on macOS, KiB on Linux


def gb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(prog="student-smoke")
    parser.add_argument("--sim", choices=["fake", "mujoco"], default="mujoco")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ckpt", default=BASE_CHECKPOINT, help="ckpt-r0 or <run_id>/ckpt-r<round> under runs/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot: RobotSim = make_sim(args.sim)
    robot.reset(args.seed)
    policy = StudentPolicy(args.ckpt)
    device = vlm.pick_device()
    feedback = "start"
    latencies: list[float] = []
    readings: list[float] = []
    for _ in range(args.steps):
        obs = robot.observe()
        call = policy.act(obs, INSTRUCTION, feedback)
        latencies.append(policy.last_latency_ms)
        # MPS reports only what it holds now, so the peak is the largest reading after any reply.
        if (reading := device_memory_gb(device)) is not None:
            readings.append(reading)
        if call is None:
            print("student: done")
            break
        result = robot.execute(call)
        actions = len(result.executed_actions)
        print(f"step {obs.step}: {call.model_dump_json()} ok={result.ok} actions={actions} error={result.error}")
        feedback = f"{call.skill}: {'ok' if result.ok else result.error}"
    median = statistics.median(latencies)
    print(f"{args.ckpt} on {device}: median {median:.0f} ms over {len(latencies)} replies")
    print(f"peak memory: {device} {gb(max(readings, default=None))}, process {gb(process_peak_gb())}")


if __name__ == "__main__":
    main()
