# ruff: noqa: E501, SIM115  (kept as run, for the record)
"""Time the CPU-side pieces on this Mac: head-camera render, the walking policy, physics, and whole skills."""

import statistics
import time

import mujoco
import numpy as np

from isaacs_mom.contracts import SkillCall
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.task import TASKS


def timed(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out), max(out), n


walk = MujocoSim()  # across_room: walking legs
walk.reset(1000)
print("observe (320x240 render + JPEG), ms median/max/n:", timed(walk.observe, 50))
w = walk.walker
cmd = np.zeros(3)
print("walking policy act (LSTM+MLP, numpy), ms:", timed(lambda: w.act(walk.data, cmd), 2000))
print("mj_step (G1 29-DoF + scene), ms:", timed(lambda: mujoco.mj_step(walk.model, walk.data), 2000))
print("timestep", walk.model.opt.timestep)

fixed = MujocoSim(TASKS["fixed_feet"])
for seed in (1000, 1001, 1002):
    fixed.reset(seed)
    for call in (
        SkillCall(skill="hug", params={}, reason="bench"),
        SkillCall(skill="lift", params={}, reason="bench"),
        SkillCall(skill="turn_left", params={"angle_deg": 45}, reason="bench"),
    ):
        t = time.perf_counter()
        r = fixed.execute(call)
        dt = time.perf_counter() - t
        print(
            f"fixed seed {seed} {call.skill}: ok={r.ok} wall={dt * 1000:.0f} ms, {len(r.executed_actions)} executed targets, sim {r.timestamps[-1] - r.timestamps[0] if r.timestamps else 0:.2f} s"
        )
walk.reset(1000)
for call in (
    SkillCall(skill="hug", params={}, reason="b"),
    SkillCall(skill="lift", params={}, reason="b"),
    SkillCall(skill="walk_back", params={"distance_cm": 30}, reason="b"),
    SkillCall(skill="turn_left", params={"angle_deg": 90}, reason="b"),
):
    t = time.perf_counter()
    r = walk.execute(call)
    dt = time.perf_counter() - t
    print(
        f"walk {call.skill}: ok={r.ok} err={r.error} wall={dt * 1000:.0f} ms, sim {r.timestamps[-1] - r.timestamps[0] if r.timestamps else 0:.2f} s"
    )
