"""Workstream 1: MuJoCo scene, Unitree G1 (base fixed), head camera, IK and skills (SPEC §3.1)."""

from typing import Literal

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall


class MujocoSim:
    sim_version = "mujoco"
    robot_config = "unitree_g1_fixed_base"

    def __init__(self, robot: Literal["g1", "panda"] = "g1") -> None:
        raise NotImplementedError("MujocoSim: workstream 1")

    def reset(self, seed: int) -> str:
        raise NotImplementedError

    def observe(self) -> Observation:
        raise NotImplementedError

    def execute(self, call: SkillCall) -> ExecResult:
        raise NotImplementedError

    def ground_truth(self) -> GroundTruth:
        raise NotImplementedError

    def perturb(self, perturbation: Perturbation) -> None:
        raise NotImplementedError

    def render_third_person(self) -> bytes:
        raise NotImplementedError
