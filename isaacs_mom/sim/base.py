"""Simulator protocols (SPEC §4.1).

The student and teacher get a RobotSim; only the validator and orchestrator get a TruthSim.
"""

from typing import Protocol

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall


class RobotSim(Protocol):
    sim_version: str
    robot_config: str

    def reset(self, seed: int) -> str: ...

    def observe(self) -> Observation: ...

    def execute(self, call: SkillCall) -> ExecResult: ...


class TruthSim(RobotSim, Protocol):
    def ground_truth(self) -> GroundTruth: ...

    def perturb(self, perturbation: Perturbation) -> bool:
        """Move the box (only while it rests on the table, out of the robot's hands) or the bin (only to where it
        touches nothing). True when it moved; False leaves the scene as it was."""
        ...

    def render_third_person(self) -> bytes: ...
