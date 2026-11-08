"""A MujocoSim whose after-lift bin move puts the bin at one fixed bearing and distance, instead of redrawing it from
the task's own arc. Everything else (the trigger in run_episode, the refusals, the scene) is the repo's, unchanged.

The repo's MujocoSim.perturb("move_bin") draws a new pose with `bin_pose(rng, task, around=stance)`, i.e. from the
spec's own [85, 112] deg x [0.40, 0.50] m arc. Here only that draw is replaced; the same two refusals apply (a box
already on the bin floor; a bin that would land touching the box or the robot).
"""

import copy

import mujoco
import numpy as np

from isaacs_mom.contracts import Perturbation
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.task import TaskConfig


class FixedBinMoveSim(MujocoSim):
    def __init__(self, task: TaskConfig, bearing_deg: float | None, distance_m: float | None) -> None:
        super().__init__(task)
        self.target = None if bearing_deg is None else (float(bearing_deg), float(distance_m or 0.45))
        if self.target is not None:
            self.sim_version += f" move_bin_to={self.target[0]:g}deg@{self.target[1]:g}m"

    def perturb(self, perturbation: Perturbation) -> bool:
        if perturbation.kind != "move_bin" or self.target is None:
            return super().perturb(perturbation)
        if self._bin_floor in self._touching(self.data, {self._box_geom}):  # moving it would tip the box out
            return False
        bearing, dist = np.deg2rad(self.target[0]), self.target[1]
        # Around the stance the legs hold (fixed feet: the origin), facing it, as bin_pose does.
        xy = (float(self.goal_xy[0] + dist * np.cos(bearing)), float(self.goal_xy[1] + dist * np.sin(bearing)))
        trial = copy.copy(self.data)
        self._put_bin(trial, xy, float(bearing))
        mujoco.mj_fwdPosition(self.model, trial)
        if self._touching(trial, self._bin_geoms):  # it would land on the box or the robot
            return False
        self._put_bin(self.data, xy, float(bearing))
        mujoco.mj_forward(self.model, self.data)
        return True
