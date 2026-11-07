# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""Unitree's pretrained G1 walking policy, run in numpy, driving the 12 leg actuators of a free-floating G1.

The policy (an LSTM with an MLP head) reads the pelvis IMU, the leg joints and a gait clock, and outputs leg
position targets at 50 Hz. Waist and arms stay under the caller's control, so the robot can walk while it holds things.
It never stands still: at a zero command it steps in place, so `command_to` holds a floor pose instead.
"""

from pathlib import Path

import mujoco
import numpy as np

POLICY = Path(__file__).parent / "assets" / "unitree_g1_walk" / "policy.npz"
LEG_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
# Gains, stance and observation scales the policy was trained with (unitree_rl_gym deploy_mujoco/configs/g1.yaml).
LEG_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
LEG_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
STAND_LEGS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float32)
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
MAX_COMMAND = np.array([1.0, 1.0, 1.0])  # the training ranges: vx, vy in m/s, yaw rate in rad/s
GAIT_PERIOD = 0.8
POLICY_DT = 0.02
POSITION_GAIN = 3.0
YAW_GAIN = 1.5
LEAD_S = 0.3  # aim where the pelvis will be this far ahead, so the gait's momentum doesn't overshoot the goal


def configure_legs(spec: mujoco.MjSpec) -> None:
    """Give the leg position actuators the PD gains the policy was trained with."""
    for name, kp, kd in zip(LEG_JOINTS, LEG_KP, LEG_KD, strict=True):
        spec.actuator(name).set_to_position(kp=float(kp), kv=float(kd))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Walker:
    """Call `act` once per POLICY_DT of sim time with a body-frame command (vx, vy, yaw_rate); step physics between."""

    def __init__(self, model: mujoco.MjModel) -> None:
        substeps = round(POLICY_DT / model.opt.timestep)
        if substeps < 1 or not np.isclose(substeps * model.opt.timestep, POLICY_DT):
            raise ValueError(f"timestep {model.opt.timestep} does not divide the policy period {POLICY_DT}")
        base = model.body_jntadr[model.body("pelvis").id]
        if base < 0 or model.jnt_type[base] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("the pelvis needs its free joint to walk")
        self._base_q = int(model.jnt_qposadr[base])
        self._base_v = int(model.jnt_dofadr[base])
        joints = [model.joint(name).id for name in LEG_JOINTS]
        self._leg_q = model.jnt_qposadr[joints]
        self._leg_v = model.jnt_dofadr[joints]
        by_joint = {int(model.actuator_trnid[i, 0]): i for i in range(model.nu)}
        self.actuators = np.array([by_joint[j] for j in joints])
        gain = model.actuator_gainprm[self.actuators, 0]
        bias = model.actuator_biasprm[self.actuators]
        if not (np.allclose(gain, LEG_KP) and np.allclose(bias[:, 1], -LEG_KP) and np.allclose(bias[:, 2], -LEG_KD)):
            raise ValueError("leg actuators lack the policy's PD gains; build the model with configure_legs")
        with np.load(POLICY) as weights:
            self._w_ih, self._w_hh, self._b = weights["w_ih"], weights["w_hh"], weights["b"]
            self._w1, self._b1, self._w2, self._b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
        self.reset()

    def reset(self) -> None:
        """Forget the gait: zero the LSTM memory, the last action and the gait clock."""
        self._h = np.zeros(self._w_hh.shape[1], dtype=np.float32)
        self._c = np.zeros_like(self._h)
        self._action = np.zeros(len(LEG_JOINTS), dtype=np.float32)
        self._ticks = 0

    def act(self, data: mujoco.MjData, command: np.ndarray) -> None:
        """Read the pelvis and leg state from `data` and write the next leg position targets into `data.ctrl`."""
        self._ticks += 1
        phase = 2 * np.pi * (self._ticks * POLICY_DT % GAIT_PERIOD) / GAIT_PERIOD
        obs = np.concatenate(
            [
                data.qvel[self._base_v + 3 : self._base_v + 6] * ANG_VEL_SCALE,
                self._gravity_in_pelvis(data),
                np.clip(command, -MAX_COMMAND, MAX_COMMAND) * COMMAND_SCALE,
                data.qpos[self._leg_q] - STAND_LEGS,
                data.qvel[self._leg_v] * JOINT_VEL_SCALE,
                self._action,
                [np.sin(phase), np.cos(phase)],
            ]
        ).astype(np.float32)
        self._action = self._policy(obs)
        data.ctrl[self.actuators] = self._action * ACTION_SCALE + STAND_LEGS

    def command_to(
        self,
        data: mujoco.MjData,
        goal_xy: np.ndarray,
        goal_yaw: float,
        max_speed: float = 0.3,
        max_yaw_rate: float = 0.6,
    ) -> np.ndarray:
        """A command that walks the pelvis to a floor pose and then holds it there."""
        x, y = data.qpos[self._base_q : self._base_q + 2] + LEAD_S * data.qvel[self._base_v : self._base_v + 2]
        yaw = self.yaw(data)
        dx, dy = goal_xy[0] - x, goal_xy[1] - y
        local = POSITION_GAIN * np.array([np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy])
        speed = np.linalg.norm(local)
        if speed > max_speed:
            local *= max_speed / speed
        turn = np.clip(YAW_GAIN * wrap(goal_yaw - yaw), -max_yaw_rate, max_yaw_rate)
        return np.array([local[0], local[1], turn])

    def yaw(self, data: mujoco.MjData) -> float:
        w, x, y, z = data.qpos[self._base_q + 3 : self._base_q + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _gravity_in_pelvis(self, data: mujoco.MjData) -> np.ndarray:
        w, x, y, z = data.qpos[self._base_q + 3 : self._base_q + 7]
        return np.array([2 * (w * y - z * x), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        # PyTorch LSTM gate order: input, forget, cell, output.
        i, f, g, o = np.split(self._w_ih @ obs + self._w_hh @ self._h + self._b, 4)
        self._c = _sigmoid(f) * self._c + _sigmoid(i) * np.tanh(g)
        self._h = _sigmoid(o) * np.tanh(self._c)
        hidden = self._w1 @ self._h + self._b1
        hidden = np.where(hidden > 0, hidden, np.expm1(hidden))
        return (self._w2 @ hidden + self._b2).astype(np.float32)
