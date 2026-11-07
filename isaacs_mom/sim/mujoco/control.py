# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""Arm IK and smooth joint-position motions for the standing G1."""

from collections.abc import Callable

import mujoco
import numpy as np

ARM_JOINTS = {
    side: [
        f"{side}_shoulder_pitch_joint",
        f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint",
        f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    ]
    for side in ("left", "right")
}
CONTROL_DT = 0.02  # 50 Hz targets; physics runs at the model timestep underneath
IK_DAMPING = 1e-2
IK_POSTURE_WEIGHT = 0.05
IK_TOLERANCE = 0.01  # metres; a target further than this from the solved hand is out of reach


class Robot:
    """Index maps between joints and actuators, plus a scratch copy of the state for IK."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.joint_actuator = {model.joint(model.actuator_trnid[i, 0]).name: i for i in range(model.nu)}
        self.actuated_qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
        self.arm_qadr = {side: [model.jnt_qposadr[model.joint(j).id] for j in js] for side, js in ARM_JOINTS.items()}
        self.arm_dadr = {side: [model.jnt_dofadr[model.joint(j).id] for j in js] for side, js in ARM_JOINTS.items()}
        self.arm_act = {side: [self.joint_actuator[j] for j in js] for side, js in ARM_JOINTS.items()}
        self.arm_range = {side: model.jnt_range[[model.joint(j).id for j in js]] for side, js in ARM_JOINTS.items()}
        self.waist_yaw_act = self.joint_actuator["waist_yaw_joint"]
        self.waist_yaw_range = model.jnt_range[model.joint("waist_yaw_joint").id]
        self.pelvis = model.body("pelvis").id

    def to_world(self, data: mujoco.MjData, p: np.ndarray, yaw_offset: float = 0.0) -> np.ndarray:
        """A point in the pelvis heading frame (x forward, y left, z up from the floor), turned by `yaw_offset`."""
        x, y = data.xpos[self.pelvis][:2]
        rot = data.xmat[self.pelvis].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0]) + yaw_offset
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])

    def solve_arm(
        self, data: mujoco.MjData, ctrl: np.ndarray, side: str, target: np.ndarray, iters: int = 100
    ) -> tuple[np.ndarray, float]:
        """Joint targets, within limits, that bring `side`'s hand closest to world point `target`, and the miss
        distance. Every other joint is taken as commanded in `ctrl`."""
        m, d = self.model, self.scratch
        d.qpos[:] = data.qpos
        d.qpos[self.actuated_qadr] = ctrl
        qadr, dadr = self.arm_qadr[side], self.arm_dadr[side]
        lo, hi = self.arm_range[side].T
        posture = d.qpos[qadr].copy()
        site = m.site(f"{side}_hand").id
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = target - d.site_xpos[site]
            if np.linalg.norm(err) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jac, None, site)
            j = jac[:, dadr]
            # Damped least squares, with a pull back towards the starting posture in the null space.
            jjt = j @ j.T + IK_DAMPING * np.eye(3)
            dq = j.T @ np.linalg.solve(jjt, err)
            null = np.eye(len(dadr)) - j.T @ np.linalg.solve(jjt, j)
            dq += null @ (IK_POSTURE_WEIGHT * (posture - d.qpos[qadr]))
            d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
        mujoco.mj_kinematics(m, d)
        return d.qpos[qadr].copy(), float(np.linalg.norm(target - d.site_xpos[site]))


def move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    goal: np.ndarray,
    duration: float,
    on_tick: Callable[[], None],
) -> None:
    """Ease the actuator targets from the current command to `goal` over `duration` seconds of sim time."""
    start = data.ctrl.copy()
    ticks = max(1, round(duration / CONTROL_DT))
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    for k in range(1, ticks + 1):
        a = k / ticks
        data.ctrl[:] = start + (goal - start) * (0.5 - 0.5 * np.cos(np.pi * a))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        on_tick()
