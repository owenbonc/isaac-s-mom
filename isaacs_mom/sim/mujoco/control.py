# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""Arm IK and smooth joint-position motions for the walking G1."""

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
CONTROL_DT = 0.02  # 50 Hz targets, the walking policy's rate; physics runs at the model timestep underneath
IK_DAMPING = 1e-2
IK_POSTURE_WEIGHT = 0.05
IK_ROTATION_WEIGHT = 0.3  # metres per radian: how much hand orientation trades against position
IK_TOLERANCE = 0.01  # metres; a target further than this from the solved hand is out of reach


class Robot:
    """Index maps between joints and actuators, plus a scratch copy of the robot standing for IK."""

    def __init__(self, model: mujoco.MjModel, stand: np.ndarray) -> None:
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.stand = stand
        self.joint_actuator = {model.joint(model.actuator_trnid[i, 0]).name: i for i in range(model.nu)}
        self.actuated_qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
        self.arm_qadr = {side: [model.jnt_qposadr[model.joint(j).id] for j in js] for side, js in ARM_JOINTS.items()}
        self.arm_dadr = {side: [model.jnt_dofadr[model.joint(j).id] for j in js] for side, js in ARM_JOINTS.items()}
        self.arm_act = {side: [self.joint_actuator[j] for j in js] for side, js in ARM_JOINTS.items()}
        self.arm_range = {side: model.jnt_range[[model.joint(j).id for j in js]] for side, js in ARM_JOINTS.items()}
        self.pelvis = model.body("pelvis").id

    def solve_arm(
        self, ctrl: np.ndarray, side: str, target: np.ndarray, rotation: np.ndarray | None = None, iters: int = 100
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Joint targets, within limits, that bring `side`'s hand closest to `target`, the miss distance, and the
        hand's orientation there. `target` (and `rotation`, if given) are in the frame of the robot standing upright
        at the origin facing +x, so a solution holds wherever the robot has walked; arm joints start from `ctrl`."""
        m, d = self.model, self.scratch
        d.qpos[:] = self.stand
        arm = [self.joint_actuator[name] for name in ARM_JOINTS[side]]
        qadr, dadr = self.arm_qadr[side], self.arm_dadr[side]
        d.qpos[qadr] = ctrl[arm]
        lo, hi = self.arm_range[side].T
        posture = d.qpos[qadr].copy()
        site = m.site(f"{side}_hand").id
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = target - d.site_xpos[site]
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            j = jacp[:, dadr]
            if rotation is not None:
                # Small-angle orientation error, weighted so a radian costs as much as IK_ROTATION_WEIGHT metres.
                r = rotation @ d.site_xmat[site].reshape(3, 3).T
                spin = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
                err = np.concatenate([err, IK_ROTATION_WEIGHT * spin])
                j = np.vstack([j, IK_ROTATION_WEIGHT * jacr[:, dadr]])
            if np.linalg.norm(err) < 1e-3:
                break
            # Damped least squares, with a pull back towards the starting posture in the null space.
            jjt = j @ j.T + IK_DAMPING * np.eye(len(err))
            dq = j.T @ np.linalg.solve(jjt, err)
            null = np.eye(len(dadr)) - j.T @ np.linalg.solve(jjt, j)
            dq += null @ (IK_POSTURE_WEIGHT * (posture - d.qpos[qadr]))
            d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
        mujoco.mj_kinematics(m, d)
        miss = float(np.linalg.norm(target - d.site_xpos[site]))
        return d.qpos[qadr].copy(), miss, d.site_xmat[site].reshape(3, 3).copy()


def move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    goal: np.ndarray,
    duration: float,
    on_tick: Callable[[], None],
    drive: Callable[[], None],
) -> None:
    """Ease the actuator targets from the current command to `goal` over `duration` seconds of sim time. `drive` runs
    after the ease each tick, so the actuators it writes (the walking legs) follow it instead."""
    start = data.ctrl.copy()
    ticks = max(1, round(duration / CONTROL_DT))
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    for k in range(1, ticks + 1):
        a = k / ticks
        data.ctrl[:] = start + (goal - start) * (0.5 - 0.5 * np.cos(np.pi * a))
        drive()
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        on_tick()
