# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""The MuJoCo scene: a free-standing Unitree G1, a table with a big box, and a bin beside the robot.

The G1 keeps its floating base and balances on its own feet under joint-position control; it reaches the bin by
twisting at the waist, so nothing moves it that a real robot couldn't.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

G1_XML = Path(__file__).parent / "assets" / "unitree_g1" / "g1.xml"

# Unitree's own standing pose: soft knees keep both feet flat, which a straight-legged G1 can't.
STAND_POSE = {
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.28,
}

TABLE_TOP_Z = 0.72
TABLE_CENTER = (0.40, 0.0)
TABLE_HALF = (0.2, 0.25)
BOX_HALF = (0.125, 0.15, 0.125)  # x depth, y width, z height
BOX_MASS = 0.8
BOX_X = TABLE_CENTER[0] - 0.06
BIN_INNER_HALF = 0.22  # wider than the box's 39 cm diagonal, so a tumbling box can't wedge across it
BIN_WALL = 0.02
BIN_FLOOR_Z = 0.30
BIN_RIM_Z = 0.50  # below the hanging hands, so the arms clear it at rest
HEAD_SIZE = (320, 240)
THIRD_SIZE = (640, 480)


@dataclass(frozen=True)
class Layout:
    """Where the box and bin start. A pure function of the seed (SPEC §3.1)."""

    box_xy: tuple[float, float]
    bin_xy: tuple[float, float]
    bin_yaw: float  # the bin faces the robot, so a box carried straight at it fits


def box_xy(rng: np.random.Generator) -> tuple[float, float]:
    return float(BOX_X + rng.uniform(-0.02, 0.02)), float(rng.uniform(-0.1, 0.1))


def bin_pose(rng: np.random.Generator) -> tuple[tuple[float, float], float]:
    # On the robot's left, clear of its legs and the table; a bigger twist with the box held out tips it over.
    angle = rng.uniform(np.deg2rad(97), np.deg2rad(112))
    dist = rng.uniform(0.43, 0.45)
    return (float(dist * np.cos(angle)), float(dist * np.sin(angle))), float(angle)


def layout(seed: int) -> Layout:
    rng = np.random.default_rng(seed)
    box = box_xy(rng)
    bin_xy, bin_yaw = bin_pose(rng)
    return Layout(box_xy=box, bin_xy=bin_xy, bin_yaw=bin_yaw)


def _look_at(pos: np.ndarray, target: np.ndarray) -> list[float]:
    """xyaxes for a camera at `pos` looking at `target` with world +z up."""
    forward = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right, *up]


def _grip_feet(spec: mujoco.MjSpec) -> None:
    for geom in spec.geoms:
        if geom.parent.name.endswith("ankle_roll_link"):
            geom.friction = [1.2, 0.02, 0.002]


def _add_hands(spec: mujoco.MjSpec) -> None:
    # Menagerie's rubber hands are visual only; a padded box per hand gives the hug something to grip with.
    for side, sign in (("left", 1), ("right", -1)):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        wrist.add_site(name=f"{side}_hand", pos=[0.108, -sign * 0.008, 0.011])
        wrist.add_geom(
            name=f"{side}_hand_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.03, 0.065],
            pos=[0.108, -sign * 0.008, 0.011],
            group=3,
            friction=[1.5, 0.02, 0.002],
            condim=4,
            rgba=[0.2, 0.2, 0.2, 1],
        )


def _add_head_camera(spec: mujoco.MjSpec) -> None:
    # The G1 head is part of torso_link; mount at the face, tilted 50 degrees down at the table.
    tilt = np.deg2rad(50)
    spec.body("torso_link").add_camera(
        name="head",
        pos=[0.09, 0.0, 0.40],
        xyaxes=[0, -1, 0, np.sin(tilt), 0, np.cos(tilt)],
        fovy=75.0,
    )


def _add_world(spec: mujoco.MjSpec) -> None:
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.rgba.haze = [0.88, 0.91, 0.95, 1]
    spec.add_texture(
        name="floor",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.82, 0.82, 0.80],
        rgb2=[0.74, 0.74, 0.72],
        width=512,
        height=512,
    )
    spec.add_material(name="floor", textures=["", "floor"], texrepeat=[2, 2], texuniform=True, reflectance=0.05)
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=[0.88, 0.91, 0.95],
        rgb2=[0.88, 0.91, 0.95],
        width=64,
        height=384,
    )
    world = spec.worldbody
    world.add_light(
        pos=[1.0, -1.0, 3.0],
        dir=[-0.3, 0.3, -1.0],
        diffuse=[0.6, 0.6, 0.6],
        castshadow=True,
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[15, 15, 0.05], material="floor")

    wood = [0.62, 0.45, 0.30, 1]
    table = world.add_body(name="table", pos=[*TABLE_CENTER, 0.0])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[*TABLE_HALF, 0.02],
        pos=[0, 0, TABLE_TOP_Z - 0.02],
        rgba=wood,
        friction=[1.0, 0.01, 0.001],
    )
    leg_half = (TABLE_TOP_Z - 0.04) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.025, 0.025, leg_half],
                pos=[sx * (TABLE_HALF[0] - 0.03), sy * (TABLE_HALF[1] - 0.03), leg_half],
                rgba=wood,
                contype=0,
                conaffinity=0,
            )

    # A mocap body, so `reset` and `perturb` can place it anywhere without it being knocked about.
    green = [0.25, 0.62, 0.35, 1]
    bin_body = world.add_body(name="bin", mocap=True, pos=[0.0, 0.5, 0.0])
    outer = BIN_INNER_HALF + BIN_WALL
    bin_body.add_geom(
        name="bin_stand",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[outer, outer, BIN_FLOOR_Z / 2],  # as wide as the bin, so nothing can get under its floor
        pos=[0, 0, BIN_FLOOR_Z / 2],
        rgba=[0.35, 0.35, 0.38, 1],
    )
    wall_h = (BIN_RIM_Z - BIN_FLOOR_Z) / 2
    bin_body.add_geom(
        name="bin_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BIN_INNER_HALF, BIN_INNER_HALF, 0.01],  # inside the walls, so only a box in the bin can touch it
        pos=[0, 0, BIN_FLOOR_Z + 0.01],
        rgba=green,
    )
    for name, size, pos in [
        ("bin_wall_px", [BIN_WALL / 2, outer, wall_h], [outer - BIN_WALL / 2, 0, BIN_FLOOR_Z + wall_h]),
        ("bin_wall_nx", [BIN_WALL / 2, outer, wall_h], [-outer + BIN_WALL / 2, 0, BIN_FLOOR_Z + wall_h]),
        ("bin_wall_py", [outer, BIN_WALL / 2, wall_h], [0, outer - BIN_WALL / 2, BIN_FLOOR_Z + wall_h]),
        ("bin_wall_ny", [outer, BIN_WALL / 2, wall_h], [0, -outer + BIN_WALL / 2, BIN_FLOOR_Z + wall_h]),
    ]:
        bin_body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=green)

    box = world.add_body(name="box", pos=[BOX_X, 0.0, TABLE_TOP_Z + BOX_HALF[2]])
    box.add_freejoint(name="box_free")
    box.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BOX_HALF),
        mass=BOX_MASS,
        rgba=[0.80, 0.33, 0.18, 1],
        friction=[1.5, 0.02, 0.002],
        condim=4,
    )

    # From the robot's front-left, where the bin is.
    eye, target = np.array([1.3, 1.5, 1.6]), np.array([0.05, 0.15, 0.75])
    world.add_camera(name="third_person", pos=eye, xyaxes=_look_at(eye, target), fovy=50.0)


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    # Menagerie's "stand" keyframe has no room for the scene's extra joints.
    for key in list(spec.keys):
        spec.delete(key)
    _grip_feet(spec)
    _add_hands(spec)
    _add_head_camera(spec)
    _add_world(spec)
    return spec.compile()


def stand_qpos(model: mujoco.MjModel) -> np.ndarray:
    """The stand pose with the pelvis lowered until the feet just touch the floor."""
    data = mujoco.MjData(model)
    for name, value in STAND_POSE.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_kinematics(model, data)
    feet = [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.endswith("ankle_roll_link")
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    data.qpos[2] -= min(data.geom_xpos[g][2] - model.geom_rbound[g] for g in feet)
    return data.qpos.copy()
