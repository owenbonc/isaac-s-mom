# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""The MuJoCo scene: a walking Unitree G1, a table with a big box, and a bin across the room.

The G1 keeps its floating base; its legs run Unitree's walking policy (`walk.py`) and its arms are joint-position
controlled, so nothing moves it that a real robot couldn't.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from isaacs_mom.sim.mujoco.task import BIN_INNER_HALF, TaskConfig
from isaacs_mom.sim.mujoco.walk import configure_legs

G1_XML = Path(__file__).parent / "assets" / "unitree_g1" / "g1.xml"

# Unitree's standing pose, which is also the walking policy's default stance.
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
BOX_X = TABLE_CENTER[0] - 0.06
BIN_WALL = 0.02
BIN_FLOOR_Z = 0.30
BIN_RIM_Z = 0.50  # below the hanging hands, so the arms clear it at rest
TABLE_CLEARANCE = 0.45  # metres between the table top and the bin's walls, so the robot can walk between them
# Where the robot stands once it has backed away from the table with the box, and how far the bin must be from there,
# so it can always back off and turn before it walks.
BACKED_OFF_XY = (-0.30, 0.0)
BACKED_OFF_CLEARANCE = 0.60
LAYOUT_TRIES = 1000  # a task whose bin can't be placed within this many draws is misconfigured
HEAD_SIZE = (320, 240)
THIRD_SIZE = (640, 480)


@dataclass(frozen=True)
class Layout:
    """Where the box and bin start. A pure function of the seed and the task (SPEC §3.1)."""

    box_xy: tuple[float, float]
    bin_xy: tuple[float, float]
    bin_yaw: float  # the bin faces where the robot stood, so a box carried straight at it fits


def box_xy(rng: np.random.Generator, task: TaskConfig) -> tuple[float, float]:
    return float(BOX_X + rng.uniform(-0.02, 0.02)), float(rng.uniform(-task.box_offset_m, task.box_offset_m))


def bin_pose(
    rng: np.random.Generator, task: TaskConfig, around: tuple[float, float] = (0.0, 0.0)
) -> tuple[tuple[float, float], float]:
    """A bin the task's distance from `around`, facing it, at one of the task's bearings, and clear of the table."""
    for _ in range(LAYOUT_TRIES):
        bearing = np.deg2rad(rng.uniform(*task.bin_bearing_deg))
        dist = rng.uniform(*task.bin_distance_m)
        xy = (around[0] + dist * np.cos(bearing), around[1] + dist * np.sin(bearing))
        if table_clearance(xy) > BIN_INNER_HALF + BIN_WALL + TABLE_CLEARANCE:
            return (float(xy[0]), float(xy[1])), float(bearing)
    raise ValueError(f"task {task.name!r}: no bin pose clears the table in {LAYOUT_TRIES} draws")


def table_clearance(xy: tuple[float, float]) -> float:
    """Distance on the floor from `xy` to the table top's edge (0 inside it)."""
    dx = max(abs(xy[0] - TABLE_CENTER[0]) - TABLE_HALF[0], 0.0)
    dy = max(abs(xy[1] - TABLE_CENTER[1]) - TABLE_HALF[1], 0.0)
    return float(np.hypot(dx, dy))


def layout(seed: int, task: TaskConfig) -> Layout:
    rng = np.random.default_rng(seed)
    box = box_xy(rng, task)
    for _ in range(LAYOUT_TRIES):
        bin_xy, bin_yaw = bin_pose(rng, task)
        if np.hypot(bin_xy[0] - BACKED_OFF_XY[0], bin_xy[1] - BACKED_OFF_XY[1]) > BACKED_OFF_CLEARANCE:
            return Layout(box_xy=box, bin_xy=bin_xy, bin_yaw=bin_yaw)
    raise ValueError(f"task {task.name!r}: no bin pose leaves room to back off from the table in {LAYOUT_TRIES} draws")


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
    # Top of the face, tilted 45 degrees down with a wide lens: the hands at the box, and a bin 1-1.4 m ahead above
    # the carried box or, once close, below it.
    tilt = np.deg2rad(45)
    spec.body("torso_link").add_camera(
        name="head",
        pos=[0.10, 0.0, 0.45],
        xyaxes=[0, -1, 0, np.sin(tilt), 0, np.cos(tilt)],
        fovy=100.0,
    )


def _add_world(spec: mujoco.MjSpec, task: TaskConfig) -> None:
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

    box = world.add_body(name="box", pos=[BOX_X, 0.0, TABLE_TOP_Z + task.box_half[2]])
    box.add_freejoint(name="box_free")
    box.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(task.box_half),
        mass=task.box_mass_kg,
        rgba=[0.80, 0.33, 0.18, 1],
        friction=[1.5, 0.02, 0.002],
        condim=4,
    )

    # High over the robot's front-right shoulder, wide enough for the table and a bin anywhere around the robot.
    eye, target = np.array([2.3, -2.3, 2.9]), np.array([-0.25, 0.1, 0.3])
    world.add_camera(name="third_person", pos=eye, xyaxes=_look_at(eye, target), fovy=55.0)


def build_model(task: TaskConfig) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    # Menagerie's "stand" keyframe has no room for the scene's extra joints.
    for key in list(spec.keys):
        spec.delete(key)
    spec.option.impratio = 10  # firmer friction cones, so the hugged box doesn't creep out of the hands
    configure_legs(spec)
    _grip_feet(spec)
    _add_hands(spec)
    _add_head_camera(spec)
    _add_world(spec, task)
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
