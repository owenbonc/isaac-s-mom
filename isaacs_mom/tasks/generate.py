"""instruction -> TaskSpec | Refusal (SPEC §3.7).

Argon proposes the spec as JSON. Everything is checked here, in order: the JSON and its schema, the objects' kinds and
ids, the robot's limits, and then, on the real scene, that the start is stable and the rule doesn't already hold.
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pydantic import ValidationError

from isaacs_mom.contracts import SkillCall
from isaacs_mom.seeds import DEV_SEED_START
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.scene import TABLE_TOP_Z
from isaacs_mom.sim.mujoco.task import (
    BOX_HALF_DEPTH_RANGE,
    BOX_HALF_HEIGHT_RANGE,
    BOX_HALF_WIDTH_RANGE,
    MAX_BOX_MASS_KG,
    MIN_BIN_DISTANCE_M,
    PLATFORM_TOP_RANGE_M,
)
from isaacs_mom.tasks.rules import judge
from isaacs_mom.tasks.spec import AtRest, Bin, Inside, On, Platform, Refusal, RobotStanding, TaskSpec
from isaacs_mom.teacher.argon import Argon

Propose = Callable[[str], str | None]  # the instruction -> Argon's JSON answer, or None when it gave none

SUPPORTED = {
    "box": (("box",), "the hug holds only a box"),
    "destination": (("bin", "shelf", "platform"), "a destination is a bin, shelf or platform"),
}
CHECK_SEEDS = (DEV_SEED_START, DEV_SEED_START + 1, DEV_SEED_START + 2)  # development seeds: held-out ones stay sealed
SETTLE_STEPS = 5  # reobserves of 0.2 s: a second for the start to show itself stable
START_DRIFT_M = 0.005  # a stable start leaves the box where it began
# Further off-centre, the waiting arms nudge the box as the robot steps in place (5-14 mm on 15 of 50 seeds at 0.10 m;
# none of 50 at 0.06 m), so a generated task starts it no further out.
MAX_START_OFFSET_M = 0.06
MAX_REASONS = 8

PROMPT = f"""You turn a person's instruction for a Unitree G1 humanoid into a task specification, as JSON only.

The robot stands at the origin facing +x, with a table ahead of it. It hugs one box off the table with both arms, lifts
it, walks it to a destination and lets go. It can do nothing else, so a task is always: this box, to this destination.

Answer with exactly one JSON object of this shape (an example, not the answer):
{{"name": "box_in_bin",
 "scene": {{"box": {{"kind": "box", "id": "box", "half_m": [0.125, 0.15, 0.125], "mass_kg": 0.8, "offset_m": 0.05}},
           "destination": {{"kind": "bin", "id": "bin", "distance_m": [1.0, 1.4], "bearing_deg": [75, 285]}}}},
 "success": {{"all_of": [{{"fact": "inside", "object": "box", "container": "bin"}},
                        {{"fact": "at_rest", "object": "box"}}, {{"fact": "robot_standing"}}]}},
 "perturbations": [{{"object": "bin"}}]}}

- name and ids: lowercase letters, digits and underscores, starting with a letter.
- box: half_m is half its depth, width and height, within {BOX_HALF_DEPTH_RANGE}, {BOX_HALF_WIDTH_RANGE} and
  {BOX_HALF_HEIGHT_RANGE} m; mass_kg up to {MAX_BOX_MASS_KG}; offset_m, how far off the table's centre line it may
  start, up to {MAX_START_OFFSET_M}.
- destination: kind "bin" (walls, a floor), or "shelf" or "platform" (a flat top with no walls, with "top_m" from
  {PLATFORM_TOP_RANGE_M[0]} to {PLATFORM_TOP_RANGE_M[1]} m). distance_m from {MIN_BIN_DISTANCE_M} m up, low to high;
  bearing_deg anticlockwise from straight ahead, low to high, between 0 and 360 (the table is at 0).
- success.all_of: facts that must all hold at the end. "inside" (object, container: a bin), "on" (object, support: a
  shelf or platform), "at_rest" (object), "robot_standing". Say where the box must end up.
- perturbations: the objects that may be moved mid-run (the box, the destination, or neither).
- If the instruction needs anything else (another object, a grasp, a pour, a stack), answer {{"refuse": "<why>"}}.
"""


def argon_proposer(argon: Argon) -> Propose:
    """Argon through the run's budgeted door (rule 5). It sees only the instruction: no frame, no ground truth."""

    def propose(instruction: str) -> str | None:
        return argon.ask("task", [PROMPT, f"Instruction: {instruction}"])

    return propose


def generate_task(instruction: str, propose: Propose, check_seeds: Sequence[int] = CHECK_SEEDS) -> TaskSpec | Refusal:
    def refuse(*reasons: str) -> Refusal:
        return Refusal(instruction=instruction, reasons=list(reasons[:MAX_REASONS]))

    answer = propose(instruction)
    if answer is None:
        return refuse("no task proposed: Argon did not answer")
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return refuse("Argon's answer is not JSON")
    if not isinstance(data, dict):
        return refuse("Argon's answer is not a JSON object")
    if isinstance(data.get("refuse"), str):
        return refuse(f"Argon declined: {data['refuse'][:300]}")
    if unsupported := _unsupported_kinds(data):
        return refuse(*unsupported)
    try:
        spec = TaskSpec.model_validate({**data, "instruction": instruction})
    except ValidationError as error:
        return refuse(*(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()))
    if problems := _inconsistencies(spec):
        return refuse(*problems)
    try:
        task = spec.task_config()
    except ValueError as error:
        return refuse(*str(error).split(": ", 1)[-1].split("; "))
    if spec.scene.box.offset_m > MAX_START_OFFSET_M:
        return refuse(
            f"box offset {spec.scene.box.offset_m} m is past the {MAX_START_OFFSET_M} m clear of the waiting arms"
        )
    return _check_on_scene(spec, MujocoSim(task), check_seeds, refuse) or spec


def _unsupported_kinds(data: dict[str, Any]) -> list[str]:
    scene = data.get("scene")
    if not isinstance(scene, dict):
        return []  # the schema check names what's missing
    found = []
    for slot, (kinds, why) in SUPPORTED.items():
        part = scene.get(slot)
        kind = part.get("kind") if isinstance(part, dict) else None
        if kind is not None and kind not in kinds:
            found.append(f"unsupported {slot} {kind!r}: {why}")
    return found


def _inconsistencies(spec: TaskSpec) -> list[str]:
    box, dest = spec.scene.box, spec.scene.destination
    problems = []
    if box.id == dest.id:
        problems.append(f"the box and the {dest.kind} share the id {box.id!r}")
    for fact in spec.success.all_of:
        match fact:
            case Inside() | On() | AtRest() if fact.object != box.id:
                problems.append(f"{fact.fact} names {fact.object!r}, but only the box {box.id!r} is moved")
            case Inside() if not (fact.container == dest.id and isinstance(dest, Bin)):
                problems.append(f"inside needs the bin as its container, not {fact.container!r}")
            case On() if not (fact.support == dest.id and isinstance(dest, Platform)):
                problems.append(f"on needs the shelf or platform as its support, not {fact.support!r}")
            case _:
                pass
    if not any(isinstance(f, Inside | On) for f in spec.success.all_of):
        problems.append("the rule never says where the box must end up")
    if not any(isinstance(f, RobotStanding) for f in spec.success.all_of):
        problems.append("the rule must keep the robot standing")
    movable = [move.object for move in spec.perturbations]
    for name in movable:
        if name not in (box.id, dest.id):
            problems.append(f"perturbation names {name!r}, which isn't in the scene")
    if len(set(movable)) < len(movable):
        problems.append("a perturbation is listed twice")
    return problems


def _check_on_scene(
    spec: TaskSpec, sim: MujocoSim, seeds: Sequence[int], refuse: Callable[..., Refusal]
) -> Refusal | None:
    """On the built scene: every check seed lays out, starts stable, and doesn't already meet the rule."""
    resting_z = TABLE_TOP_Z + spec.scene.box.half_m[2]
    for seed in seeds:
        try:
            sim.reset(seed)
        except ValueError as error:  # a placement the layout can't satisfy
            return refuse(str(error))
        start = np.array(sim.ground_truth().box_xyz)
        for _ in range(SETTLE_STEPS):
            sim.execute(SkillCall(skill="reobserve"))
        gt = sim.ground_truth()
        if judge(spec, gt).accepted:
            return refuse("the success rule already holds at the start")
        unstable = [
            why
            for flagged, why in [
                (
                    float(np.linalg.norm(np.array(gt.box_xyz) - start)) > START_DRIFT_M,
                    "the box moves before the robot acts",
                ),
                (abs(gt.box_xyz[2] - resting_z) > START_DRIFT_M, "the box doesn't start resting on the table"),
                (gt.robot_fallen, "the robot falls before it acts"),
                (gt.penetration, "objects overlap at the start"),
            ]
            if flagged
        ]
        if unstable:
            return refuse(*(f"seed {seed}: {why}" for why in unstable))
    return None
