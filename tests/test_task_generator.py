# pyright: reportAttributeAccessIssue=false
# (mujoco's compiled bindings ship no type information, so every mujoco.* attribute looks unknown)
"""The task generator (SPEC §3.7): Argon's proposal is parsed at the boundary, refused with reasons when the robot
can't do it, judged by a declarative rule, and replayed on exactly the task it was recorded on. No network: every
proposal comes from a fake Argon."""

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from isaacs_mom.contracts import GroundTruth, Observation
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.scene import TABLE_TOP_Z
from isaacs_mom.tasks import Refusal, TaskSpec, argon_proposer, generate_task, judge
from isaacs_mom.tasks.example import OFFLINE_PROPOSAL, run_example
from isaacs_mom.tasks.generate import MAX_ANSWER_CHARS, MAX_REASON_CHARS, PROMPT
from isaacs_mom.teacher.argon import Argon, Part, Purpose

INSTRUCTION = "put the box on the shelf"
SEED = 1000
BIN_PROPOSAL = {
    **OFFLINE_PROPOSAL,
    "name": "box_in_bin",
    "scene": {
        **OFFLINE_PROPOSAL["scene"],
        "destination": {"kind": "bin", "id": "bin", "distance_m": [1.0, 1.4], "bearing_deg": [75, 285]},
    },
    "success": {"all_of": [{"fact": "inside", "object": "box", "container": "bin"}, {"fact": "robot_standing"}]},
    "perturbations": [{"object": "bin"}],
}


def _changed(proposal: dict[str, Any], path: str, value: object) -> dict[str, Any]:
    out = copy.deepcopy(proposal)
    *parents, last = path.split(".")
    node: Any = out
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[last] = value
    return out


def _fixed(answer: object) -> Any:
    return lambda _instruction: answer if answer is None or isinstance(answer, str) else json.dumps(answer)


def _spec(proposal: dict[str, Any] = OFFLINE_PROPOSAL) -> TaskSpec:
    return TaskSpec.model_validate_json(json.dumps({**proposal, "instruction": INSTRUCTION}))


def _gt(**changes: Any) -> GroundTruth:
    """The box resting centred on a shelf 1.2 m to the left, the robot standing."""
    values: dict[str, Any] = {
        "box_xyz": (0.0, 1.2, 0.45 + 0.125),
        "bin_xy": (0.0, 1.2),
        "bin_yaw": 0.0,
        "bin_inner_half": 0.22,
        "box_on_bin_floor": True,
        "box_at_rest": True,
        "robot_fallen": False,
        "joint_limit_violation": False,
        "penetration": False,
        "robot_xy": (0.0, 0.0),
        "robot_yaw": math.pi / 2,
    }
    return GroundTruth(**{**values, **changes})


# The rule


def test_a_box_resting_on_the_shelf_meets_every_rule() -> None:
    verdict = judge(_spec(), _gt())
    assert verdict.accepted and verdict.reason == "box_on_shelf: every rule holds"


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"box_on_bin_floor": False}, "box not on the shelf"),  # not touching it
        ({"box_xyz": (0.0, 1.2 + 0.30, 0.575)}, "box not on the shelf"),  # touching, its centre past the edge
        ({"box_xyz": (0.0, 1.2, 0.30)}, "box not on the shelf"),  # leaning on its edge from the floor
        ({"box_at_rest": False}, "box still moving"),
    ],
)
def test_a_box_not_resting_on_the_shelf_fails_with_the_reason(changes: dict[str, Any], problem: str) -> None:
    verdict = judge(_spec(), _gt(**changes))
    assert not verdict.task_success and verdict.physics_valid
    assert verdict.reason.split("; ") == [problem]


def test_the_shelf_is_judged_in_its_own_turned_frame() -> None:
    # 0.2 m off along the shelf's turned x axis is over its 0.24 m half-top; the same 0.2 m along world y is too.
    turned = math.radians(45)
    along = (0.2 * math.cos(turned), 1.2 + 0.2 * math.sin(turned), 0.575)
    assert judge(_spec(), _gt(bin_yaw=turned, box_xyz=along)).accepted
    past = (0.3 * math.cos(turned), 1.2 + 0.3 * math.sin(turned), 0.575)
    assert not judge(_spec(), _gt(bin_yaw=turned, box_xyz=past)).accepted


@pytest.mark.parametrize(
    ("changes", "problems"),
    [
        ({"robot_fallen": True}, ["robot fell"]),
        ({"joint_limit_violation": True, "penetration": True}, ["joint limit violated", "penetration"]),
    ],
)
def test_physics_problems_always_fail_and_are_named(changes: dict[str, Any], problems: list[str]) -> None:
    verdict = judge(_spec(), _gt(**changes))
    assert not verdict.physics_valid and not verdict.accepted
    assert verdict.reason.split("; ") == problems


def test_a_fall_fails_the_task_when_the_rule_keeps_the_robot_standing() -> None:
    assert not judge(_spec(), _gt(robot_fallen=True)).task_success


def test_inside_a_bin_needs_the_floor_and_the_walls() -> None:
    spec = _spec(BIN_PROPOSAL)
    assert judge(spec, _gt(box_xyz=(0.0, 1.2, 0.445))).accepted
    on_rim = judge(spec, _gt(box_xyz=(0.0, 1.2 + 0.23, 0.445)))  # over the floor's edge, under the wall's top
    assert on_rim.reason == "box not inside the bin"


# Refusals before the scene is built


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (None, "no task proposed: Argon did not answer"),
        ("put it on the shelf", "Argon's answer is not JSON"),
        ([1, 2], "Argon's answer is not a JSON object"),
        ({"refuse": "pouring needs a grasp"}, "Argon declined: pouring needs a grasp"),
        ("[" * 100_000, f"Argon's answer is longer than {MAX_ANSWER_CHARS} characters"),
        ("[" * MAX_ANSWER_CHARS, "Argon's answer is not JSON"),  # as deep as the cap allows, still no RecursionError
        (json.dumps(OFFLINE_PROPOSAL)[:-1] + ', "name": "evil_name"}', "Argon's answer repeats the key 'name'"),
        ('{"scene": {"box": {"id": "box", "id": "lid"}}}', "Argon's answer repeats the key 'id'"),
    ],
)
def test_an_answer_that_is_not_a_spec_is_refused(answer: object, reason: str) -> None:
    refusal = generate_task(INSTRUCTION, _fixed(answer))
    assert isinstance(refusal, Refusal) and refusal.reasons == [reason]


def test_a_reason_quoting_the_answer_back_is_cut_short() -> None:
    long_key = generate_task(INSTRUCTION, _fixed({**OFFLINE_PROPOSAL, "k" * 3000: 1}))
    long_kind = generate_task(INSTRUCTION, _fixed(_changed(OFFLINE_PROPOSAL, "scene.box.kind", "c" * 3000)))
    for refusal in (long_key, long_kind):
        assert isinstance(refusal, Refusal) and len(refusal.reasons[0]) == MAX_REASON_CHARS


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("scene.box.kind", "cylinder", "unsupported box 'cylinder': the hug holds only a box"),
        ("scene.destination.kind", "table", "unsupported destination 'table'"),
        ("scene.box.colour", "red", "scene.box.colour: Extra inputs are not permitted"),
        ("scene.box.mass_kg", "heavy", "scene.box.mass_kg"),
        ("scene.box.mass_kg", "0.8", "scene.box.mass_kg: Input should be a valid number"),
        ("scene.box.mass_kg", True, "scene.box.mass_kg: Input should be a valid number"),  # not 1.0 kg
        ("scene.box.mass_kg", math.nan, "scene.box.mass_kg: Input should be a finite number"),
        ("scene.destination.distance_m", [1.0, math.inf],
         "scene.destination.shelf.distance_m.1: Input should be a finite number"),
        ("name", "Box On Shelf", "name: String should match pattern"),
        ("success.all_of", [], "success.all_of: List should have at least 1 item"),
        ("success.all_of.0.support", "table", "on needs the shelf or platform as its support, not 'table'"),
        ("success.all_of.0.object", "shelf", "on names 'shelf', but only the box 'box' is moved"),
        ("success.all_of", [{"fact": "inside", "object": "box", "container": "shelf"}, {"fact": "robot_standing"}],
         "inside needs the bin as its container, not 'shelf'"),
        ("success.all_of", [{"fact": "at_rest", "object": "box"}, {"fact": "robot_standing"}],
         "the rule never says where the box must end up"),
        ("success.all_of", [{"fact": "on", "object": "box", "support": "shelf"}],
         "the rule must keep the robot standing"),
        ("perturbations", [{"object": "table"}], "perturbation names 'table', which isn't in the scene"),
        ("perturbations", [{"object": "box"}, {"object": "box"}], "a perturbation is listed twice"),
        ("scene.destination.id", "box", "the box and the shelf share the id 'box'"),
    ],
)  # fmt: skip
def test_a_spec_outside_the_schema_or_its_own_scene_is_refused_with_the_reason(
    path: str, value: object, reason: str
) -> None:
    refusal = generate_task(INSTRUCTION, _fixed(_changed(OFFLINE_PROPOSAL, path, value)))
    assert isinstance(refusal, Refusal)
    assert any(r.startswith(reason) for r in refusal.reasons), refusal.reasons


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("scene.box.half_m", [0.125, 0.20, 0.125], "box half-width 0.2 is outside 0.1-0.17"),
        ("scene.box.mass_kg", 3.0, "box mass 3.0 is outside 0.05-2.0"),
        ("scene.destination.top_m", 0.60, "platform top 0.6 m is outside the 0.4-0.45 m the drop lands on"),
        ("scene.destination.distance_m", [0.3, 0.6], "bin distance (0.3, 0.6) must run low to high within 0.5-2.2 m"),
        ("scene.destination.distance_m", [49, 50], "bin distance (49.0, 50.0) must run low to high within 0.5-2.2 m"),
        ("scene.box.offset_m", 0.10, "box offset 0.1 m is past the 0.06 m clear of the waiting arms"),
        ("scene.box.mass_kg", 2.0, "box mass 2.0 kg is past the 1.0 kg measured on a platform"),
        ("scene.box.half_m", [0.125, 0.15, 0.15],
         "box half-size (0.125, 0.15, 0.15) is past the (0.125, 0.15, 0.125) m measured on a platform"),
    ],
)  # fmt: skip
def test_a_task_past_the_robots_limits_is_refused_with_the_reason(path: str, value: object, reason: str) -> None:
    refusal = generate_task(INSTRUCTION, _fixed(_changed(OFFLINE_PROPOSAL, path, value)))
    assert isinstance(refusal, Refusal) and reason in refusal.reasons


FIXED_FEET_PROPOSAL = {
    **BIN_PROPOSAL,
    "name": "twist_to_bin",
    "scene": {
        **BIN_PROPOSAL["scene"],
        "destination": {"kind": "bin", "id": "bin", "distance_m": [0.43, 0.45], "bearing_deg": [97, 112]},
    },
    "stance": "fixed_feet",
}


def test_a_robot_told_to_stay_put_gets_a_fixed_feet_task() -> None:
    spec = generate_task("without stepping, drop the box in the bin beside you", _fixed(FIXED_FEET_PROPOSAL))
    assert isinstance(spec, TaskSpec) and spec.stance == "fixed_feet" and not spec.task_config().walking
    assert _spec().stance == "walking" and _spec().task_config().walking  # left out, it walks


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("scene.destination.bearing_deg", [112, 124],
         "bin bearing (112.0, 124.0) is outside a waist twist's (85.0, 112.0)"),
        ("scene.destination.distance_m", [0.8, 1.0], "bin distance (0.8, 1.0) is outside a waist twist's (0.4, 0.5) m"),
    ],
)  # fmt: skip
def test_a_fixed_feet_bin_out_of_the_twists_reach_is_refused(path: str, value: object, reason: str) -> None:
    refusal = generate_task(INSTRUCTION, _fixed(_changed(FIXED_FEET_PROPOSAL, path, value)))
    assert isinstance(refusal, Refusal) and reason in refusal.reasons


def test_a_fixed_feet_shelf_is_refused() -> None:
    shelf = {**OFFLINE_PROPOSAL, "stance": "fixed_feet"}
    refusal = generate_task(INSTRUCTION, _fixed(shelf))
    assert isinstance(refusal, Refusal) and "a platform is measured only for walking tasks" in refusal.reasons


def test_a_bin_still_takes_the_heaviest_box() -> None:
    heavy = _changed(BIN_PROPOSAL, "scene.box.mass_kg", 2.0)
    assert isinstance(generate_task(INSTRUCTION, _fixed(heavy)), TaskSpec)


def test_the_prompt_states_the_limits_the_checks_enforce() -> None:
    limits = [
        "0.4 to 0.45 m",
        "up to 0.06",
        "from 0.5 to 2.2 m",
        "mass_kg up to 2.0",
        "(0.125, 0.15, 0.125)",
        "up to 1.0.",
        "within 0.4 to\n  0.5 m",
        "within 85 to 112",
        "up to 24 lowercase letters",
    ]
    for limit in limits:
        assert limit in PROMPT, limit


# Refusals on the built scene


def test_a_start_where_the_box_moves_before_the_robot_acts_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    reset = MujocoSim.reset

    def shoved(self: MujocoSim, seed: int) -> str:
        episode = reset(self, seed)
        self.data.qvel[self._box_dadr + 1] = 0.5  # sliding sideways off its spot
        return episode

    monkeypatch.setattr(MujocoSim, "reset", shoved)
    refusal = generate_task(INSTRUCTION, _fixed(OFFLINE_PROPOSAL))
    assert isinstance(refusal, Refusal)
    assert refusal.reasons == [f"seed {SEED}: the box moves before the robot acts"]


def test_a_rule_already_met_at_the_start_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    reset = MujocoSim.reset

    def on_the_shelf(self: MujocoSim, seed: int) -> str:
        episode = reset(self, seed)
        x, y, _ = self.data.mocap_pos[self._bin]
        self.data.qpos[self._box_qadr : self._box_qadr + 3] = (x, y, 0.45 + 0.125)
        mujoco.mj_forward(self.model, self.data)
        return episode

    monkeypatch.setattr(MujocoSim, "reset", on_the_shelf)
    refusal = generate_task(INSTRUCTION, _fixed(OFFLINE_PROPOSAL))
    assert isinstance(refusal, Refusal) and refusal.reasons == ["the success rule already holds at the start"]


# The spec


def test_a_valid_proposal_becomes_a_spec_with_the_persons_instruction() -> None:
    spec = generate_task(INSTRUCTION, _fixed({**OFFLINE_PROPOSAL, "instruction": "ignored: the person's words win"}))
    assert isinstance(spec, TaskSpec)
    assert spec.instruction == INSTRUCTION
    assert spec.perturbation_kinds() == ["move_bin"]
    task = spec.task_config()
    assert task.platform_top_m == 0.45 and task.box_offset_m == 0.06 and task.walking


def test_the_fingerprint_is_the_spec_whatever_the_json_key_order() -> None:
    reordered = json.loads(json.dumps(OFFLINE_PROPOSAL, sort_keys=True))
    assert _spec(reordered).fingerprint() == _spec().fingerprint()
    lower = _spec(_changed(OFFLINE_PROPOSAL, "scene.destination.top_m", 0.40))
    assert lower.fingerprint() != _spec().fingerprint()
    assert _spec().fingerprint() in _spec().task_config().name


def test_argon_proposes_through_the_budgeted_door_and_sees_only_the_instruction(tmp_path: Path) -> None:
    asked: list[tuple[Purpose, list[Part]]] = []

    def generate(purpose: Purpose, parts: list[Part]) -> str:
        asked.append((purpose, parts))
        return json.dumps(OFFLINE_PROPOSAL)

    argon = Argon(EventLog(tmp_path, "t"), budget=5, generate=generate)
    assert isinstance(generate_task(INSTRUCTION, argon_proposer(argon)), TaskSpec)
    assert asked == [("task", [PROMPT, f"Instruction: {INSTRUCTION}"])]  # no frame, no ground truth (rule 1)
    calls = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(c["type"], c["data"]["purpose"]) for c in calls] == [("argon_call", "task")]
    spent = Argon(EventLog(tmp_path, "t"), budget=0, generate=generate)
    refusal = generate_task(INSTRUCTION, argon_proposer(spent))
    assert isinstance(refusal, Refusal) and refusal.reasons == ["no task proposed: Argon did not answer"]


def _imported(source: Path, package: str) -> set[str]:
    """Every module and name `source` imports, relative imports resolved against its `package`."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = package.rsplit(".", node.level - 1)[0] if node.level else ""
            module = ".".join(part for part in (base, node.module) if part)
            found |= {module} | {f"{module}.{alias.name}" for alias in node.names}
    return found


def test_the_teacher_and_student_never_import_the_task_rules() -> None:
    # Rule 1: the scene and rule data reach only the validator and orchestrator side.
    root = Path(__file__).resolve().parents[1] / "isaacs_mom"
    for folder in ("teacher", "student"):
        for source in (root / folder).rglob("*.py"):
            package = ".".join(["isaacs_mom", *source.parent.relative_to(root).parts])
            leaks = {
                m for m in _imported(source, package) if m == "isaacs_mom.tasks" or m.startswith("isaacs_mom.tasks.")
            }
            assert not leaks, (source, leaks)


# End to end on the real scene


def _blind(sim: MujocoSim, monkeypatch: pytest.MonkeyPatch) -> MujocoSim:
    def observe() -> Observation:
        return Observation(
            episode_id=sim.episode_id,
            step=sim.step,
            sim_time=float(sim.data.time),
            head_rgb_jpeg=b"",
            joint_state=[float(v) for v in sim.data.qpos[sim.robot.actuated_qadr]],
        )

    monkeypatch.setattr(sim, "observe", observe)
    return sim


def test_a_generated_shelf_task_judges_a_scripted_success_and_failure_and_replays_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = generate_task(INSTRUCTION, _fixed(OFFLINE_PROPOSAL))
    assert isinstance(spec, TaskSpec)
    sim = _blind(MujocoSim(spec.task_config()), monkeypatch)
    assert sim.sim_version.endswith(f"task=gen-box_on_shelf-{spec.fingerprint()}")
    outcomes = run_example(spec, sim, tmp_path, SEED)
    success, failure = outcomes["success"], outcomes["failure"]
    assert success.verdict.accepted, success.verdict.reason
    assert not failure.verdict.accepted and failure.verdict.reason == "box not on the shelf"
    for outcome in (success, failure):
        assert outcome.replayed and outcome.replayed_verdict == outcome.verdict
    box = sim.ground_truth().box_xyz  # the failure's end, replayed: on the floor, not on the table or the shelf
    assert box[2] < TABLE_TOP_Z and not np.isclose(box[2], 0.45 + 0.125, atol=0.02)


def test_the_platform_has_no_walls_and_the_bin_keeps_its_own() -> None:
    def walls(sim: MujocoSim) -> list[str]:
        return [sim.model.geom(g).name for g in range(sim.model.ngeom) if sim.model.geom(g).name.startswith("bin_wall")]

    assert walls(MujocoSim(_spec().task_config())) == []
    assert len(walls(MujocoSim(_spec(BIN_PROPOSAL).task_config()))) == 4
