"""Teaching a generated task end to end (SPEC §3.6, §3.7): demonstrations on development seeds in a batch, each
judged by the spec's own rule, a training set of the accepted ones, one fine-tune, then the student alone on the
held-out seeds. No network and no GPU: scripted teachers, a fake fine-tune and a blank head camera stand in."""

import itertools
import json
import os
import re
import sys
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from google.genai import errors

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    Event,
    GroundTruth,
    Observation,
    SkillCall,
    TrainingExample,
    TrainingSetSummary,
    Validation,
)
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.seeds import SealedSeedError, heldout_seeds
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.preview import OraclePolicy
from isaacs_mom.sim.mujoco.scene import BIN_WALL
from isaacs_mom.sim.mujoco.task import BIN_INNER_HALF
from isaacs_mom.student import BASE_CHECKPOINT
from isaacs_mom.tasks import Refusal, TaskSpec, judge
from isaacs_mom.tasks.example import OFFLINE_PROPOSAL, DropShortPolicy
from isaacs_mom.tasks.generate import Propose, argon_proposer
from isaacs_mom.tasks.learn import (
    DEMO_SEED_START,
    Collected,
    ExamResult,
    LearnReport,
    argon_calls,
    collect,
    demonstrate,
    exam,
    finish,
    learn,
    main,
    start,
    summary,
)
from isaacs_mom.teacher.argon import Argon, ArgonUnavailable, MissingKey, Part, Purpose
from isaacs_mom.validator import replay, validate
from isaacs_mom.validator.rules import JOINT_LIMIT, PENETRATION, ROBOT_FELL, SEPARATOR

INSTRUCTION = "put the box on the shelf"
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
ON_THE_SHELF = Validation(task_success=True, physics_valid=True, reason="box_on_shelf: every rule holds")
OFF_THE_SHELF = Validation(task_success=False, physics_valid=True, reason="box not on the shelf")
METRICS = {"examples": 14, "epoch_losses": [1.9, 0.7]}


def _blank(sim: MujocoSim) -> Observation:
    return Observation(
        episode_id=sim.episode_id,
        step=sim.step,
        sim_time=float(sim.data.time),
        head_rgb_jpeg=b"",
        joint_state=[float(v) for v in sim.data.qpos[sim.robot.actuated_qadr]],
    )


@pytest.fixture(autouse=True)
def blank_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every sim built here renders nothing; its physics and ground truth are real."""
    monkeypatch.setattr(MujocoSim, "observe", _blank)


def blind_oracle(sim: MujocoSim) -> OraclePolicy:
    """The scripted teacher, blanking the camera itself: a spawned teacher process never sees the test's patch."""
    sim.observe = partial(_blank, sim)
    return OraclePolicy(sim)


class FakeFinetune:
    """Trains nothing and records each call; as the real fine-tune, a set with no examples keeps its checkpoint."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, int]] = []

    def __call__(self, training_set: Path, previous_checkpoint: str, round_: int) -> str:
        self.calls.append((training_set, previous_checkpoint, round_))
        if not training_set.read_text():
            return previous_checkpoint
        out = training_set.parent / "checkpoints" / f"ckpt-r{round_}"
        out.mkdir(parents=True)
        (out / "metrics.json").write_text(json.dumps(METRICS))
        return f"{training_set.parent.name}/ckpt-r{round_}"


class Stops:
    """A student that stops at once: what is under test is how the exam judges and counts, not its skill."""

    controller: Controller = "student"

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint: str | None = checkpoint

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return None


class Crashes(Stops):
    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        raise RuntimeError("the student's adapter failed to load")


def _fixed(proposal: dict[str, Any]) -> Propose:
    return lambda _instruction: json.dumps(proposal)


def _spec(proposal: dict[str, Any] = OFFLINE_PROPOSAL) -> TaskSpec:
    return TaskSpec.model_validate_json(json.dumps({**proposal, "instruction": INSTRUCTION}))


def _manifests(episodes: Path) -> list[EpisodeManifest]:
    found = [EpisodeManifest.model_validate_json(path.read_text()) for path in episodes.glob("*/manifest.json")]
    return sorted(found, key=lambda m: m.seed)


def test_an_instruction_argon_refuses_returns_its_reasons_and_records_nothing(tmp_path: Path) -> None:
    train = FakeFinetune()
    report = learn(
        "pour the tea",
        _fixed({"refuse": "pouring needs a grasp"}),
        runs_dir=tmp_path,
        demos=2,
        workers=1,
        exam_seeds=2,
        train=train,
        make_student=Stops,
        make_teacher=OraclePolicy,
    )
    assert report == Refusal(instruction="pour the tea", reasons=["Argon declined: pouring needs a grasp"])
    assert list(tmp_path.iterdir()) == [] and train.calls == []


def test_a_generated_task_trains_on_its_accepted_demonstrations_and_is_examined_on_held_out_seeds(
    tmp_path: Path,
) -> None:
    teachers = itertools.cycle([OraclePolicy, DropShortPolicy])  # a demonstration the shelf's rule accepts, then not
    train = FakeFinetune()
    report = learn(
        INSTRUCTION,
        _fixed(OFFLINE_PROPOSAL),
        runs_dir=tmp_path,
        demos=3,
        workers=1,
        exam_seeds=2,
        train=train,
        make_student=Stops,
        base=BASE_CHECKPOINT,
        make_teacher=lambda sim: next(teachers)(sim),
    )
    assert isinstance(report, LearnReport)
    run_dir = tmp_path / report.run_id
    spec = TaskSpec.model_validate_json((run_dir / "spec.json").read_text())
    assert report.spec == spec and report.fingerprint == spec.fingerprint() and report.taught == "batch"
    assert re.fullmatch(rf"learn-box_on_shelf-{spec.fingerprint()}-\d{{8}}-\d{{6}}", report.run_id)
    assert LearnReport.model_validate_json((run_dir / "report.json").read_text()) == report

    demos = _manifests(run_dir / "episodes")
    assert [m.seed for m in demos] == [DEMO_SEED_START, DEMO_SEED_START + 1, DEMO_SEED_START + 2]
    assert all(m.split == "dev" and m.seed not in heldout_seeds() for m in demos)  # rule 2
    assert [m.validation for m in demos] == [ON_THE_SHELF, OFF_THE_SHELF, ON_THE_SHELF]  # the spec's verdicts
    logged = [
        Event.model_validate_json(e) for e in (run_dir / "workers" / "w0" / "events.jsonl").read_text().splitlines()
    ]
    assert [e.data["reason"] for e in logged if e.type == "validation"] == [
        m.validation.reason for m in demos if m.validation
    ]
    assert {m.run_id for m in demos} == {e.run_id for e in logged} == {report.run_id}  # §4.2: they name the run

    accepted = [demos[0], demos[2]]
    lines = (run_dir / "training_r1.jsonl").read_text().splitlines()
    examples = [TrainingExample.model_validate_json(line) for line in lines]
    assert len(examples) == sum(len(m.steps) for m in accepted)  # every scripted step is the teacher's
    assert {e.instruction for e in examples} == {m.instruction for m in demos} == {INSTRUCTION}  # the person's words
    assert {Path(e.image).parents[1].name for e in examples} == {m.episode_id for m in accepted}  # rule 3
    summary = TrainingSetSummary.model_validate_json((run_dir / "training_r1.summary.json").read_text())
    assert (summary.demonstrations, summary.accepted, summary.rejected, summary.examples) == (3, 2, 1, len(examples))
    assert (report.demos, report.accepted, report.examples) == (3, 2, len(examples))

    assert train.calls == [(run_dir / "training_r1.jsonl", BASE_CHECKPOINT, 1)]
    assert report.checkpoint == f"{report.run_id}/ckpt-r1" and report.finetune_metrics == METRICS
    assert report.argon_calls == 0  # scripted teachers ask Argon nothing

    held_out = sorted(heldout_seeds())[:2]
    assert report.exam_seeds == held_out
    assert report.exam == {c: ExamResult(successes=0, errored=0, n=2) for c in (BASE_CHECKPOINT, report.checkpoint)}
    stopped = ExamResult(
        successes=0, errored=0, n=2, adaptation_tests=0, adapted=0, not_applied=0, trigger_not_reached=2
    )
    assert report.move_bin_share == 0.5 and report.exam_bin_moved == {c: stopped for c in report.exam}  # never lifted
    assert (report.reproduced, report.replay_failures, report.demos_errored) == (2, [], 0)  # by the shelf's rule
    for checkpoint in report.exam:
        examined = _manifests(run_dir / "exam" / checkpoint.replace("/", "_") / "episodes")
        assert [(m.seed, m.split, m.driver) for m in examined] == [(seed, "heldout", "student") for seed in held_out]
        assert {m.run_id for m in examined} == {report.run_id} and {m.instruction for m in examined} == {INSTRUCTION}


def test_a_demonstration_the_specs_rule_rejects_gives_no_training_examples(tmp_path: Path) -> None:
    train = FakeFinetune()
    report = learn(
        "put the box in the bin",
        _fixed(BIN_PROPOSAL),
        runs_dir=tmp_path,
        demos=2,
        workers=1,
        exam_seeds=1,
        train=train,
        make_student=Stops,
        base=BASE_CHECKPOINT,
        make_teacher=DropShortPolicy,
    )
    assert isinstance(report, LearnReport)
    run_dir = tmp_path / report.run_id
    demos = _manifests(run_dir / "episodes")
    assert all(m.steps for m in demos)  # recorded, just not trained on
    missed = Validation(task_success=False, physics_valid=True, reason="box not inside the bin")
    assert [m.validation for m in demos] == [missed, missed]
    assert (run_dir / "training_r1.jsonl").read_text() == ""
    assert (report.demos, report.accepted, report.examples) == (2, 0, 0)
    assert report.checkpoint == BASE_CHECKPOINT and report.finetune_metrics == {}  # nothing was trained
    assert list(report.exam) == list(report.exam_bin_moved) == [BASE_CHECKPOINT]  # so the base is examined once
    examined = [_manifests(run_dir / kind / BASE_CHECKPOINT / "episodes") for kind in ("exam", "exam-bin-moved")]
    assert [len(episodes) for episodes in examined] == [1, 1]


def test_the_specs_physics_check_is_the_validators() -> None:
    gt = GroundTruth(
        box_xyz=(0.0, 1.2, 0.45 + 0.125),  # resting centred on the shelf
        bin_xy=(0.0, 1.2),
        bin_yaw=0.0,
        bin_inner_half=0.22,
        box_on_bin_floor=True,
        box_at_rest=True,
        robot_fallen=True,
        joint_limit_violation=True,
        penetration=True,
        robot_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )
    verdict, validator = judge(_spec(), gt), validate(gt)
    assert not verdict.physics_valid and not validator.physics_valid
    physics = [ROBOT_FELL, JOINT_LIMIT, PENETRATION]
    assert verdict.reason.split(SEPARATOR)[:3] == validator.reason.split(SEPARATOR)[:3] == physics


def test_a_batch_over_two_teacher_processes_gathers_every_demonstration_into_the_run(tmp_path: Path) -> None:
    collect(_spec(), tmp_path, demos=3, workers=2, make_teacher=blind_oracle)
    demos = _manifests(tmp_path / "episodes")
    assert [m.seed for m in demos] == [DEMO_SEED_START, DEMO_SEED_START + 1, DEMO_SEED_START + 2]
    assert not {m.seed for m in demos} & heldout_seeds()  # rule 2
    assert [m.validation for m in demos] == [ON_THE_SHELF] * 3
    started = {
        worker.name: [e.data["seed"] for e in map(Event.model_validate_json, lines) if e.type == "episode_started"]
        for worker in sorted((tmp_path / "workers").iterdir())
        if (lines := (worker / "events.jsonl").read_text().splitlines())
    }
    assert started == {"w0": [DEMO_SEED_START, DEMO_SEED_START + 2], "w1": [DEMO_SEED_START + 1]}
    assert not list((tmp_path / "workers").glob("*/episodes/*"))  # moved into the run, not copied


def test_a_teacher_process_refuses_a_held_out_seed_before_recording_it(tmp_path: Path) -> None:
    out = tmp_path / "workers" / "w0"
    with pytest.raises(SealedSeedError):
        demonstrate(_spec().model_dump_json(), [min(heldout_seeds())], str(out), tmp_path.name, OraclePolicy)
    assert not (out / "episodes").exists()  # rule 2


def test_a_teacher_process_spends_no_more_than_its_own_argon_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def done(purpose: Purpose, parts: list[Part]) -> str:
        return '{"done": true, "reason": "the box looks placed"}'

    monkeypatch.setattr("isaacs_mom.teacher.argon.gemini_generate", lambda model="": done)
    monkeypatch.setenv("ARGON_CALL_BUDGET", "2")
    out = tmp_path / "workers" / "w0"
    seeds = [DEMO_SEED_START, DEMO_SEED_START + 1, DEMO_SEED_START + 2]  # one call each, were there budget for three
    demonstrate(_spec().model_dump_json(), seeds, str(out), tmp_path.name)
    logged = [Event.model_validate_json(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert sum(e.type == "argon_call" for e in logged) == 2  # rule 5
    closing = [e.data["reason"] for e in logged if e.type == "decision" and e.data["skill"]]
    assert closing == ["Argon: the task looks complete"] * 2 + ["ended: Argon call budget used up"]


def test_the_batch_counts_every_argon_call_of_every_teacher_process(tmp_path: Path) -> None:
    def answers(purpose: Purpose, parts: list[Part]) -> str:
        return '{"done": true}'

    def unreachable(purpose: Purpose, parts: list[Part]) -> str:
        raise ArgonUnavailable("ServerError: 503")

    first = Argon(EventLog(tmp_path / "workers" / "w0", "w0"), budget=2, generate=answers)
    for _ in range(3):  # the third is past this teacher's own budget: refused, so not a call
        first.ask("act", ["frame"])
    second = Argon(EventLog(tmp_path / "workers" / "w1", "w1"), budget=2, generate=unreachable)
    second.ask("act", ["frame"])  # failed, and still a call (rule 5)
    assert argon_calls(tmp_path) == 3


def test_the_exam_counts_the_specs_verdict_that_each_held_out_manifest_was_saved_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_the_shelf_past_the_bins_inner_half = GroundTruth(
        box_xyz=(BIN_INNER_HALF + BIN_WALL / 2, 1.2, 0.45 + 0.125),
        bin_xy=(0.0, 1.2),
        bin_yaw=0.0,
        bin_inner_half=BIN_INNER_HALF,
        box_on_bin_floor=True,
        box_at_rest=True,
        robot_fallen=False,
        joint_limit_violation=False,
        penetration=False,
        robot_xy=(0.0, 0.0),
        robot_yaw=0.0,
    )
    verdict = judge(_spec(), on_the_shelf_past_the_bins_inner_half)
    assert verdict.accepted and not validate(on_the_shelf_past_the_bins_inner_half).accepted  # the rules disagree
    monkeypatch.setattr(MujocoSim, "ground_truth", lambda _sim: on_the_shelf_past_the_bins_inner_half)
    seeds = sorted(heldout_seeds())[:2]
    assert exam(_spec(), tmp_path, "run/ckpt-r1", seeds, Stops) == ExamResult(successes=2, errored=0, n=2)
    out = tmp_path / "exam" / "run_ckpt-r1"
    assert [m.validation for m in _manifests(out / "episodes")] == [verdict, verdict]  # rule 6: the report's source
    logged = map(Event.model_validate_json, (out / "events.jsonl").read_text().splitlines())
    assert [e.data["reason"] for e in logged if e.type == "validation"] == [verdict.reason] * 2


def test_a_held_out_episode_that_crashes_counts_as_errored_not_as_a_failure(tmp_path: Path) -> None:
    students = itertools.cycle([Crashes, Stops])
    result = exam(_spec(), tmp_path, "run/ckpt-r1", [1, 2], lambda checkpoint: next(students)(checkpoint))
    assert result == ExamResult(successes=0, errored=1, n=2)


@pytest.mark.parametrize(
    ("flags", "cuda", "message"),
    [
        ([INSTRUCTION], False, "fine-tuning needs CUDA"),
        ([INSTRUCTION, "--workers", "0"], True, "must be at least 1, not 0"),
        ([INSTRUCTION, "--move-bin-share", "2"], True, "must be from 0 to 1, not 2.0"),
        ([], True, "give an instruction, or --resume with a run dir, not both"),
        ([INSTRUCTION, "--resume", "runs/x"], True, "give an instruction, or --resume with a run dir, not both"),
        (["--resume", "runs/x"], True, "has no finished batch"),
        (["--resume", "runs/x", "--collect-only"], False, "--collect-only stops after the batch and --resume starts"),
        (["--resume", "elsewhere/x"], True, "--resume takes a dir in ./runs, where the exam finds its adapter"),
    ],
)
def test_the_cli_refuses_before_asking_argon_anything(
    flags: list[str],
    cuda: bool,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("isaacs_mom.tasks.learn.has_cuda", lambda: cuda)
    monkeypatch.setattr(sys, "argv", ["learn", "--demos", "2", *flags])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []  # no runs/: not even the generator's Argon was made


class LiftsThenStops:
    """A student that plays the scripted teacher up to its lift, then stops: the move is applied but never met."""

    controller: Controller = "student"

    def __init__(self, checkpoint: str, sims: list[MujocoSim]) -> None:
        self.checkpoint: str | None = checkpoint
        self.sims = sims
        self.policy: OraclePolicy | None = None
        self.lifted = False

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if self.lifted:
            return None
        self.policy = self.policy or OraclePolicy(self.sims[-1])  # the exam's own sim, built before its episodes
        call = self.policy.act(obs, instruction, feedback)
        self.lifted = call is not None and call.skill == "lift"
        return call


def test_a_spec_that_lets_the_bin_move_moves_it_in_the_stated_share_and_replays_every_accepted_demo(
    tmp_path: Path,
) -> None:
    spec = _spec(BIN_PROPOSAL)
    collected = collect(spec, tmp_path, demos=2, workers=1, make_teacher=OraclePolicy, move_bin_share=0.5)
    assert collected == Collected(demos=2, workers=1, move_bin_share=0.5, moved_seeds=[DEMO_SEED_START + 1])
    demos = _manifests(tmp_path / "episodes")
    assert not {m.seed for m in demos} & heldout_seeds()  # rule 2
    assert [[p.perturbation.kind for p in m.perturbations] for m in demos] == [[], ["move_bin"]]
    assert all(m.validation and m.validation.accepted for m in demos)  # found the bin again
    assert all(m.replay_valid for m in demos)  # by the bin spec's own rule, the move included
    assert Collected.model_validate_json((tmp_path / "collected.json").read_text()) == collected


def test_a_spec_that_lets_nothing_move_never_moves_the_bin_whatever_the_share(tmp_path: Path) -> None:
    still = _spec({**OFFLINE_PROPOSAL, "perturbations": []})
    collected = collect(still, tmp_path, demos=2, workers=1, make_teacher=OraclePolicy, move_bin_share=1.0)
    assert (collected.move_bin_share, collected.moved_seeds) == (0.0, [])
    assert all(m.perturbations == [] for m in _manifests(tmp_path / "episodes"))


class CrashesOnTheSecond(OraclePolicy):
    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.episode_id.startswith(f"ep-{DEMO_SEED_START + 1}-"):
            raise ArgonUnavailable("ServerError: 503")
        return super().act(obs, instruction, feedback)


def test_a_demonstration_that_crashes_is_counted_and_the_batch_goes_on(tmp_path: Path) -> None:
    spec, run_dir = _started(tmp_path)
    collect(spec, run_dir, demos=3, workers=1, make_teacher=CrashesOnTheSecond)
    report = finish(run_dir, exam_seeds=1, train=FakeFinetune(), make_student=Stops)
    assert (report.demos_errored, report.accepted) == (1, 2)
    errors = [json.loads(line) for line in (run_dir / "workers" / "w0" / "events.jsonl").read_text().splitlines()]
    assert [e["data"]["seed"] for e in errors if e["type"] == "error"] == [DEMO_SEED_START + 1]


def test_a_replay_that_raises_fails_that_demonstration_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replays = []

    def raises_once(sim: MujocoSim, path: Path, verdict: Any) -> bool:
        replays.append(path)
        if len(replays) == 1:
            raise RuntimeError("mj_stackAlloc: out of memory")
        return replay(sim, path, verdict)

    monkeypatch.setattr("isaacs_mom.tasks.learn.replay", raises_once)
    collect(_spec(), tmp_path, demos=2, workers=1, make_teacher=OraclePolicy)
    demos = _manifests(tmp_path / "episodes")
    assert [m.validation for m in demos] == [ON_THE_SHELF] * 2 and [m.replay_valid for m in demos] == [False, True]
    logged = map(Event.model_validate_json, (tmp_path / "workers" / "w0" / "events.jsonl").read_text().splitlines())
    assert [(e.data["episode_id"], e.data["error"]) for e in logged if e.type == "error"] == [
        (demos[0].episode_id, "replay raised RuntimeError: mj_stackAlloc: out of memory")
    ]


FORBIDDEN = errors.ClientError(
    403, {"error": {"code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED"}}
)


class StopsTheRunOnTheSecond(OraclePolicy):
    def __init__(self, sim: MujocoSim, fatal: Exception) -> None:
        super().__init__(sim)
        self.fatal = fatal

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.episode_id.startswith(f"ep-{DEMO_SEED_START + 1}-"):
            raise self.fatal
        return super().act(obs, instruction, feedback)


@pytest.mark.parametrize("fatal", [FORBIDDEN, MissingKey("GEMINI_API_KEY is not set")], ids=["4xx", "no key"])
def test_a_fatal_argon_error_stops_the_batch_and_leaves_what_it_recorded_to_resume(
    tmp_path: Path, fatal: Exception
) -> None:
    spec, run_dir = _started(tmp_path)
    with pytest.raises(type(fatal)):
        collect(spec, run_dir, demos=3, workers=1, make_teacher=lambda sim: StopsTheRunOnTheSecond(sim, fatal))
    assert [(m.seed, m.validation) for m in _manifests(run_dir / "episodes")] == [
        (DEMO_SEED_START, ON_THE_SHELF),
        (DEMO_SEED_START + 1, None),  # cut off at its first step; the third seed never ran
    ]
    collected = Collected.model_validate_json((run_dir / "collected.json").read_text())
    assert collected.stopped_early == {"w0": f"{type(fatal).__name__}: {fatal}"}
    report = finish(run_dir, exam_seeds=1, train=FakeFinetune(), make_student=Stops)
    assert (report.demos, report.accepted, report.demos_errored) == (2, 1, 2)


class DiesOnTheSecond(OraclePolicy):
    """Ends its teacher process outright on the second seed, as a segfault or the OOM killer would."""

    def __init__(self, sim: MujocoSim) -> None:
        sim.observe = partial(_blank, sim)
        super().__init__(sim)

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.episode_id.startswith(f"ep-{DEMO_SEED_START + 1}-"):
            os._exit(1)
        return super().act(obs, instruction, feedback)


def test_a_teacher_process_that_dies_leaves_a_run_that_resumes(tmp_path: Path) -> None:
    spec, run_dir = _started(tmp_path)
    with pytest.raises(BrokenProcessPool):
        collect(spec, run_dir, demos=3, workers=2, make_teacher=DiesOnTheSecond)
    collected = Collected.model_validate_json((run_dir / "collected.json").read_text())
    assert collected.stopped_early["w1"].startswith("BrokenProcessPool")
    assert not list((run_dir / "workers").glob("*/episodes/*"))  # whatever any process recorded is in the run
    report = finish(run_dir, exam_seeds=1, train=FakeFinetune(), make_student=Stops)
    assert report.demos_errored >= 1 and report.accepted + report.demos_errored == 3


def _started(tmp_path: Path, proposal: dict[str, Any] = OFFLINE_PROPOSAL) -> tuple[TaskSpec, Path]:
    """As the CLI starts: the generator's startup check and proposal, logged where another start logs its calls."""
    shared = tmp_path / "learn-generator"
    Argon(EventLog(shared, "learn-generator-before"), budget=1, generate=lambda _p, _parts: "{}").ask("task", ["?"])
    generator = EventLog(shared, "learn-generator-this")
    argon = Argon(generator, budget=2, generate=lambda _purpose, _parts: json.dumps(proposal))
    argon.check()
    started = start(INSTRUCTION, argon_proposer(argon), tmp_path, generator)
    assert not isinstance(started, Refusal)
    return started


def test_a_batch_collected_alone_is_finished_later_from_its_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, run_dir = _started(tmp_path, BIN_PROPOSAL)
    copied = [Event.model_validate_json(e) for e in (run_dir / "generator" / "events.jsonl").read_text().splitlines()]
    assert [(e.run_id, e.type, e.data["purpose"]) for e in copied] == [
        ("learn-generator-this", "argon_call", "startup"),
        ("learn-generator-this", "argon_call", "task"),
    ]
    collect(spec, run_dir, demos=1, workers=1, make_teacher=OraclePolicy)
    sims: list[MujocoSim] = []
    built = MujocoSim.__init__

    def tracked(self: MujocoSim, *args: Any, **kwargs: Any) -> None:
        built(self, *args, **kwargs)
        sims.append(self)

    monkeypatch.setattr(MujocoSim, "__init__", tracked)
    report = finish(run_dir, exam_seeds=2, train=FakeFinetune(), make_student=lambda c: LiftsThenStops(c, sims))
    assert report.run_id == run_dir.name and report.generator_calls == 2 and report.accepted == 1
    moved = report.exam_bin_moved[report.checkpoint]
    assert (moved.n, moved.adaptation_tests, moved.not_applied, moved.trigger_not_reached) == (2, 2, 0, 0)
    assert moved.adapted == moved.successes == 0  # it stops once lifted, so the move is applied but never met
    assert report.exam[report.checkpoint] == ExamResult(successes=0, errored=0, n=2)
    lines = summary(report).splitlines()
    assert lines[1].startswith("demonstrations: 1/1 accepted (0 errored)") and "1/1 accepted replay" in lines[1]
    assert lines[-1] == (
        f"  {report.checkpoint}: 0/2; bin moved after the lift 0/2, adapted 0/2 (not applied 0, no lift 0)"
    )
    one_errored = {
        "exam": {report.checkpoint: ExamResult(successes=0, errored=1, n=2)},
        "exam_bin_moved": {
            report.checkpoint: ExamResult(
                successes=0, errored=1, n=2, adaptation_tests=1, adapted=0, not_applied=0, trigger_not_reached=0
            )
        },
    }
    assert summary(report.model_copy(update=one_errored)).splitlines()[-1] == (
        f"  {report.checkpoint}: 0/1; bin moved after the lift 0/1, adapted 0/1 (not applied 0, no lift 0)"
    )
