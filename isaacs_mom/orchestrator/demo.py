"""The judged single take (SPEC §3.5): the student on an unseen scene, a perturbation mid-run, a re-observe, a finish.

Demo seeds come from a fresh range (>= 900000), so no training or evaluation episode has seen the scene.
The teacher may take over, so the seed still goes through `assert_not_heldout` (SPEC §2 rule 2).
"""

from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import EpisodeManifest, ExecResult, GroundTruth, Observation, Perturbation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import INSTRUCTION, StudentFactory
from isaacs_mom.orchestrator.video import record
from isaacs_mom.seeds import assert_not_heldout
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.student import StudentPolicy
from isaacs_mom.teacher import ArgonPolicy
from isaacs_mom.teacher.argon import Argon

DEMO_SEED_START = 900_000


class _ThirdPersonCapture:
    """Wraps a TruthSim and keeps the third-person render taken with each observation, keyed by step."""

    def __init__(self, sim: TruthSim) -> None:
        self._sim = sim
        self.third_person: dict[int, bytes] = {}
        self.sim_version = sim.sim_version
        self.robot_config = sim.robot_config

    def reset(self, seed: int) -> str:
        return self._sim.reset(seed)

    def observe(self) -> Observation:
        obs = self._sim.observe()
        self.third_person[obs.step] = self._sim.render_third_person()
        return obs

    def execute(self, call: SkillCall) -> ExecResult:
        return self._sim.execute(call)

    def ground_truth(self) -> GroundTruth:
        return self._sim.ground_truth()

    def perturb(self, perturbation: Perturbation) -> bool:
        return self._sim.perturb(perturbation)

    def render_third_person(self) -> bytes:
        return self._sim.render_third_person()


def _verdict(manifest: EpisodeManifest) -> str:
    """The take's verdict in the dashboard's words (SPEC §3.5): whether the student finished it alone."""
    verdict = manifest.validation
    if verdict is None:
        return "pending"
    took = "" if manifest.takeover_step is None else f" after the teacher took over at step {manifest.takeover_step}"
    return f"succeeded{took or ', by the student alone'}" if verdict.accepted else f"failed{took}: {verdict.reason}"


def _captions(manifest: EpisodeManifest) -> list[str]:
    final = _verdict(manifest)
    # Only an applied move is recorded, so a refused one, or none, is never captioned (SPEC §2 rule 6).
    first = manifest.perturbations[0] if manifest.perturbations else None
    moved = first.perturbation.kind.removeprefix("move_") if first else ""
    captions = []
    for i, step in enumerate(manifest.steps):
        who = f"{step.controller} {step.checkpoint}" if step.checkpoint else step.controller
        amounts = [f"{v:g} cm" if k == "distance_cm" else f"{v:g}°" for k, v in step.skill_call.params.items()]
        skill = " ".join([step.skill_call.skill, *amounts])
        state = "validator: " + (final if i == len(manifest.steps) - 1 else "pending")
        perturbed = f"{moved} moved   " if first and i >= first.before_step else ""
        captions.append(f"{who}   {skill}   {perturbed}{state}")
    return captions


def refuse_seen_take(seed: int, run_dir: Path) -> None:
    """The judged take needs a scene no episode has seen, and a run directory of its own so it never enters a loop
    run's training set (SPEC §3.5)."""
    assert_not_heldout(seed)
    if seed < DEMO_SEED_START:
        raise ValueError(f"seed {seed} is below {DEMO_SEED_START}: the judged take needs a scene no episode has seen")
    if run_dir.exists():
        raise ValueError(f"{run_dir} exists: the judged take records into a run directory of its own")


def run_demo(
    sim: TruthSim,
    *,
    ckpt: str,
    seed: int,
    perturb_at: int,
    out: Path,
    run_dir: Path,
    run_id: str,
    fps: int = 2,
    make_argon: Callable[[EventLog], Argon],
    make_student: StudentFactory = StudentPolicy,
) -> EpisodeManifest:
    """Refuses a seen scene or an existing run directory before it writes anything or builds its student or teacher."""
    refuse_seen_take(seed, run_dir)
    student = make_student(ckpt)
    run_dir.mkdir(parents=True)
    capture = _ThirdPersonCapture(sim)
    events = EventLog(run_dir, run_id)
    manifest = run_episode(
        capture,
        student,
        run_dir=run_dir,
        events=events,
        seed=seed,
        split="dev",
        round_=0,
        instruction=INSTRUCTION,
        takeover=ArgonPolicy(None, make_argon(events)),
        perturb_at=perturb_at,
        demo=True,
    )
    episode_dir = run_dir / "episodes" / manifest.episode_id
    record(
        (
            ((episode_dir / step.frame).read_bytes(), capture.third_person[step.step], caption)
            for step, caption in zip(manifest.steps, _captions(manifest), strict=True)
        ),
        out,
        fps,
    )
    return manifest
