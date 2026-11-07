from pathlib import Path

import mujoco
import numpy as np

from isaacs_mom.contracts import EpisodeManifest, ReplayFailure, ReplayReport
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.validator.recorder import write_atomically, write_manifest
from isaacs_mom.validator.rules import validate

# The recorder saves actions as float32, so a faithful re-run matches only to float32 precision.
TOLERANCE = 1e-5
# MuJoCo's FatalError subclasses none of the built-in errors.
REPLAY_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    ArithmeticError,
    LookupError,
    TypeError,
    AttributeError,
    mujoco.FatalError,  # pyright: ignore[reportAttributeAccessIssue]  (mujoco ships no types)
)


def replay(sim: TruthSim, manifest_path: Path) -> bool:
    """SPEC §3.4: re-run the stored skill calls and perturbations from the stored seed; sets `replay_valid`.

    True when every recorded perturbation applies again, every step's executed actions, timestamps and outcome match
    the recording and the final verdict matches the stored one, so a rejected episode replays as valid when its
    failure reproduces.
    """
    manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
    reproduced = _divergence(sim, manifest, manifest_path.parent) is None
    manifest.replay_valid = reproduced
    write_manifest(manifest_path, manifest)
    return reproduced


def replay_accepted(sim: TruthSim, run_dir: Path) -> ReplayReport:
    """Replay every accepted episode in a run (the ones that can become lessons); writes `replay_report.json`.

    An episode that can't be read, re-run (any of `REPLAY_ERRORS`) or saved is recorded as a failed replay with a
    reason instead of raising, so one bad episode can't lose the report for the whole run.
    """
    replayed = 0
    failures: list[ReplayFailure] = []
    for manifest_path in sorted((run_dir / "episodes").glob("*/manifest.json")):
        try:
            manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
        except (OSError, ValueError) as error:
            replayed += 1
            failures.append(ReplayFailure(episode_id=manifest_path.parent.name, reason=f"manifest unreadable: {error}"))
            continue
        if manifest.validation is None or not manifest.validation.accepted:
            continue
        replayed += 1
        try:
            reason = _divergence(sim, manifest, manifest_path.parent)
        except REPLAY_ERRORS as error:
            reason = f"replay raised {type(error).__name__}: {error}"
        manifest.replay_valid = reason is None
        try:
            write_manifest(manifest_path, manifest)
        except OSError as error:
            saving = f"could not save the replay result: {error}"
            reason = f"{reason}; {saving}" if reason else saving
        if reason is not None:
            failures.append(ReplayFailure(episode_id=manifest.episode_id, reason=reason))
    report = ReplayReport(replayed=replayed, reproduced=replayed - len(failures), failures=failures)
    write_atomically(run_dir / "replay_report.json", report.model_dump_json(indent=2))
    return report


def _divergence(sim: TruthSim, manifest: EpisodeManifest, episode_dir: Path) -> str | None:
    """Where a re-run first differs from the recording, or None when it reproduces."""
    sim.reset(manifest.seed)
    for i, step in enumerate(manifest.steps):
        applied = _perturb_before(sim, manifest, i)
        result = sim.execute(step.skill_call)
        stored = np.load(episode_dir / step.actions)
        actual = np.asarray(result.executed_actions, dtype=np.float32)
        if not applied:
            return f"step {i}: a recorded perturbation no longer applies"
        if result.ok != step.ok:
            return f"step {i}: outcome differs"
        if actual.shape != stored.shape or not np.allclose(actual, stored, atol=TOLERANCE):
            return f"step {i}: executed actions differ"
        if len(result.timestamps) != len(step.timestamps) or not np.allclose(
            result.timestamps, step.timestamps, atol=TOLERANCE
        ):
            return f"step {i}: timestamps differ"
    if not _perturb_before(sim, manifest, len(manifest.steps)):
        return "a recorded perturbation after the last step no longer applies"
    if validate(sim.ground_truth()) != manifest.validation:
        return "final verdict differs"
    return None


def _perturb_before(sim: TruthSim, manifest: EpisodeManifest, step: int) -> bool:
    """Apply the perturbations recorded before `step`; False when one no longer applies."""
    return all([sim.perturb(r.perturbation) for r in manifest.perturbations if r.before_step == step])
