from pathlib import Path

import numpy as np

from isaacs_mom.contracts import EpisodeManifest, ReplayReport
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.validator.recorder import write_manifest
from isaacs_mom.validator.rules import validate

# The recorder saves actions as float32, so a faithful re-run matches only to float32 precision.
TOLERANCE = 1e-5


def replay(sim: TruthSim, manifest_path: Path) -> bool:
    """SPEC §3.4: re-run the stored skill calls and perturbations from the stored seed; sets `replay_valid`.

    True when every step's executed actions, timestamps and outcome match the recording and the final verdict
    matches the stored one, so a rejected episode replays as valid when its failure reproduces.
    """
    manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
    sim.reset(manifest.seed)
    reproduced = True
    for i, step in enumerate(manifest.steps):
        _perturb_before(sim, manifest, i)
        result = sim.execute(step.skill_call)
        stored = np.load(manifest_path.parent / step.actions)
        actual = np.asarray(result.executed_actions, dtype=np.float32)
        reproduced = (
            reproduced
            and result.ok == step.ok
            and actual.shape == stored.shape
            and np.allclose(actual, stored, atol=TOLERANCE)
            and np.allclose(result.timestamps, step.timestamps, atol=TOLERANCE)
        )
    _perturb_before(sim, manifest, len(manifest.steps))
    reproduced = reproduced and validate(sim.ground_truth()) == manifest.validation
    manifest.replay_valid = reproduced
    write_manifest(manifest_path, manifest)
    return reproduced


def _perturb_before(sim: TruthSim, manifest: EpisodeManifest, step: int) -> None:
    for record in manifest.perturbations:
        if record.before_step == step:
            sim.perturb(record.perturbation)


def replay_accepted(sim: TruthSim, run_dir: Path) -> ReplayReport:
    """Replay every accepted episode in a run (the ones that can become lessons); writes `replay_report.json`."""
    not_reproduced: list[str] = []
    checked = 0
    for manifest_path in sorted((run_dir / "episodes").glob("*/manifest.json")):
        manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
        if manifest.validation is None or not manifest.validation.accepted:
            continue
        checked += 1
        if not replay(sim, manifest_path):
            not_reproduced.append(manifest.episode_id)
    report = ReplayReport(checked=checked, reproduced=checked - len(not_reproduced), not_reproduced=not_reproduced)
    (run_dir / "replay_report.json").write_text(report.model_dump_json(indent=2))
    return report
