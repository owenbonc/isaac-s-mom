from pathlib import Path

import numpy as np

from isaacs_mom.contracts import EpisodeManifest
from isaacs_mom.sim.base import TruthSim
from isaacs_mom.validator.rules import validate

# The recorder saves actions as float32, so a faithful re-run matches only to float32 precision.
TOLERANCE = 1e-5


def replay(sim: TruthSim, manifest_path: Path) -> bool:
    """SPEC §3.4: re-run the stored skill calls and perturbations from the stored seed; sets `replay_valid`.

    True when every recorded perturbation applies again, every step's executed actions, timestamps and outcome match
    the recording and the final verdict matches the stored one, so a rejected episode replays as valid when its
    failure reproduces.
    """
    manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
    sim.reset(manifest.seed)
    reproduced = True
    for i, step in enumerate(manifest.steps):
        applied = _perturb_before(sim, manifest, i)
        result = sim.execute(step.skill_call)
        stored = np.load(manifest_path.parent / step.actions)
        actual = np.asarray(result.executed_actions, dtype=np.float32)
        reproduced = (
            reproduced
            and applied
            and result.ok == step.ok
            and actual.shape == stored.shape
            and np.allclose(actual, stored, atol=TOLERANCE)
            and np.allclose(result.timestamps, step.timestamps, atol=TOLERANCE)
        )
    applied = _perturb_before(sim, manifest, len(manifest.steps))
    reproduced = reproduced and applied and validate(sim.ground_truth()) == manifest.validation
    manifest.replay_valid = reproduced
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return reproduced


def _perturb_before(sim: TruthSim, manifest: EpisodeManifest, step: int) -> bool:
    """Apply the perturbations recorded before `step`; False when one no longer applies."""
    return all([sim.perturb(r.perturbation) for r in manifest.perturbations if r.before_step == step])
