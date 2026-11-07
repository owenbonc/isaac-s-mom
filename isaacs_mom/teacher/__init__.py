"""Workstream 2: the Argon teacher (SPEC §3.3). The bodies below are skeleton stand-ins."""

from isaacs_mom.contracts import Controller, Diagnosis, EpisodeManifest, Observation, SkillCall
from isaacs_mom.seeds import DEV_SEED_START, assert_not_heldout, heldout_seeds


def diagnose(failed: EpisodeManifest, frames: list[bytes]) -> Diagnosis:
    """From a failed episode's frames, skill calls and validator reason, a failure hypothesis."""
    reason = failed.validation.reason if failed.validation else "no validation"
    return Diagnosis(hypothesis="skeleton: not diagnosed", evidence=reason, confidence=0.0, lesson="")


def choose_scenes(diagnoses: list[Diagnosis], n: int, start: int = DEV_SEED_START) -> list[int]:
    """Development seeds that target the diagnosed failures. Skeleton: the next n dev seeds."""
    seeds = [s for s in range(start, start + n + len(heldout_seeds())) if s not in heldout_seeds()][:n]
    for seed in seeds:
        assert_not_heldout(seed)
    return seeds


class ArgonPolicy:
    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, diagnosis: Diagnosis | None) -> None:
        self.diagnosis = diagnosis

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        # Skeleton: look once, then stop. Argon replaces this.
        return SkillCall(skill="reobserve") if obs.step == 0 else None
