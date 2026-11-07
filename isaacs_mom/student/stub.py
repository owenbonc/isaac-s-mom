"""Stand-ins for runs without the model (`mom run --student stub`, CI): random whitelisted skills, no training."""

import random
from pathlib import Path

from isaacs_mom.contracts import SKILLS, Controller, Observation, SkillCall


class StubStudent:
    controller: Controller = "student"

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint: str | None = checkpoint
        self._rng = random.Random(checkpoint)

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.step >= 8:
            return None
        return SkillCall(skill=self._rng.choice(SKILLS))


def stub_finetune(training_set: Path, base_checkpoint: str, round_: int) -> str:
    """No training, so the student keeps its checkpoint."""
    return base_checkpoint
