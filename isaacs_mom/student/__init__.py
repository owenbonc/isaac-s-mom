"""Workstream 4: SmolVLM2 student and LoRA fine-tuning (SPEC §3.2). The bodies below are skeleton stand-ins."""

import random
from pathlib import Path

from isaacs_mom.contracts import SKILLS, Controller, Observation, SkillCall
from isaacs_mom.student.checkpoints import BASE_CHECKPOINT


class StudentPolicy:
    controller: Controller = "student"

    def __init__(self, checkpoint: str = BASE_CHECKPOINT) -> None:
        self.checkpoint: str | None = checkpoint
        self._rng = random.Random(checkpoint)

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        # Skeleton: a random whitelisted skill until SmolVLM2 lands; stops after 8 steps.
        if obs.step >= 8:
            return None
        return SkillCall(skill=self._rng.choice(SKILLS))


def finetune(training_set: Path, base_checkpoint: str, round_: int) -> str:
    """LoRA fine-tune on Modal; returns the new checkpoint id. Skeleton: no training."""
    return f"ckpt-r{round_}"
