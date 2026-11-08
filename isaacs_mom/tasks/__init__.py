"""The task generator (SPEC §3.7): a typed task from a typed instruction, judged by a declarative rule."""

from isaacs_mom.tasks.generate import Propose, argon_proposer, generate_task
from isaacs_mom.tasks.rules import judge
from isaacs_mom.tasks.spec import Refusal, TaskSpec

__all__ = ["Propose", "Refusal", "TaskSpec", "argon_proposer", "generate_task", "judge"]
