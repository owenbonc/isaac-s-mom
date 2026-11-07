from typing import Protocol

from isaacs_mom.contracts import Controller, Observation, SkillCall


class Policy(Protocol):
    """Chooses the next skill from what the robot sees. Returns None when it believes the task is done."""

    @property
    def controller(self) -> Controller: ...

    @property
    def checkpoint(self) -> str | None: ...

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None: ...
