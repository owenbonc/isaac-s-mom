"""The SmolVLM2 student: one reply per step, parsed into a whitelisted SkillCall (SPEC §3.2)."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import Controller, Observation, Skill, SkillCall
from isaacs_mom.student.checkpoints import BASE_CHECKPOINT, RUNS_DIR, resolve_adapter
from isaacs_mom.student.prompt import UnparseableReply, parse_reply

Generate = Callable[[bytes, str, str], str]
log = logging.getLogger(__name__)


class StudentPolicy:
    controller: Controller = "student"

    def __init__(
        self,
        checkpoint: str = BASE_CHECKPOINT,
        *,
        runs_dir: Path = RUNS_DIR,
        generate: Generate | None = None,
    ) -> None:
        self.checkpoint: str | None = checkpoint
        self._adapter = resolve_adapter(checkpoint, runs_dir)
        self._generate = generate
        self._episode_id: str | None = None
        self._last_skill: Skill | None = None
        self.last_latency_ms = 0.0

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if obs.episode_id != self._episode_id:
            self._episode_id, self._last_skill = obs.episode_id, None
        generate = self._generate or self._load()
        started = time.perf_counter()
        text = generate(obs.head_rgb_jpeg, instruction, feedback)
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        try:
            call = parse_reply(text)
        except UnparseableReply as error:
            log.warning(
                "student %s step %d: unparseable reply, re-observing (%s): %r",
                self.checkpoint,
                obs.step,
                error,
                text[:200],
            )
            self._last_skill = None
            return SkillCall(skill="reobserve")
        log.info(
            "student %s step %d: %s in %.0f ms", self.checkpoint, obs.step, call.model_dump_json(), self.last_latency_ms
        )
        # SPEC §3.2/§3.3: the teacher ends every demo with one reobserve, so a second in a row from the model is done.
        if call.skill == "reobserve" and self._last_skill == "reobserve":
            return None
        self._last_skill = call.skill
        return call

    def _load(self) -> Generate:
        from isaacs_mom.student import vlm  # torch loads only when the real model is needed

        self._generate = vlm.generator(self._adapter)
        return self._generate
