import math
import os
from itertools import pairwise
from pathlib import Path

import numpy as np

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    ExecResult,
    Observation,
    Perturbation,
    PerturbationRecord,
    SkillCall,
    Split,
    StepRecord,
    Validation,
)


class RecordingError(ValueError):
    pass


class Recorder:
    """Writes one episode to runs/<run_id>/episodes/<episode_id>/ (SPEC §4.2): frames, executed actions, manifest.

    The manifest is rewritten after every change, so an episode that crashes leaves a diary with no verdict.
    """

    def __init__(self, run_dir: Path, manifest: EpisodeManifest) -> None:
        self.dir = run_dir / "episodes" / manifest.episode_id
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "actions").mkdir(exist_ok=True)
        self.manifest = manifest
        self._save()

    @classmethod
    def start(
        cls,
        run_dir: Path,
        *,
        episode_id: str,
        run_id: str,
        seed: int,
        split: Split,
        round_: int,
        instruction: str,
        sim_version: str,
        robot_config: str,
        driver: Controller,
    ) -> "Recorder":
        manifest = EpisodeManifest(
            episode_id=episode_id,
            run_id=run_id,
            seed=seed,
            split=split,
            round=round_,
            instruction=instruction,
            sim_version=sim_version,
            robot_config=robot_config,
            driver=driver,
        )
        return cls(run_dir, manifest)

    def add_step(
        self, obs: Observation, call: SkillCall, result: ExecResult, controller: Controller, checkpoint: str | None
    ) -> None:
        _check_executed(result)
        i = len(self.manifest.steps)
        frame = f"frames/{i:04d}.jpg"
        actions = f"actions/{i:04d}.npy"
        (self.dir / frame).write_bytes(obs.head_rgb_jpeg)
        np.save(self.dir / actions, np.asarray(result.executed_actions, dtype=np.float32))
        self.manifest.steps.append(
            StepRecord(
                step=obs.step,
                sim_time=obs.sim_time,
                frame=frame,
                joint_state=obs.joint_state,
                skill_call=call,
                controller=controller,
                checkpoint=checkpoint,
                actions=actions,
                timestamps=result.timestamps,
                ok=result.ok,
                error=result.error,
            )
        )
        self._save()

    def add_perturbation(self, perturbation: Perturbation) -> None:
        self.manifest.perturbations.append(
            PerturbationRecord(before_step=len(self.manifest.steps), perturbation=perturbation)
        )
        self._save()

    def mark_takeover(self) -> None:
        self.manifest.takeover_step = len(self.manifest.steps)
        self._save()

    def finish(self, validation: Validation) -> EpisodeManifest:
        self.manifest.validation = validation
        self._save()
        return self.manifest

    def _save(self) -> None:
        write_manifest(self.dir / "manifest.json", self.manifest)


def write_manifest(path: Path, manifest: EpisodeManifest) -> None:
    write_atomically(path, manifest.model_dump_json(indent=2))


def write_atomically(path: Path, text: str) -> None:
    # Write then rename, so a process crash mid-write never leaves a half-written file.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def _check_executed(result: ExecResult) -> None:
    """SPEC §2 rule 4: every executed action needs its own timestamp, in time order."""
    if len(result.executed_actions) != len(result.timestamps):
        raise RecordingError(f"{len(result.executed_actions)} executed actions but {len(result.timestamps)} timestamps")
    if not all(math.isfinite(t) for t in result.timestamps):
        raise RecordingError("timestamps are not finite")
    if any(later < earlier for earlier, later in pairwise(result.timestamps)):
        raise RecordingError("timestamps go backwards")
