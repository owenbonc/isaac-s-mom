from pathlib import Path

import numpy as np

from isaacs_mom.contracts import (
    Controller,
    EpisodeManifest,
    ExecResult,
    Observation,
    SkillCall,
    Split,
    StepRecord,
    Validation,
)


class Recorder:
    """Writes one episode to runs/<run_id>/episodes/<episode_id>/ (SPEC §4.2): frames, executed actions, manifest."""

    def __init__(self, run_dir: Path, manifest: EpisodeManifest) -> None:
        self.dir = run_dir / "episodes" / manifest.episode_id
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "actions").mkdir(exist_ok=True)
        self.manifest = manifest

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
        )
        return cls(run_dir, manifest)

    def add_step(
        self, obs: Observation, call: SkillCall, result: ExecResult, controller: Controller, checkpoint: str | None
    ) -> None:
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

    def mark_takeover(self) -> None:
        self.manifest.takeover_step = len(self.manifest.steps)

    def finish(self, validation: Validation) -> EpisodeManifest:
        self.manifest.validation = validation
        (self.dir / "manifest.json").write_text(self.manifest.model_dump_json(indent=2))
        return self.manifest
