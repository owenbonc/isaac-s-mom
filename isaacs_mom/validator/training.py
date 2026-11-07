from pathlib import Path

from isaacs_mom.contracts import EpisodeManifest, TrainingExample
from isaacs_mom.seeds import assert_not_heldout


def training_examples(episode_dir: Path, manifest: EpisodeManifest) -> list[TrainingExample]:
    """Teacher steps of one accepted dev episode (SPEC §2 rules 2 and 3); nothing otherwise."""
    if manifest.split != "dev" or manifest.validation is None or not manifest.validation.accepted:
        return []
    assert_not_heldout(manifest.seed)
    examples: list[TrainingExample] = []
    feedback = "start"
    for record in manifest.steps:
        if record.controller == "teacher":
            examples.append(
                TrainingExample(
                    image=str(episode_dir / record.frame),
                    instruction=manifest.instruction,
                    feedback=feedback,
                    target=record.skill_call.model_copy(update={"reason": ""}),
                )
            )
        feedback = f"{record.skill_call.skill}: {'ok' if record.ok else record.error}"
    return examples


def build_training_set(run_dir: Path, round_: int) -> Path:
    """All accepted teacher demonstrations recorded so far in this run, as JSON lines (SPEC §4.3)."""
    out = run_dir / f"training_r{round_}.jsonl"
    lines: list[str] = []
    for manifest_path in sorted((run_dir / "episodes").glob("*/manifest.json")):
        manifest = EpisodeManifest.model_validate_json(manifest_path.read_text())
        lines += [e.model_dump_json() for e in training_examples(manifest_path.parent, manifest)]
    out.write_text("".join(line + "\n" for line in lines))
    return out
