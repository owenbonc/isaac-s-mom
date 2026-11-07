from collections import Counter
from pathlib import Path

from isaacs_mom.contracts import EpisodeManifest, SkillTarget, TrainingExample, TrainingSetSummary
from isaacs_mom.seeds import assert_not_heldout
from isaacs_mom.validator.rules import failure_categories


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
                    target=SkillTarget(skill=record.skill_call.skill, params=record.skill_call.params),
                )
            )
        feedback = f"{record.skill_call.skill}: {'ok' if record.ok else record.error}"
    return examples


def build_training_set(run_dir: Path, round_: int) -> Path:
    """All accepted teacher demonstrations recorded so far in this run, as JSON lines (SPEC §4.3).

    Also writes `training_r<round>.summary.json`: across every round so far, how many demonstrations were seen,
    kept and rejected, and why.
    """
    out = run_dir / f"training_r{round_}.jsonl"
    lines: list[str] = []
    manifests = [
        (path.parent, EpisodeManifest.model_validate_json(path.read_text()))
        for path in sorted((run_dir / "episodes").glob("*/manifest.json"))
    ]
    for episode_dir, manifest in manifests:
        lines += [e.model_dump_json() for e in training_examples(episode_dir, manifest)]
    out.write_text("".join(line + "\n" for line in lines))
    out.with_suffix(".summary.json").write_text(
        _summarise(round_, [m for _, m in manifests], len(lines)).model_dump_json(indent=2)
    )
    return out


def _summarise(round_: int, manifests: list[EpisodeManifest], examples: int) -> TrainingSetSummary:
    # A demonstration is a dev episode the teacher drove at least one step of.
    demos = [m for m in manifests if m.split == "dev" and any(s.controller == "teacher" for s in m.steps)]
    verdicts = [m.validation for m in demos if m.validation is not None]
    rejected = [v for v in verdicts if not v.accepted]
    return TrainingSetSummary(
        round=round_,
        episodes_scanned=len(manifests),
        demonstrations=len(demos),
        accepted=len(verdicts) - len(rejected),
        rejected=len(rejected),
        unfinished=len(demos) - len(verdicts),
        rejected_by=dict(Counter(c for v in rejected for c in failure_categories(v.reason) or ["other"])),
        # A teacher that stops before its first step leaves no teacher step, so count these rather than drop them.
        empty_dev_episodes=sum(1 for m in manifests if m.split == "dev" and not m.steps),
        examples=examples,
    )
