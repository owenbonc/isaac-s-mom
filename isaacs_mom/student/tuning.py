"""Fine-tuning (SPEC §3.2): package a training set, train on the local CUDA GPU, store the run's ckpt-r<round>."""

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import TrainingExample
from isaacs_mom.student.checkpoints import ADAPTER_FILES, MODEL_ID, checkpoint_id, valid_run_id
from isaacs_mom.student.prompt import format_target

Trainer = Callable[[list[dict[str, str]], dict[str, bytes]], dict[str, bytes]]
log = logging.getLogger(__name__)
NO_CUDA = "fine-tuning needs a CUDA GPU (the Windows RTX 5070 or 5060); --student stub or --rounds 0 runs without one"


def has_cuda() -> bool:
    """Checked at startup, so a run without a GPU stops before any episode instead of at its first fine-tune."""
    import torch

    return torch.cuda.is_available()


def local_trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
    from isaacs_mom.student.lora import lora_train

    return lora_train(examples, images, "cuda")


def package(training_set: Path) -> tuple[list[dict[str, str]], dict[str, bytes]]:
    """Each frame is read once, under a short key, however many examples share it."""
    keys: dict[str, str] = {}
    images: dict[str, bytes] = {}
    examples: list[dict[str, str]] = []
    for line in training_set.read_text().splitlines():
        row = TrainingExample.model_validate_json(line)
        if row.image not in keys:
            keys[row.image] = f"{len(keys):05d}.jpg"
            images[keys[row.image]] = Path(row.image).read_bytes()
        examples.append(
            {
                "image": keys[row.image],
                "instruction": row.instruction,
                "feedback": row.feedback,
                "target": format_target(row.target),
            }
        )
    return examples, images


def finetune(training_set: Path, previous_checkpoint: str, round_: int, *, trainer: Trainer = local_trainer) -> str:
    """Retrains from the base model on the cumulative set, so a checkpoint depends only on its training set."""
    run_dir = training_set.parent
    if not valid_run_id(run_dir.name):
        raise ValueError(f"run id {run_dir.name!r} cannot name a checkpoint: use letters, digits, '.', '_' or '-'")
    examples, images = package(training_set)
    if not examples:
        log.warning("%s has no accepted teacher steps; keeping %s", training_set, previous_checkpoint)
        return previous_checkpoint
    started = time.monotonic()
    files = trainer(examples, images)
    missing = [name for name in ADAPTER_FILES if name not in files]
    if missing:
        raise RuntimeError(f"trainer returned no {', '.join(missing)}")
    for name in files:
        if Path(name).name != name or name in {".", ".."}:
            raise RuntimeError(f"trainer returned a file name that is not plain: {name!r}")
    checkpoint = checkpoint_id(run_dir.name, round_)
    out = run_dir / "checkpoints" / f"ckpt-r{round_}"
    staging = out.with_name(f"{out.name}.tmp")
    old = out.with_name(f"{out.name}.old")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name, data in files.items():
        (staging / name).write_bytes(data)
    meta = {
        "checkpoint": checkpoint,
        "previous": previous_checkpoint,
        "model": MODEL_ID,
        "training_set": str(training_set),
        "examples": len(examples),
        "frames": len(images),
        "seconds": round(time.monotonic() - started, 1),
    }
    (staging / "meta.json").write_text(json.dumps(meta, indent=2))
    # Two renames, so a crash leaves either the old or the new complete adapter on disk.
    if out.exists():
        shutil.rmtree(old, ignore_errors=True)
        out.replace(old)
    staging.replace(out)
    shutil.rmtree(old, ignore_errors=True)
    log.info("fine-tuned %s on %d examples in %.0f s", checkpoint, len(examples), meta["seconds"])
    return checkpoint
