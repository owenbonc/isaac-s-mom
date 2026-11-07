"""Fine-tuning, local side (SPEC §3.2): package a training set, train on Modal, store ckpt-r<round>.

Every round retrains from the base model on the cumulative set, so a checkpoint depends only on its training set.
"""

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from isaacs_mom.contracts import TrainingExample
from isaacs_mom.student.checkpoints import ADAPTER_FILES, CHECKPOINT_DIR, MODEL_ID, checkpoint_id
from isaacs_mom.student.prompt import format_target

Trainer = Callable[[list[dict[str, str]], dict[str, bytes]], dict[str, bytes]]
log = logging.getLogger(__name__)


def modal_trainer(examples: list[dict[str, str]], images: dict[str, bytes]) -> dict[str, bytes]:
    raise NotImplementedError("Task 6 wires the Modal A10G trainer")


def package(training_set: Path) -> tuple[list[dict[str, str]], dict[str, bytes]]:
    """Training lines as prompt-ready examples, each distinct frame shipped once under a short key."""
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


def finetune(
    training_set: Path,
    base_checkpoint: str,
    round_: int,
    *,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    trainer: Trainer = modal_trainer,
) -> str:
    examples, images = package(training_set)
    if not examples:
        log.warning("%s has no accepted teacher steps; keeping %s", training_set, base_checkpoint)
        return base_checkpoint
    started = time.monotonic()
    files = trainer(examples, images)
    missing = [name for name in ADAPTER_FILES if name not in files]
    if missing:
        raise RuntimeError(f"trainer returned no {', '.join(missing)}")
    for name in files:
        if Path(name).name != name or name in {".", ".."}:
            raise RuntimeError(f"trainer returned a file name that is not plain: {name!r}")
    checkpoint = checkpoint_id(round_)
    out = checkpoint_dir / checkpoint
    staging = checkpoint_dir / f"{checkpoint}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name, data in files.items():
        (staging / name).write_bytes(data)
    meta = {
        "checkpoint": checkpoint,
        "previous": base_checkpoint,
        "model": MODEL_ID,
        "training_set": str(training_set),
        "examples": len(examples),
        "frames": len(images),
        "seconds": round(time.monotonic() - started, 1),
    }
    (staging / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out, ignore_errors=True)
    staging.replace(out)
    log.info("fine-tuned %s on %d examples in %.0f s", checkpoint, len(examples), meta["seconds"])
    return checkpoint
