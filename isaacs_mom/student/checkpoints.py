"""Checkpoint ids and where their LoRA adapters live (SPEC §3.2)."""

import re
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BASE_CHECKPOINT = "ckpt-r0"
RUNS_DIR = Path("runs")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CHECKPOINT_ID = re.compile(r"(?P<run_id>[\w.-]+)/(?P<name>ckpt-r\d+)")


def checkpoint_id(run_id: str, round_: int) -> str:
    return f"{run_id}/ckpt-r{round_}"


def adapter_dir(checkpoint: str, runs_dir: Path = RUNS_DIR) -> Path | None:
    """Where a fine-tuned checkpoint's adapter belongs; None for the base model or an id of another shape."""
    match = CHECKPOINT_ID.fullmatch(checkpoint)
    if match is None or match["run_id"] in {".", ".."}:
        return None
    return runs_dir / match["run_id"] / "checkpoints" / match["name"]


def resolve_adapter(checkpoint: str, runs_dir: Path = RUNS_DIR) -> Path | None:
    """None for the base model, else the directory holding the checkpoint's complete adapter."""
    if checkpoint == BASE_CHECKPOINT:
        return None
    path = adapter_dir(checkpoint, runs_dir)
    if path is None:
        raise ValueError(f"checkpoint id {checkpoint!r} is neither {BASE_CHECKPOINT} nor <run_id>/ckpt-r<round>")
    missing = [name for name in ADAPTER_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint {checkpoint} has no adapter at {path} (missing {', '.join(missing)})")
    return path
