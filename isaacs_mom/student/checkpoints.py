"""Checkpoint ids and where their LoRA adapters live (SPEC §3.2). checkpoints/ is gitignored."""

from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BASE_CHECKPOINT = "ckpt-r0"
CHECKPOINT_DIR = Path("checkpoints")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def checkpoint_id(round_: int) -> str:
    return f"ckpt-r{round_}"


def resolve_adapter(checkpoint: str, checkpoint_dir: Path) -> Path | None:
    """None for the base model, else the directory holding the checkpoint's complete adapter."""
    if checkpoint == BASE_CHECKPOINT:
        return None
    path = checkpoint_dir / checkpoint
    missing = [name for name in ADAPTER_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint {checkpoint} has no adapter at {path} (missing {', '.join(missing)})")
    return path
