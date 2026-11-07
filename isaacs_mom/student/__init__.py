"""Workstream 4: SmolVLM2 student and LoRA fine-tuning (SPEC §3.2)."""

from isaacs_mom.student.checkpoints import BASE_CHECKPOINT, adapter_dir, valid_run_id
from isaacs_mom.student.policy import StudentPolicy
from isaacs_mom.student.stub import StubStudent, stub_finetune
from isaacs_mom.student.tuning import NO_CUDA, TRAINERS, finetune, has_cuda

__all__ = [
    "BASE_CHECKPOINT",
    "NO_CUDA",
    "TRAINERS",
    "StubStudent",
    "StudentPolicy",
    "adapter_dir",
    "finetune",
    "has_cuda",
    "stub_finetune",
    "valid_run_id",
]
