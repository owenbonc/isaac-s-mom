"""Workstream 4: SmolVLM2 student and LoRA fine-tuning (SPEC §3.2)."""

from isaacs_mom.student.checkpoints import BASE_CHECKPOINT
from isaacs_mom.student.stub import StubStudent, stub_finetune

StudentPolicy = StubStudent
finetune = stub_finetune

__all__ = ["BASE_CHECKPOINT", "StubStudent", "StudentPolicy", "finetune", "stub_finetune"]
