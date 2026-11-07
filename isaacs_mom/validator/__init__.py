"""Workstream 3: validator, recorder, training sets, replay (SPEC §3.4)."""

from isaacs_mom.validator.recorder import Recorder
from isaacs_mom.validator.rules import validate

__all__ = ["Recorder", "validate"]
