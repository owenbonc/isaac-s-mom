"""Held-out seeds are sealed (SPEC §2 rule 2): the teacher and the training sets refuse them."""

import json
from functools import cache
from pathlib import Path

HELDOUT_PATH = Path(__file__).resolve().parent.parent / "configs" / "heldout_seeds.json"
DEV_SEED_START = 1000


class SealedSeedError(ValueError):
    pass


@cache
def heldout_seeds() -> frozenset[int]:
    return frozenset(json.loads(HELDOUT_PATH.read_text()))


def assert_not_heldout(seed: int) -> None:
    if seed in heldout_seeds():
        raise SealedSeedError(f"seed {seed} is held out")
