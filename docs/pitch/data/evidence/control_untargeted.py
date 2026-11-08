"""Gate E's control: the same `mom run --from-run runs/fixed-v2 --rounds 1` on the same code, with the targeting
taken out. The teacher gets no diagnosis (no Argon diagnosis call, no lesson), and its demonstrations go to fresh
dev seeds (1520-1539) instead of the rejected trials. Same demo count, bin-move share, fine-tune and held-out exam."""

import sys

import isaacs_mom.orchestrator.loop as loop
from isaacs_mom.contracts import Diagnosis
from isaacs_mom.orchestrator import cli
from isaacs_mom.seeds import assert_not_heldout

_run_episode, _ArgonPolicy = loop.run_episode, loop.ArgonPolicy


def no_diagnosis(manifest, frames, argon):
    return Diagnosis(hypothesis="control: not diagnosed", evidence="", confidence=0.0, lesson="")


def fresh_scenes(failures, n, start):
    seeds = list(range(start + 20, start + 20 + n))
    for seed in seeds:
        assert_not_heldout(seed)
    return seeds


def episode(sim, policy, **kw):
    if kw.get("acting_on"):
        kw["acting_on"] = None
    return _run_episode(sim, policy, **kw)


loop.diagnose = no_diagnosis
loop.choose_scenes = fresh_scenes
loop.ArgonPolicy = lambda diagnosis, argon: _ArgonPolicy(None, argon)
loop.run_episode = episode
sys.argv = [
    "mom",
    "run",
    "--task",
    "fixed_feet",
    "--from-run",
    "runs/fixed-v2",
    "--rounds",
    "1",
    "--run-id",
    sys.argv[1],
]
cli.main()
