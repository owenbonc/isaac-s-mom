"""WS4 rehearsal on FakeSim: oracle demonstrations -> training set -> Modal LoRA -> base vs fine-tuned, held out.

The oracle reads FakeSim ground truth, so this tests the fine-tuning pipeline only. Its numbers never go on the
dashboard or in the pitch (SPEC §2 rules 1 and 6).
"""

import argparse
import logging
import time
from pathlib import Path

from isaacs_mom.contracts import Controller, EpisodeManifest, Observation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.seeds import heldout_seeds
from isaacs_mom.sim.fake import BLOCK_HALF, FakeSim
from isaacs_mom.student import BASE_CHECKPOINT, StudentPolicy, finetune
from isaacs_mom.validator.training import build_training_set

DEMO_SEED_START = 5000


class Oracle:
    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        truth = self.sim.ground_truth()
        dy = truth.block_xy[1] - obs.joint_state[1]
        if abs(dy) > 0.01:
            skill = "approach_right" if dy > 0 else "approach_left"
            return SkillCall(skill=skill, params={"distance_cm": abs(dy) * 100})
        gap = truth.target_xy[0] - truth.block_xy[0]
        if gap > 0.01:
            behind = truth.block_xy[0] - BLOCK_HALF - obs.joint_state[0]
            return SkillCall(skill="push_forward", params={"distance_cm": min((behind + gap) * 100, 20)})
        # Ends the way the teacher should: one reobserve on the finished scene, then stop.
        return None if feedback.startswith("reobserve") else SkillCall(skill="reobserve")


def succeeded(manifest: EpisodeManifest) -> bool:
    return manifest.validation is not None and manifest.validation.accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="FakeSim rehearsal of WS4 fine-tuning")
    parser.add_argument("--demos", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    run_id = time.strftime("rehearsal-%Y%m%d-%H%M%S")
    run_dir = Path("rehearsal") / run_id
    events = EventLog(run_dir, run_id)
    sim = FakeSim()

    accepted = sum(
        succeeded(
            run_episode(
                sim,
                Oracle(sim),
                run_dir=run_dir,
                events=events,
                seed=seed,
                split="dev",
                round_=0,
                instruction=INSTRUCTION,
            )
        )
        for seed in range(DEMO_SEED_START, DEMO_SEED_START + args.demos)
    )
    training_set = build_training_set(run_dir, 1)
    print(
        f"oracle demos accepted {accepted}/{args.demos}, {len(training_set.read_text().splitlines())} training examples"
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint = finetune(training_set, BASE_CHECKPOINT, 1, checkpoint_dir=checkpoint_dir)
    if checkpoint == BASE_CHECKPOINT:
        print("nothing to train on")
        return
    print((checkpoint_dir / checkpoint / "metrics.json").read_text())

    seeds = sorted(heldout_seeds())[: args.eval_seeds]
    for ckpt in (BASE_CHECKPOINT, checkpoint):
        wins = sum(
            succeeded(
                run_episode(
                    sim,
                    StudentPolicy(ckpt, checkpoint_dir=checkpoint_dir),
                    run_dir=run_dir,
                    events=events,
                    seed=seed,
                    split="heldout",
                    round_=1,
                    instruction=INSTRUCTION,
                )
            )
            for seed in seeds
        )
        print(f"{ckpt}: {wins}/{len(seeds)} FakeSim held-out successes")
    print(f"rehearsal saved to {run_dir}")


if __name__ == "__main__":
    main()
