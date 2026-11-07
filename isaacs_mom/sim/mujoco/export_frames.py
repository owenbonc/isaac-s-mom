"""Labelled head-camera frames for smoke-testing the student: `uv run python -m isaacs_mom.sim.mujoco.export_frames`.

Runs scripted episodes on development seeds and writes `labels.jsonl` in the training-example format (SPEC §4.3),
with image paths relative to the output folder, plus a zip of the lot. The script reads ground truth, so these are
for testing prompts, parsing and latency only: they are not teacher demonstrations and never enter a training set
(SPEC §2 rule 3).
"""

import argparse
import shutil
from pathlib import Path

from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.seeds import assert_not_heldout
from isaacs_mom.sim.mujoco.preview import OraclePolicy
from isaacs_mom.sim.mujoco.sim import MujocoSim
from isaacs_mom.validator.training import training_examples

FIRST_SEED = 9000  # clear of the held-out seeds and of the dev seeds the loop uses

README = """Head-camera frames from the MuJoCo scene, each labelled with the next skill a scripted policy chose.

labels.jsonl: one line per frame, {image, instruction, feedback, target: {skill, params}}, the training-example
format of SPEC §4.3. `feedback` is the previous skill's result ("start" for the first). Images are 320x240 JPEGs.

Skills: reach_left / reach_right (slide the arms sideways, params.distance_cm), walk_forward / walk_back
(params.distance_cm), turn_left / turn_right (turn the whole body, params.angle_deg), hug, lift, place, reobserve.

The labels come from a script that reads simulator ground truth, so use them to test prompts, JSON parsing and
latency. Do not train the student on them: training data must be accepted teacher demonstrations (SPEC §2 rule 3).
Every other episode has the box moved before step 1, so some frames show the arms re-aligning.
"""


def export(out: Path, episodes: int) -> tuple[int, int]:
    """Episodes on fresh dev seeds; returns (accepted episodes, labelled frames)."""
    sim = MujocoSim()
    events = EventLog(out, "frames")
    accepted, lines = 0, []
    for i in range(episodes):
        seed = FIRST_SEED + i
        assert_not_heldout(seed)
        manifest = run_episode(
            sim,
            OraclePolicy(sim),
            run_dir=out,
            events=events,
            seed=seed,
            split="dev",
            round_=0,
            instruction=INSTRUCTION,
            perturb_at=1 if i % 2 else None,
        )
        examples = training_examples(Path("episodes") / manifest.episode_id, manifest)
        accepted += bool(examples)
        lines += [e.model_dump_json() for e in examples]
    (out / "labels.jsonl").write_text("".join(line + "\n" for line in lines))
    (out / "README.txt").write_text(README)
    return accepted, len(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("runs/frames"))
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    accepted, frames = export(args.out, args.episodes)
    archive = shutil.make_archive(str(args.out), "zip", args.out)
    print(f"{frames} labelled frames from {accepted}/{args.episodes} accepted episodes in {args.out} and {archive}")


if __name__ == "__main__":
    main()
