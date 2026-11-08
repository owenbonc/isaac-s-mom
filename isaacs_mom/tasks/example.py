"""The task generator end to end (SPEC §3.7): instruction -> validated spec -> scene -> the rule's verdicts on a
scripted success and a scripted failure -> a replay that reproduces both.

    uv run python -m isaacs_mom.tasks.example "put the box on the shelf"            # Argon proposes (needs the key)
    uv run python -m isaacs_mom.tasks.example "put the box on the shelf" --offline  # a fixed proposal stands in
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from isaacs_mom.contracts import Controller, Observation, SkillCall, Validation
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.seeds import DEV_SEED_START
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.preview import OraclePolicy
from isaacs_mom.tasks.generate import Propose, argon_proposer, generate_task
from isaacs_mom.tasks.rules import judge
from isaacs_mom.tasks.spec import Refusal, TaskSpec
from isaacs_mom.teacher.argon import Argon
from isaacs_mom.validator import replay

BACK_OFF_CM = 30.0
# Stands in for Argon offline, and labelled so wherever it's printed (rule 6).
OFFLINE_PROPOSAL = {
    "name": "box_on_shelf",
    "scene": {
        "box": {"kind": "box", "id": "box", "half_m": [0.125, 0.15, 0.125], "mass_kg": 0.8, "offset_m": 0.06},
        "destination": {
            "kind": "shelf",
            "id": "shelf",
            "top_m": 0.45,
            "distance_m": [1.0, 1.4],
            "bearing_deg": [75, 285],
        },
    },
    "success": {
        "all_of": [
            {"fact": "on", "object": "box", "support": "shelf"},
            {"fact": "at_rest", "object": "box"},
            {"fact": "robot_standing"},
        ]
    },
    "perturbations": [{"object": "shelf"}],
}


class DropShortPolicy:
    """The scripted failure: hugs and lifts as the oracle does, backs away from the table and lets go there."""

    controller: Controller = "teacher"
    checkpoint: str | None = None

    def __init__(self, sim: MujocoSim) -> None:
        self.sim, self.oracle = sim, OraclePolicy(sim)
        self.backed_off = self.placed = False

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        if not self.sim.lifted and not self.placed:
            return self.oracle.act(obs, instruction, feedback)
        if not self.backed_off:
            self.backed_off = True
            return SkillCall(skill="walk_back", params={"distance_cm": BACK_OFF_CM})
        if not self.placed:
            self.placed = True
            return SkillCall(skill="place")
        return None


@dataclass(frozen=True)
class Outcome:
    verdict: Validation
    replayed: bool  # the recorded steps re-ran to the same actions and default verdict
    replayed_verdict: Validation  # the generated rule's verdict on the replayed end state


def run_example(spec: TaskSpec, sim: MujocoSim, run_dir: Path, seed: int) -> dict[str, Outcome]:
    """Record a scripted success and a scripted failure on `seed`, judge each by the spec's rule, replay each."""
    events = EventLog(run_dir, f"task-{spec.name}")
    outcomes = {}
    for label, policy in [("success", OraclePolicy(sim)), ("failure", DropShortPolicy(sim))]:
        manifest = run_episode(
            sim, policy, run_dir=run_dir, events=events, seed=seed, split="dev", round_=0, instruction=spec.instruction
        )
        verdict = judge(spec, sim.ground_truth())
        replayed = replay(sim, run_dir / "episodes" / manifest.episode_id / "manifest.json")
        outcomes[label] = Outcome(verdict, replayed, judge(spec, sim.ground_truth()))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instruction")
    parser.add_argument("--offline", action="store_true", help="a fixed shelf proposal instead of Argon")
    parser.add_argument("--seed", type=int, default=DEV_SEED_START)
    args = parser.parse_args()
    run_dir = Path("runs") / f"task-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.offline:
        print("offline: a fixed proposal stands in for Argon")
        propose: Propose = lambda _instruction: json.dumps(OFFLINE_PROPOSAL)  # noqa: E731
    else:
        argon = Argon(EventLog(run_dir, "task-generator"))
        argon.check()
        propose = argon_proposer(argon)
    spec = generate_task(args.instruction, propose)
    if isinstance(spec, Refusal):
        print("refused:\n  " + "\n  ".join(spec.reasons))
        return
    print(f"spec {spec.name} ({spec.fingerprint()}):\n{spec.model_dump_json(indent=2)}")
    sim = MujocoSim(spec.task_config())
    print(f"scene: {sim.sim_version}")
    for label, outcome in run_example(spec, sim, run_dir, args.seed).items():
        print(
            f"scripted {label}: {'accepted' if outcome.verdict.accepted else 'rejected'} ({outcome.verdict.reason}); "
            f"replay {'reproduced' if outcome.replayed else 'DIVERGED'}, "
            f"verdict {'the same' if outcome.replayed_verdict == outcome.verdict else 'CHANGED'}"
        )
    print(f"recorded in {run_dir}")


if __name__ == "__main__":
    main()
