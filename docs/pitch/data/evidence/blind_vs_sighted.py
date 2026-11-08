"""Control: does the student use its camera? The same checkpoint on the same held-out seeds, once with the real head
frame and once with a fixed grey frame. Same device (this Mac), same sim, same seeds; only the image differs."""

import io
import json
import sys
from pathlib import Path

from PIL import Image

from isaacs_mom.contracts import Observation, SkillCall
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.loop import INSTRUCTION
from isaacs_mom.seeds import heldout_seeds
from isaacs_mom.sim.mujoco import MujocoSim
from isaacs_mom.sim.mujoco.task import TASKS
from isaacs_mom.student import StudentPolicy

checkpoint, task, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
buf = io.BytesIO()
Image.new("RGB", (512, 384), (128, 128, 128)).save(buf, format="JPEG")
GREY = buf.getvalue()


class Blind:
    controller = "student"

    def __init__(self, student: StudentPolicy) -> None:
        self.student, self.checkpoint = student, student.checkpoint

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return self.student.act(obs.model_copy(update={"head_rgb_jpeg": GREY}), instruction, feedback)


student = StudentPolicy(checkpoint, runs_dir=Path("runs"))
if task.endswith(".json"):  # a generated task's spec: its own scene, and the person's words as the instruction
    from isaacs_mom.tasks.rules import judge
    from isaacs_mom.tasks.spec import TaskSpec

    spec = TaskSpec.model_validate_json(Path(task).read_text())
    sim, INSTRUCTION = MujocoSim(spec.task_config()), spec.instruction
    verdict = lambda gt: judge(spec, gt)  # noqa: E731
else:
    sim, verdict = MujocoSim(TASKS[task]), None
for label, policy in (("sighted", student), ("blind", Blind(student))):
    run_dir = out / label
    events = EventLog(run_dir, label)
    wins = 0
    for seed in sorted(heldout_seeds()):
        m = run_episode(
            sim, policy, run_dir=run_dir, events=events, seed=seed, split="heldout", round_=0, instruction=INSTRUCTION
        )
        if verdict is not None:
            m.validation = verdict(sim.ground_truth())
        ok = bool(m.validation and m.validation.accepted)
        wins += ok
        skills = " ".join(
            s.skill_call.skill + "".join(f"{v:.0f}" for v in (s.skill_call.params or {}).values()) for s in m.steps
        )
        print(
            json.dumps(
                {
                    "label": label,
                    "seed": seed,
                    "accepted": ok,
                    "skills": skills,
                    "reason": m.validation.reason if m.validation else "",
                }
            ),
            flush=True,
        )
    print(f"SUMMARY {label}: {wins}/20", flush=True)
