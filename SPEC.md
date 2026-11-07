# Isaac's Mom — SPEC

AgentVerse 2026 (UCL AI Society), Robotics track (NVIDIA-supported). Hacking ends Sun 8 Nov 2026 18:00 London. Source notes: Notion hackathon workspace, "AgentVerse 2026 — UCL" and Resources › "Initial idea".

## 1. Purpose

Robots learn from demonstrations that humans record by teleoperation or motion capture. Isaac's Mom takes the human out: an **Argon teacher agent** watches a weak robot student fail in simulation, diagnoses why from what the robot's camera saw, demonstrates corrected behaviour by driving the simulated robot itself, keeps only the demonstrations an independent validator accepts, and **fine-tunes the student** on them. The loop repeats until the student succeeds on held-out scenes. The fine-tuned student then runs on its own, locally.

Pitch: "A robot teacher that finds the gaps in a robot's skills and creates its own lessons."

## 2. Non-negotiable rules

1. **One sensor.** The student and teacher perceive the scene only through the head RGB camera frame, plus the task instruction and feedback text. Simulator ground truth (object poses) reaches only the validator and the orchestrator's perturbations. The student and teacher take a `RobotSim` (reset, observe, execute); only the validator and orchestrator take a `TruthSim`, which adds `ground_truth` and `perturb`. `Observation` has no pose fields.
2. **Held-out seeds stay sealed.** `configs/heldout_seeds.json` is fixed at the start of the run. The teacher refuses a held-out seed, and the recorder refuses to put a held-out episode into a training set. Both have tests.
3. **Only accepted teacher steps train the student.** Student steps before the teacher's takeover are stored but labelled `controller="student"` and excluded from training. Only episodes the validator accepts (`task_success and physics_valid`) enter a training set.
4. **Executed actions, not intentions.** The recorder stores the joint targets the simulator actually executed at each step, with timestamps, alongside the skill call that produced them.
5. **Secrets.** Keys live in `.env` (gitignored) and never in code, logs, or the dashboard. Argon calls are capped per run (`ARGON_CALL_BUDGET`, default 800) and every call is logged with latency.
6. **No fabricated results.** Every number shown comes from saved run files and carries its sample size. The dashboard labels each panel live, recorded, or placeholder.
7. **Safe motion.** The controller accepts only whitelisted skills, clamps to joint limits, and reports a violation instead of executing it.

## 3. The system

### 3.1 Simulation (workstream 1)
- **MuJoCo** (organisers' Option A), running in-process on every teammate's Mac. Modal is used only for fine-tuning.
- Task: hug a big box (30 × 25 × 25 cm, 0.8 kg) off a table with both arms, lift it, twist to the bin beside the robot and drop it in.
- Robot: Unitree G1 (29 DoF) from MuJoCo Menagerie, free-standing: it keeps its floating base and balances on its own feet under joint-position control, in Unitree's soft-knee standing pose. It does not walk; it reaches the bin by twisting at the waist. Collision pads on the hands, since Menagerie's hands are visual only. A table, an orange box, a green bin on the robot's left.
- One head-mounted RGB camera, 320×240, rendered with `mujoco.Renderer`.
- Skills (whitelist): `reach_left`, `reach_right` (slide the arms' working point sideways, `distance_cm` 0–20, default 5, within ±12 cm), `turn_left`, `turn_right` (twist the waist, `angle_deg` 0–45, default 30, within ±150°), `hug`, `lift`, `place` (reach out over the bin, lower, let go, settle 2.5 s), `reobserve`. Damped least-squares IK plus a joint-position controller turns a skill into simulator steps; a skill whose IK target is out of reach, or that would pass a reach or twist limit, is refused with a reason and nothing moves (rule 7).
- `MujocoSim` implements `TruthSim` (§4.1). Scene layout (box position on the table, bin position and heading beside the robot) is a pure function of the seed.
- `FakeSim` (foundation) implements the same protocol with drawn frames and trivial physics so every other workstream and CI build without the real scene.
- The demo video is recorded from the rendered frames (head camera next to a third-person camera), not screen capture.

### 3.2 Student (workstream 4)
- SmolVLM2-500M-Video-Instruct, local (Mac MPS or the 4070 Ti laptop). `predict_skill(obs, instruction, feedback) -> SkillCall`, output parsed as JSON and checked against the whitelist; an unparseable answer becomes `reobserve` and is logged.
- Fine-tuning: LoRA on Modal A10G (`student/finetune_modal.py`), on training examples (§4.3). Each round produces a checkpoint id `ckpt-r<round>`; `ckpt-r0` is the base model.

### 3.3 Teacher (workstream 2)
- Argon (`gemini-4-argon`, Gemini API) with a typed tool adapter over `RobotSim`.
- `diagnose(failure) -> Diagnosis`: from the failed episode's frames, skill calls, and the validator's reason, a failure hypothesis with evidence and confidence.
- `demonstrate(seed, instruction, diagnosis) -> Episode`: drives the robot skill by skill from fresh camera frames until done or the step limit (20), re-observing after every skill so a mid-run change alters the next decision.
- `choose_scenes(diagnoses, n) -> list[int]`: picks development seeds that target the diagnosed failure. Rejects held-out seeds.
- If Argon is unreachable or over budget, the robot holds and re-observes, and the run reports it.

### 3.4 Validator and recorder (workstream 3)
- `validate_episode(episode_id) -> Validation` from simulator ground truth: box on the bin's floor, at rest for 1 s of sim time; the robot hasn't fallen, no joint-limit violation, no penetration.
- Recorder writes `runs/<run_id>/episodes/<episode_id>/` (manifest §4.2, frames as JPEG, actions as `.npy`) and `runs/<run_id>/events.jsonl`.
- `build_training_set(run_id, round) -> path` applies rules 2 and 3.
- `replay(sim, manifest_path)` re-executes the stored skill calls and perturbations from the stored seed and reports whether the executed actions and the final verdict reproduce; it sets `replay_valid`.

### 3.5 Orchestrator, dashboard, demo (workstream 5)
- One command, `uv run mom run`, executes the loop (§3.6) and writes events.
- Dashboard (FastAPI plus one HTML page, server-sent events): live head camera, the current controller (student checkpoint or teacher), each decision with its reasoning, validator verdicts, the success-rate table per round, and where each model runs with its measured latency.
- `uv run mom demo --ckpt <id> --seed <s> --perturb-at <step>` runs the judged single take: student on an unseen scene, a perturbation mid-run, re-observe, finish.
- Perturbations (`move_box`, `move_bin`) go through `TruthSim.perturb` and reach the agents only as a changed camera frame.

### 3.6 The teach → fine-tune loop
1. Evaluate the current student checkpoint on the 20 held-out seeds (success rate, n=20).
2. Run the student on 10 development seeds; collect failures.
3. Teacher diagnoses the failures and chooses 20 development seeds; demonstrates each; the validator accepts or rejects.
4. Build the training set from accepted teacher episodes of all rounds so far; LoRA fine-tune; new checkpoint.
5. Repeat from 1 until held-out success ≥ 80% or 3 rounds.
6. If time allows, B1: the same budget of randomly chosen seeds demonstrated by the teacher, fine-tuned and evaluated the same way.

## 4. Contracts

All shared types live in `isaacs_mom/contracts.py` (pydantic v2). A change to a contract updates every caller in the same PR.

### 4.1 Simulator protocols (`isaacs_mom/sim/base.py`)
```python
class RobotSim(Protocol):  # student and teacher
    def reset(self, seed: int) -> str: ...  # episode_id
    def observe(self) -> Observation: ...
    def execute(self, call: SkillCall) -> ExecResult: ...


class TruthSim(RobotSim, Protocol):  # validator and orchestrator only
    def ground_truth(self) -> GroundTruth: ...
    def perturb(self, p: Perturbation) -> None: ...
    def render_third_person(self) -> bytes: ...  # JPEG, for the video
```
- `Observation{episode_id, step, sim_time, head_rgb_jpeg: bytes, joint_state: list[float]}`
- `SkillTarget{skill: Skill, params: dict[str, float]}`; `SkillCall(SkillTarget){reason: str}`, where `reason` is shown on the dashboard and never trained on; `ExecResult{ok, error, executed_actions: list[list[float]], timestamps: list[float]}`
- `Perturbation{kind: move_box | move_bin, seed}`
- `GroundTruth{box_xyz, bin_xy, bin_yaw, bin_inner_half, box_on_bin_floor, box_at_rest, robot_fallen, joint_limit_violation, penetration}`
- `Validation{task_success, physics_valid, reason}` is computed by the validator from `GroundTruth`.

### 4.2 Episode manifest (`manifest.json`)
`{episode_id, run_id, seed, split: dev|heldout, round, instruction, sim_version, robot_config, steps: [{step, sim_time, frame: "frames/0003.jpg", joint_state, skill_call, controller: student|teacher, checkpoint, actions: "actions/0003.npy", timestamps}], perturbations: [{before_step, perturbation: Perturbation}], takeover_step: int|null, diagnosis: Diagnosis|null, validation: Validation, replay_valid: bool|null}`

### 4.3 Training example (one JSON line)
`{image: "<path to frame>", instruction, feedback, target: SkillTarget}` (skill and params only, never the reason) from teacher steps of accepted dev episodes only.

### 4.4 Events (`events.jsonl`)
`{t, run_id, type, data}` with `type` in `episode_started, observation, decision, action_executed, perturbation, validation, diagnosis, finetune_started, finetune_completed, evaluation_result, argon_call, error`.

## 5. Stack and layout

Python 3.12, `uv`, mujoco, pydantic, FastAPI, pytest, ruff, pyright. CI renders headless with `MUJOCO_GL=osmesa`.

```
isaacs_mom/contracts.py      shared types (foundation)
isaacs_mom/sim/base.py       RobotSim, TruthSim protocols (foundation)
isaacs_mom/sim/fake.py       FakeSim (foundation)
isaacs_mom/sim/mujoco/       MujocoSim, scene, IK, skills (WS1)
isaacs_mom/teacher/          Argon teacher (WS2)
isaacs_mom/validator/        validator, recorder, training sets, replay (WS3)
isaacs_mom/student/          SmolVLM2 inference and fine-tuning (WS4)
isaacs_mom/orchestrator/     loop, CLI, demo, dashboard (WS5)
configs/                     heldout_seeds.json, instructions
docs/                        one-page summary, pitch
```

## 6. Acceptance criteria — Phase 1 (the hackathon)

1. **Gate A.** On a Mac, `MujocoSim.reset(seed)` → `observe()` returns a head frame that changes when the box moves; `hug` then `lift` raise the box off the table with physics while the G1 stays on its feet; the same seed reproduces the same scene.
2. **Gate B.** The student reads a real frame and returns a whitelisted skill that the simulator executes; its latency is logged.
3. **Gate C.** The validator gives a correct verdict on one success and one failure; the episode is saved; `replay` reproduces it.
4. **Gate D.** The teacher diagnoses a genuine student failure and completes the task; a perturbation mid-demonstration changes its next decision, traced to the new frame.
5. **Loop.** At least one full round runs: 20 dev demonstrations, validator filtering, LoRA fine-tune, held-out evaluation before and after, with n stated. Rounds continue until ≥ 80% or 3 rounds.
6. **Gate E.** `uv run mom run` executes the loop end to end and the dashboard shows it live; `uv run mom demo` runs the judged single take.
7. **Rules.** Tests prove rules 1, 2, 3 and 7; the Argon budget stops calls at the cap.
8. **Deliverables.** A continuous ≤ 1-minute video from the simulator showing perception next to actions and one unscripted change; the one-page PDF (user and problem, features, architecture, where each model runs with hardware and latency, limits, critical failure mode, safety); the 5-minute pitch.
9. `scripts/verify.sh` prints READY on `phase-1` and CI is green.

## 7. Timeline (London)

| By | Done |
|---|---|
| Sat 17:00 | Foundation merged: contracts, sim protocols, FakeSim, skeleton loop on FakeSim, verify, CI |
| Sat 19:00 | Gate A |
| Sat 21:00 | Gate B |
| Sat 23:00 | Gate C |
| Sun 03:00 | Gate D |
| Sun 08:00 | Gate E, first full loop round |
| Sun 10:00 | Feature freeze |
| Sun 13:00 | Video and PDF |
| Sun 15:00 | Submitted |
| Sun 18:00 | Pitch |

## 8. Risks

| Risk | Fallback | Decide by |
|---|---|---|
| G1 hug slips or the robot tips while twisting | Lighter box, smaller twist range (the scripted policy succeeds on 36 of 40 seeds now) | Sat 21:00 |
| Rendering on CI | `MUJOCO_GL=osmesa`; real-scene tests skip with a reason when no GL is available | Sat 17:00 |
| SmolVLM2 too weak to fine-tune usefully | Report the measured result honestly; demo leans on the validated data pipeline | Sun 06:00 |
| Argon latency or tool mismatch | Fewer, coarser skills per call | Sat 23:00 |
| Remote Argon costs autonomy points | The demo run uses the fine-tuned local student; Argon only teaches | — |

## 9. Later phases (not built now)

Joint-level student (behaviour cloning on recorded actions), Isaac GR00T fine-tuning on the exported dataset, failure-category curricula with ablations, more tasks and embodiments, sim-to-real.
