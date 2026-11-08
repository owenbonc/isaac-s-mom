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
- **MuJoCo** (organisers' Option A), running in-process on every teammate's machine (Macs and the Windows machines). Everything runs locally; the student fine-tunes on the Windows RTX 5070 (§3.2).
- Two tasks, both available; walking is the default for `mom run` and `mom demo`:
  - **Walking** (`across_room`): hug a big box (30 × 25 × 25 cm, 0.8 kg) off a table with both arms, lift it, walk it to a green bin 1.0–1.4 m away, beside or behind the robot (never ahead, where the table is), and drop it in.
  - **Fixed feet** (`fixed_feet`): the same hug and lift, then twist at the waist to a bin beside the robot and drop it in. The robot doesn't walk.
- Tasks are named presets (`sim/mujoco/task.py`) chosen with `--task` on `mom run`, `mom demo`, the preview and the frame export; an unknown name is refused. A preset sets the box's size, mass and starting offset and the bin's distance and bearing, and refuses settings the G1 can't do. Presets: `across_room` (the default), `centred_box`, `near_bin` (0.7–0.9 m), `heavy_box` (2 kg), `small_box`, `far_bin` (1.6–2.2 m), all walking, and `fixed_feet` with its two evaluation layouts, `fixed_feet_ahead` (the bin at a bearing of 85–97°, where `fixed_feet` uses 97–112°) and `fixed_feet_far` (the bin 0.45–0.50 m away, where `fixed_feet` uses 0.43–0.45 m). The success rule is the same in every task. The task name is part of `sim_version`, so a manifest replays only on the task it was recorded in. FakeSim plays the default task only and refuses another `--task`.
- Robot: Unitree G1 (29 DoF) from MuJoCo Menagerie with its floating base. Arms are joint-position controlled, with collision pads on the hands, since Menagerie's hands are visual only. In walking tasks the legs run Unitree's pretrained walking policy (unitree_rl_gym `motion.pt`, BSD-3-Clause, run in numpy from the vendored weights `assets/unitree_g1_walk/policy.npz`) for the whole episode: the robot steps in place to hold its spot while the arms work, and walks or turns on command. In `fixed_feet` it balances in Unitree's soft-knee standing pose under joint-position control.
- Sensor: one head-mounted RGB camera, 320×240, rendered with `mujoco.Renderer`, is the only scene sensor: no depth, LiDAR, microphone, IMU or contact sensing. The robot also reads its own joint positions. The camera sees the hands at the box during the hug and the bin under the box at placing distance.
- Skills (whitelist): `reach_left`, `reach_right` (slide the arms' working point sideways, `distance_cm` default 5, within ±12 cm), `walk_forward`, `walk_back` (walking tasks only, `distance_cm` 0–150, default 30), `turn_left`, `turn_right` (`angle_deg`, default 30: in walking tasks the whole body turns by stepping, 0–180; in `fixed_feet` the waist twists, 0–45, within ±150°), `hug`, `lift`, `place` (lower the box into the bin, let go, settle), `reobserve`. `SkillCall` clamps parameters to the widest limit (`distance_cm` 0–150, `angle_deg` 0–180). Damped least-squares IK plus a joint-position controller turns an arm skill into simulator steps. A skill the task doesn't have, a parameter past the task's limit, an IK target out of reach, or a walk into the table or the bin is refused with a reason and nothing moves (rule 7).
- `MujocoSim` implements `TruthSim` (§4.1). Scene layout (box position on the table, bin position and heading) is a pure function of the seed and the task. `perturb` moves only a box resting on the table out of the robot's hands, and puts the bin only where it touches nothing (in walking tasks, at the task's distance from wherever the robot is then); otherwise it refuses and nothing moves. FakeSim follows the same rule.
- `FakeSim` (foundation) implements the same protocol with drawn frames and trivial physics so every other workstream and CI build without the real scene.
- The demo video is recorded from the rendered frames (head camera next to a third-person camera), not screen capture.

### 3.2 Student (workstream 4)
- SmolVLM2-500M-Video-Instruct, local: Mac MPS, or CUDA on the Windows RTX 5070 and 5060 machines. `predict_skill(obs, instruction, feedback) -> SkillCall`, output parsed as JSON and checked against the whitelist; an unparseable answer becomes `reobserve` and is logged.
- Done rule: `StudentPolicy.act` returns `None` after its second `reobserve` in a row, which ends the episode; two unparseable answers in a row therefore end it too. The student learns to stop from the teacher's demonstrations, which end with one `reobserve` (§3.3): each accepted demonstration also gives one closing training example of that second `reobserve` (§4.3).
- Fine-tuning: LoRA locally on the Windows RTX 5070 (CUDA), on training examples (§4.3). Each round produces a checkpoint id `<run_id>/ckpt-r<round>`, stored in `runs/<run_id>/checkpoints/ckpt-r<round>/` (gitignored); `ckpt-r0` is the base model.

### 3.3 Teacher (workstream 2)
- Argon (`gemini-4-argon`, Gemini API) with a typed tool adapter over `RobotSim`.
- `diagnose(failure) -> Diagnosis`: from the failed episode's frames, skill calls, and the validator's reason, a failure hypothesis with evidence and confidence.
- `demonstrate(seed, instruction, diagnosis) -> Episode`: drives the robot skill by skill from fresh camera frames until done or the step limit (20), reading the new frame after every skill so a mid-run change alters the next decision. A demonstration ends with exactly one `reobserve`, then stops; it never records two `reobserve`s in a row before the end (the student's done rule, §3.2).
- `choose_scenes(failures, n) -> list[int]`: the development seeds whose student trials the validator rejected, then fresh development seeds up to `n`. It reads only validator verdicts, never layout or ground truth (rule 1), and rejects held-out seeds. Stretch, if time allows: Argon picks seeds from the head-camera first frames of candidate development seeds.
- If Argon is unreachable (5xx, rate limit, transport error, unusable answer), the teacher asks again on the same frame, waiting between attempts (honouring `Retry-After`), up to 3 attempts per step; then the demonstration ends with its one closing `reobserve`. Over budget, it ends at once with its closing `reobserve`. Every attempt counts against the budget and is logged as an `argon_call`. A missing key or a setup error (any other 4xx) stops the run instead. `mom run` prints a summary of Argon calls used out of the budget, unavailable answers, and whether the budget ran out.

### 3.4 Validator and recorder (workstream 3)
- `validate_episode(episode_id) -> Validation` from simulator ground truth: box on the bin's floor, at rest (neither moving nor turning) for 1 s of sim time; the robot hasn't fallen, no joint-limit violation, no penetration. The reason lists every problem found, separated by "; ", in words and without measurements (rule 1): `robot fell`, `joint limit violated`, `penetration`, `box short of the bin`, `box past the bin`, `box left of the bin`, `box right of the bin`, `box short-left of the bin`, `box short-right of the bin`, `box past-left of the bin`, `box past-right of the bin`, `box on the rim`, `box still moving`. Directions are in the robot's torso heading at the end of the episode, so they match the head-camera view: short and past are nearer and farther than the bin along the heading, left and right are across it. An axis is named only when the box's centre is beyond `bin_inner_half` from the bin's centre along it; a miss within that on both axes is named by its larger one. An episode the recorder refused gets `recording error: <why>` and counts as its own rejection kind.
- Recorder writes `runs/<run_id>/episodes/<episode_id>/` (manifest §4.2, frames as JPEG, actions as `.npy`) and `runs/<run_id>/events.jsonl`. The manifest is rewritten after every step, so a crashed episode keeps its steps with `validation: null`; a step whose executed actions and timestamps differ in count, go back in time or aren't finite numbers is refused.
- `build_training_set(run_id, round) -> path` applies rules 2 and 3, and writes `training_r<round>.summary.json` (`TrainingSetSummary`: demonstrations seen, accepted, rejected by problem kind, unfinished, empty dev episodes, examples).
- `replay(sim, manifest_path)` re-executes the stored skill calls and perturbations from the stored seed and reports whether the executed actions and the final verdict reproduce; it sets `replay_valid`, and fails when a recorded perturbation no longer applies. `replay_accepted(sim, run_dir)` replays every accepted episode in a run and writes `replay_report.json` (`ReplayReport`: reproduced out of replayed, with a reason for each failure); an episode that can't be read, re-run or saved is recorded there as a failure and sets `replay_valid=false` instead of raising.

### 3.5 Orchestrator, dashboard, demo (workstream 5)
- People start runs from the dashboard's Teach box (below); `uv run mom run --instruction <text>` is what it launches, and what tests and CI run. It executes the loop (§3.6) and writes events. A step the recorder refuses doesn't stop the run: `run_episode` emits an `error` event and finishes that episode as rejected with `recording error: <why>`. A held-out evaluation episode that errors is reported as errored, not as a student failure: the success rate is successes out of (n − errored), with the errored count shown.
- Dashboard (FastAPI plus one HTML page, server-sent events): live head camera, the current controller (student checkpoint or teacher), each decision with its reasoning, validator verdicts, held-out success per round (successes out of n − errored, with errored episodes shown apart), each round's training-set summary (read from `training_r<round>.summary.json`), the run's replay result (reproduced out of replayed, from `replay_report.json`), and where each model runs with its measured latency.
  - Provenance (rule 6): a run on FakeSim is labelled placeholder; a run is live while events are arriving (the last within 20 s; `run_episode` emits a `heartbeat` event every 10 s while a policy decides, a skill executes or the teacher diagnoses, so a slow Argon call keeps the run live and a dead run stops claiming it) or a fine-tune started less than 30 minutes ago has not completed, recorded otherwise, so a run that dies mid-fine-tune stops claiming to be live. Each panel shows its label as a dot, with the word in its tooltip and accessible name, spelled out only where it differs from the run's label in the header (a live fine-tune on a run otherwise recorded); the legend lists the three dots. The live dot pulses slowly (about every 1.6–2 s, an opacity or soft-glow pulse) wherever it shows, and holds still under reduced motion; recorded and placeholder dots never move. An episode with a verdict is recorded even while its run is live, and nothing is live while the event stream is disconnected; the header then says the stream is disconnected and no decision reads as still running. While the student fine-tunes, the stage says so, with the time since the fine-tune started, and shows what it learns from: the last frames of the run's accepted teacher demonstrations so far, with the accepted and rejected counts and their n. Model and host names come from configuration and are labelled as configuration, not as measured data. A stream reconnect rebuilds the view from the start, never on top of what is shown; while it replays the events already written, the page neither shows nor announces the episodes it passes through, only where it settles. The header names the simulator in words (MuJoCo, FakeSim); its version and the task are in the legend's run line.
  - Each panel shows one episode at a time: its frames, decisions, diagnosis and verdict, never another episode's. A teacher demonstration shows the diagnosis it acts on, named by the diagnosed episode: the loop records it as `acting_on` (that episode's id) on the demonstration's `episode_started` event. The verdict appears as the log's last row and in the control bar; there is no separate validator panel. The control bar names the episode's kind and seed from its start, and while the view holds a step (after M or P) it names the controller and checkpoint of the frame on the stage. Until an episode's first frame arrives, the stage shows a frame-sized slate with the episode's kind and seed.
  - A run opens on its results, in two tabs. **Results** lists the held-out seeds, one row each, marked by how the seed's held-out episode changed between the first and the latest checkpoint: ↑ improved (failed, then succeeded) and ↓ regressed (succeeded, then failed) at full strength, ✓ always passed and ✗ still fails a step quieter; green marks the two that end in success and red the two that end in failure, and each mark has its own shape, so none is told apart by colour alone. A seed whose episode errored in either evaluation is a yellow ! and is counted apart (rule 6); a seed whose later evaluation hasn't run yet is a grey ·. Changes sort first (↓, ↑, then ✗, ✓). Above the rows sit the held-out figure and one count per mark, each a filter. Choosing a seed shows its held-out episodes, one per checkpoint, with a three-way switch in the seed's header (before, both side by side, after; both by default where the window is wide enough), and each opens that episode's full view: its frames, its decisions with their reasons, and its verdict. **Teaching** holds the episodes between the evaluations: the student trials, the diagnoses, the teacher demonstrations and the fine-tune. While a run is live, Now follows the episode in progress; when a run the view was following stops, the view stays where it was.
  - The held-out panel shows the first and the latest checkpoint's success, labelled in words on one line ("before fine-tuning → after round 1"), the checkpoint ids in that line's tooltip. Its large figure is the change in percentage points between them ("+55 pts", never a relative percentage), with the two counts beside it at body size: "4 → 15 of 20" when both have the same denominator (n − errored), "9/18 → 15/20" when they differ; the counts carry n, so there is no separate n line; the errored count shows only when one is non-zero ("2 errored", "2 → 0 errored"); the spoken text still says both rates. With one evaluation the figure is that rate ("20%", "4 of 20"), and an unchanged rate reads "+0 pts"; while an evaluation runs it says how many of its episodes have run, and an evaluation with n = 0 says so in words. A diagnosis's hypothesis is clamped to three lines so its evidence always shows. The reason behind the frame on the stage is shown in full; older decisions may be clamped.
  - The judged take is marked `demo: true` on its `episode_started` event and titled as the judged take on an unseen scene.
  - Verdicts are worded by episode kind: a teacher demonstration is "accepted for training" or "rejected, not used for training"; a held-out test, a student trial and the judged take "succeeded" or "failed", and a judged take says whether the student finished it alone or the teacher took over at step N.
  - Controls: Space always pauses and resumes the view, whatever has focus (events buffer while paused, and the pause key counts them); ← and → move between the seeds in Results and between the episodes in Teaching or in a seed; Move jumps to the nearest episode with a box or bin move (or stays on the shown one when it has one) and shows that move as a head-camera frame taken right before the move and the first frame after it, side by side, so the two differ only by the move (the orchestrator observes once more just before `perturb` and names that frame as `before_frame` on the `perturbation` event, relative to the run directory; it is never a step, never trained on and never replayed), with the first decision after the move as the caption, clamped to three lines; pressing Move again returns to where the view was, and while the move is shown its two frames are the largest thing on screen at every size; Diagnosis and Teacher jump, in Teaching, to the nearest episode with a diagnosis and a teacher demonstration; Latest (Now) returns to the episode in progress. The buttons are icons only: each one's name, what it does and its keyboard shortcut (Space, ←, →, D, T, M, End) show in its tooltip on hover or focus and in the legend, its name is its accessible label, and while paused the pause button shows the buffered count beside its icon; no shortcut acts while the focus is in a text field. The legend (every mark, icon and shortcut) opens from an "i" button and otherwise takes no space. The decision log follows new rows only when it is already scrolled to the bottom.
  - After a box or bin move, the stage marks it for the rest of that episode, so the presenter can press Move after the moment has passed.
  - The look is quiet and dark, sized like the Claude desktop app, with Devin's workspace as the contrast reference: a charcoal surround that still gives the rendered scene an edge, panels a step lighter with a hairline edge and a small radius, off-white body text (near 10:1, not maximal) and grey secondary text; body text 14 px and labels 12 px at 1280×720, and few sizes besides: emphasis is bold weight and colour, not size, and the one large number is the held-out change (above). Colour is a small marker on neutral grey (a dot, a thin rule, an icon, a key number, a short word), never a tinted surface: the student is blue, the teacher teal, and green and red mark success and failure only. Almost nothing is a capsule: a panel's provenance is a dot (its word only where it differs from the header's), a verdict and a controller are a dot or an icon and a word, the run picker is text and a chevron, and the buttons are bare icons. All text meets AA contrast. The head-camera frame is the largest thing on screen at every size (the filmstrip shows only where it doesn't shrink the frame, and leaves no empty band beneath it). On the frame, the skill being run sits in the top-left corner as its icon and its name, bold at body size, in a notch cut into the image rather than a floating capsule; the controller (the teacher, or the student and its checkpoint) is a small pill in the frame's bottom-right corner; nothing else covers the frame. In an episode's view the stage takes the left half at full height; beside it, one panel holds the held-out result and the diagnosis side by side, with the training sets and the replay result across its width beneath them, above the decision log; the move's before and after frames span the full width.
- **Teach and runs.** The dashboard serves a runs directory (`uv run mom dashboard --runs runs/`). A run picker in the header lists its runs, live first; the URL names the shown run, and everything on the page is that run's. The control bar has a text field ("Tell the robot what to learn", at most 200 characters) and a Teach button shown as an enter icon only (its name, Teach, in its tooltip and accessible label); Enter in the field does the same:
  - Teach sends the text to the dashboard server, which takes it only from its own page (a POST from 127.0.0.1 whose `Origin` is the dashboard's own). The server records it as the first event (`instruction`) of a new run directory, and the task generator (§3.7) judges it, its Argon call counted in that run's budget (rule 5).
  - An instruction for the task the loop trains on (the box inside the bin) starts that run with the instruction, as `mom run` would (MuJoCo, the loop's task, `fixed_feet` for now (§8), the Argon teacher, up to 3 rounds), on the machine serving the dashboard, and the view opens it. Any number of runs may run at once.
  - For the pitch the dashboard runs on Isaac's RTX 5070 Ti machine, and Owen views it from his Mac through an SSH tunnel on the same port (`ssh -L 8200:127.0.0.1:8200 <host>`, then `http://127.0.0.1:8200`), so the page stays its own origin on 127.0.0.1 and Teach works through it.
  - A refusal shows its reasons, and a valid task the loop doesn't train on says "Not yet: the loop teaches one task"; either way nothing starts, and the run is listed as not started, with why.
  - A run is taught, trialled and held-out tested on its own instruction (the same sealed seeds), and its rates are shown with that instruction. Typed text is shown escaped.
- `uv run mom demo --ckpt <id> --seed <s> --perturb-at <step>` runs the judged single take: student on an unseen scene (a seed ≥ 900000, outside the development and held-out seeds), a perturbation mid-run, re-observe, finish. It records into a run directory of its own (an existing one is refused), so the take never enters a training set. Its captions name a move only from the step where a recorded perturbation applied.
- Perturbations (`move_box`, `move_bin`) go through `TruthSim.perturb` and reach the agents only as a changed camera frame. Only an applied perturbation is recorded and emitted as a `perturbation` event; a refused one is an `error` event. The lift-triggered `move_bin` (§6 criterion 10) also runs in a stated share of teacher demonstrations (0.25 by default, `mom run --move-bin-share`), so accepted demonstrations show the bin found again and the plan changed after it moves; the trigger reads ground truth on the orchestrator side only (rule 1).

### 3.6 The teach → fine-tune loop
1. Evaluate the current student checkpoint on the 20 held-out seeds (success rate, n=20).
2. Run the student on 10 development seeds; collect failures.
3. Teacher diagnoses the failures and chooses 20 development seeds; demonstrates each; the validator accepts or rejects.
4. Build the training set from accepted teacher episodes of all rounds so far; LoRA fine-tune; new checkpoint.
5. Repeat from 1 until held-out success ≥ 80% or 3 rounds.
6. Replay every accepted episode of the run (`replay_accepted`) after the loop's results are saved.
7. If time allows, B1: the same budget of randomly chosen seeds demonstrated by the teacher, fine-tuned and evaluated the same way.

Demonstrations may also be collected in a parallel batch: several Argon teachers at once, each within its own `ARGON_CALL_BUDGET` (rule 5), on development seeds, then one fine-tune on everything the validator accepted (rules 2 and 3). A batch skips step 3's diagnosis and scene choice, so its results are reported as batch-taught, apart from the loop's rounds.

### 3.7 Task generator (workstream 1, extension)
The vision: a person types what the robot should learn, and an agent generates the success rule and the scene assets the student trains on, so the teacher can demonstrate it and the validator can judge it. For the hackathon the loop teaches one task; this is a prototype off the critical path.
- `generate_task(instruction) -> TaskSpec | Refusal` in `isaacs_mom/tasks/`. `TaskSpec{name, instruction, scene, success, perturbations}` covers a narrow, physically checked set of objects and placements: boxes the hug can hold, and a bin, shelf or platform the existing `place` can reach. Each spec is checked against what the G1 can reach and lift with the existing skills, as `TaskConfig` does.
- `success` is declarative data over named facts (the object inside a region; resting on top of another, in contact; at rest for 1 s; the robot not fallen), evaluated by our code with stated tolerances, never code the model writes. `perturbations` declares which object may move mid-run, and where to.
- Argon proposes the spec as JSON, parsed at the boundary. Anything outside the schema, an unsupported object, a setting past the robot's limits, a start where objects overlap or are unstable, or a rule already true at reset is refused with a reason.
- Rule 1: scene and rule data reach only the validator and orchestrator side. The generator's Argon call is separate from the teacher's, its output never enters the teacher's or the student's prompts, and its calls count against `ARGON_CALL_BUDGET` (rule 5).
- A hash of the spec is part of `sim_version`, so a replay runs on exactly the same task.
- One example works end to end: instruction → validated spec → runnable scene → correct verdicts on a scripted success and a scripted failure → a reproducible replay. The default task, `validate()` and `contracts.py` stay unchanged, and joining generated rules to `validate()` stays with workstream 3 for later. Scanned meshes, arbitrary objects and new motion parameters are later (§9).
- A generated task can be taught end to end: `uv run python -m isaacs_mom.tasks.learn "<instruction>" --demos N --workers K` generates the spec or refuses with reasons, collects demonstrations on development seeds of that task in a parallel batch (§3.6), judges each by the spec's own declarative rule (`tasks.judge`), trains only on the accepted ones, fine-tunes once, evaluates the fine-tuned student on the held-out seeds of that task by the same rule (successes out of n − errored), and reports the spec and its fingerprint, demonstrations accepted out of n, the example count, the fine-tune's metrics and the held-out result. `validate()`, `contracts.py`, the default task and `mom run` are unchanged.

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
    def perturb(self, p: Perturbation) -> bool: ...  # True when it applied; False changes nothing
    def render_third_person(self) -> bytes: ...  # JPEG, for the video
```
- `Observation{episode_id, step, sim_time, head_rgb_jpeg: bytes, joint_state: list[float]}`
- `SkillTarget{skill: Skill, params: dict[str, float]}`; `SkillCall(SkillTarget){reason: str}`, where `reason` is shown on the dashboard and never trained on; `ExecResult{ok, error, executed_actions: list[list[float]], timestamps: list[float]}`
- `Perturbation{kind: move_box | move_bin, seed}`
- `GroundTruth{box_xyz, bin_xy, bin_yaw, bin_inner_half, box_on_bin_floor, box_at_rest, robot_fallen, joint_limit_violation, penetration, robot_xy, robot_yaw}`, where `robot_xy` is the pelvis floor position and `robot_yaw` the torso heading, which the validator uses for the miss direction (§3.4)
- `Validation{task_success, physics_valid, reason}` is computed by the validator from `GroundTruth`.

### 4.2 Episode manifest (`manifest.json`)
`{episode_id, run_id, seed, split: dev|heldout, round, instruction, sim_version, robot_config, steps: [{step, sim_time, frame: "frames/0003.jpg", joint_state, skill_call, controller: student|teacher, checkpoint, actions: "actions/0003.npy", timestamps}], perturbations: [{before_step, perturbation: Perturbation}], takeover_step: int|null, diagnosis: Diagnosis|null, validation: Validation, replay_valid: bool|null}`

### 4.3 Training example (one JSON line)
`{image: "<path to frame>", instruction, feedback, target: SkillTarget}` (skill and params only, never the reason) from teacher steps of accepted dev episodes only. When an accepted demonstration's last step is the teacher's `reobserve`, one more example follows it: the same frame, feedback `reobserve: ok`, target `reobserve`, which is the second `reobserve` that ends the student's episode (§3.2).

### 4.4 Events (`events.jsonl`)
`{t, run_id, type, data}` with `type` in `episode_started, observation, decision, action_executed, perturbation, validation, diagnosis, finetune_started, finetune_completed, evaluation_result, argon_call, error, heartbeat, instruction`.

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
isaacs_mom/tasks/            task generator and end-to-end teaching of a generated task (WS1, extension, §3.7)
configs/                     heldout_seeds.json, instructions
docs/                        one-page summary, pitch
```

## 6. Acceptance criteria — Phase 1 (the hackathon)

1. **Gate A.** On a Mac, `MujocoSim.reset(seed)` → `observe()` returns a head frame that changes when the box moves; `hug` then `lift` raise the box off the table with physics while the G1 stays on its feet, and on the walking task the G1 walks the held box to the bin without falling; the same seed reproduces the same scene.
2. **Gate B.** The student reads a real frame and returns a whitelisted skill that the simulator executes; its latency is logged.
3. **Gate C.** The validator gives a correct verdict on one success and one failure; the episode is saved; `replay` reproduces it.
4. **Gate D.** The teacher diagnoses a genuine student failure and completes the task; a perturbation mid-demonstration changes its next decision, traced to the new frame.
5. **Loop.** At least one full round runs: 20 dev demonstrations, validator filtering, LoRA fine-tune, held-out evaluation before and after, with n stated. Rounds continue until ≥ 80% or 3 rounds.
6. **Gate E.** `uv run mom run` executes the loop end to end and the dashboard shows it live; `uv run mom demo` runs the judged single take.
7. **Rules.** Tests prove rules 1, 2, 3 and 7; the Argon budget stops calls at the cap.
8. **Deliverables.** A continuous ≤ 1-minute video from the simulator showing perception next to actions and one unscripted change; the one-page PDF (user and problem, features, architecture, where each model runs with hardware and latency, limits, critical failure mode, safety); the 5-minute pitch. The critical failure mode is a bad demonstration (the box dropped, the robot fell): the validator rejects it on screen, worded "rejected, not used for training", so it never trains the student.
9. `scripts/verify.sh` prints READY on `phase-1` and CI is green.
10. **Evaluation report.** `uv run mom eval` runs the base (`ckpt-r0`) and the fine-tuned checkpoint, student only (no teacher takeover), on the sealed held-out seeds in three conditions: the task (`--task`, the training distribution); the same seeds on two layouts the training task never uses (`near_bin` and `far_bin` for `across_room`, `fixed_feet_ahead` and `fixed_feet_far` for `fixed_feet`, §3.1); and the task with `move_bin` right after the student's first successful `lift` (read from ground truth on the orchestrator side, rule 1; the same trigger runs in teacher demonstrations, §3.5). Each condition reports successes out of (n − errored). A refused move ("not applied") and an episode that never lifts ("trigger not reached") are shown apart and don't count as adaptation tests. Each episode saves its checkpoint, task, `sim_version`, seed, frames, skill calls, decision latency with its device, perturbations and verdict, and latency is compared against the edge-device budget: the student decides within 2 s per step and fits in 3 GB of memory on a laptop (revisited once the first GPU measurements land, before the feature freeze). Reported only: the loop's stop rule (§3.6) stays on the held-out seeds of the task the loop trains on (`fixed_feet` for now, §8).

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
| Sun 10:00–12:00 | Whole-phase review on `phase-1` against SPEC §6 (orchestrated-build Phase 5) |
| Sun 12:00–13:00 | `phase-1` → `main` PR through the merge gate on Owen's go-ahead (Phase 6: merge main into phase-1 first, verify and a real run again), so the video, PDF and submission point at main |
| Sun 13:00 | Video and PDF |
| Sun 15:00 | Submitted |
| Sun 18:00 | Pitch |

## 8. Risks

| Risk | Fallback | Decide by |
|---|---|---|
| G1 hug slips, or the robot falls while walking | Lighter box, slower walking (the scripted policy succeeds on 40 of 40 walking seeds with 0 falls, and on 36 of 40 fixed-feet seeds) | Sat 21:00 |
| Rendering on CI | `MUJOCO_GL=osmesa`; real-scene tests skip with a reason when no GL is available | Sat 17:00 |
| SmolVLM2 too weak to fine-tune usefully | Report the measured result honestly; demo leans on the validated data pipeline | Sun 06:00 |
| Argon latency or tool mismatch | Fewer, coarser skills per call | Sat 23:00 |
| Remote Argon costs autonomy points | The demo run uses the fine-tuned local student; Argon only teaches | — |
| Walking, the default task, is unproven with the real teacher | Readiness trial: the real Argon teacher on the same 10 development seeds of `across_room` and `fixed_feet`, reporting accepted demonstrations and the median Argon calls per demonstration. Below 6/10 accepted or above 12 calls, Owen decides whether the loop runs `--task fixed_feet`; walking stays available either way. **Result (Sat night): walking 0/10 accepted at a median of 20 calls, `fixed_feet` 8/10 at 7. Owen chose `fixed_feet` for the loop and the judged take; walking returns if the teacher's skill guide (`ws2/teacher-skill-guide`) brings it to at least 6/10 at 12 calls or fewer** | Sat 23:00 |
| Fixed `hug`, `lift`, `place` can't be corrected by the teacher | Add only the parameter that the trial's rejected teacher demonstrations call for (e.g. grip width for slips, lift height for rim hits), with a per-skill cap and tests; no speed parameter | Sat 23:00 |

## 9. Later phases (not built now)

Joint-level student (behaviour cloning on recorded actions), Isaac GR00T fine-tuning on the exported dataset, failure-category curricula with ablations, more tasks and embodiments beyond the task generator's prototype (§3.7), scanned-in meshes and arbitrary objects, sim-to-real.
