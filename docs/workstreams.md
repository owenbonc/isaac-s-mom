# Workstreams: what each owner builds

Read `AGENTS.md` and `SPEC.md` first. Each workstream owns its folder; everything else changes only through a PR that says why. The foundation ships each module as a working stand-in with its final signature, so `uv run mom run --sim fake` runs end to end today. Replace the stand-in, keep the signature, and keep that command working.

Start: `git switch -c <ws>/<topic> origin/phase-1`, `uv sync`, `uv run mom run --sim fake --rounds 1`. Agents in a worktree run `scripts/agent-setup.sh origin/phase-1` instead.

## 1. Simulation and robot — `isaacs_mom/sim/mujoco/`
- `MujocoSim` (`sim.py`) implements `TruthSim` (`isaacs_mom/sim/base.py`, SPEC §3.1, §4.1) over `scene.py`, with arm IK and motion in `control.py`.
- Free-standing Unitree G1 from MuJoCo Menagerie (vendored in `assets/`); table, orange box, green bin beside the robot; head camera 320×240; a third-person camera for the video.
- Skills: `reach_left`, `reach_right`, `turn_left`, `turn_right`, `hug`, `lift`, `place`, `reobserve` (`distance_cm` 0–20, `angle_deg` 0–45, clamped by `SkillCall`). Out-of-reach or past-the-limit skills are refused with a reason and nothing moves (rule 7).
- Layout is a pure function of the seed; `perturb` moves the box or the bin from `Perturbation.seed`.
- `ground_truth()` fills every `GroundTruth` field, including `box_at_rest` over the last 1 s of sim time.
- See it: `uv run python -m isaacs_mom.sim.mujoco.preview --seed 3 --gif runs/seed3.gif` films a scripted episode; `--viewer` opens MuJoCo's viewer.
- Done: SPEC §6 criterion 1 (Gate A), covered by `tests/test_mujoco_sim.py`. Tests run headless (`MUJOCO_GL=osmesa` in CI).

## 2. Argon teacher — `isaacs_mom/teacher/`
- Replace `diagnose`, `choose_scenes` and `ArgonPolicy.act` (SPEC §3.3). Model `gemini-4-argon` via the `google-genai` SDK; key from `GEMINI_API_KEY` in `.env`.
- Input is only the head-camera JPEGs, the instruction, skill history and the validator's reason. Never import `isaacs_mom.sim` or call `ground_truth` (rule 1).
- `choose_scenes` must keep calling `assert_not_heldout` (rule 2). Enforce `ARGON_CALL_BUDGET` and emit an `argon_call` event per call with latency (rule 5). When Argon is unreachable or over budget, `act` returns `reobserve` a few times, then `None`.
- Done: SPEC §6 criterion 4 (Gate D). Tests use a fake Argon client; no live calls in CI.

## 3. Validator and dataset — `isaacs_mom/validator/`
- Owns `rules.validate`, `Recorder`, `training.build_training_set` (working foundation versions with tests in `tests/test_foundation.py`). Add `replay(sim, manifest_path) -> bool` and set `replay_valid` in the manifest.
- Keep rules 2, 3 and 4 tested. `validate` now reads the box-in-bin facts from `GroundTruth`; tighten it against real MuJoCo runs.
- Done: SPEC §6 criterion 3 (Gate C).

## 4. Student VLM and fine-tuning — `isaacs_mom/student/`
- Replace `StudentPolicy.act` with SmolVLM2-500M-Video-Instruct (local, MPS on Macs). Prompt for one JSON `SkillCall`; parse with the pydantic model; anything unparseable becomes `reobserve` and is logged.
- Replace `finetune` with LoRA on a Modal A10G over the `training_r<round>.jsonl` file (SPEC §4.3); return `ckpt-r<round>` and make `StudentPolicy(checkpoint)` load it. Weights stay out of git.
- Done: SPEC §6 criterion 2 (Gate B), then the fine-tune half of criterion 5.

## 5. Orchestrator, dashboard and demo — `isaacs_mom/orchestrator/`, `docs/`
- Owns `episode.run_episode`, `loop.run_loop`, the CLI and `events.EventLog` (working foundation versions).
- Add the dashboard (FastAPI plus one HTML page, server-sent events tailing `events.jsonl`): head camera, controller, decisions, validator verdicts, the per-round success table with n, and where each model runs with latency. Label live, recorded or placeholder. Use the `impeccable` skill.
- Add `mom demo --ckpt --seed --perturb-at` for the judged single take, recording head and third-person frames to an MP4.
- Deliverables: the ≤1-minute video, the one-page PDF, the 5-minute pitch (SPEC §6 criterion 8).
- Done: SPEC §6 criterion 6 (Gate E).

## Shared contracts
`isaacs_mom/contracts.py`, `isaacs_mom/sim/base.py`, `isaacs_mom/policy.py`, `isaacs_mom/seeds.py`, `configs/heldout_seeds.json`. Change them only in a PR that updates every caller and says so in its body.
