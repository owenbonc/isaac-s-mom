# Workstreams: what each owner builds

Read `AGENTS.md` and `SPEC.md` first. Each workstream owns its folder; everything else changes only through a PR that says why. The foundation ships each module as a working stand-in with its final signature, so `uv run mom run --sim fake` runs end to end today. Replace the stand-in, keep the signature, and keep that command working.

Start: `git switch -c <ws>/<topic> origin/phase-1`, `uv sync`, `uv run mom run --sim fake --student stub --rounds 1`. Agents in a worktree run `scripts/agent-setup.sh origin/phase-1` instead.

## 1. Simulation and robot — `isaacs_mom/sim/mujoco/`
- Replace `MujocoSim` (stub) with a real implementation of `TruthSim` (`isaacs_mom/sim/base.py`, SPEC §3.1, §4.1).
- Unitree G1 from MuJoCo Menagerie, floating base removed; table, red block, green target zone; head camera 320×240; a third-person camera for the video.
- Skills: `approach_left`, `approach_right`, `push_forward`, `retract`, `reobserve`, `distance_cm` 0–20 (already clamped by `SkillCall`). IK plus joint-position control; clamp to joint limits and report violations in `ExecResult` (rule 7).
- Layout is a pure function of the seed; `perturb` moves the block or target from `Perturbation.seed`.
- `ground_truth()` fills every `GroundTruth` field, including `block_at_rest` over the last 1 s of sim time.
- Done: SPEC §6 criterion 1 (Gate A) by Sat 19:00, else switch to Franka Panda and tell Owen. Tests run headless (`MUJOCO_GL=osmesa` in CI).

## 2. Argon teacher — `isaacs_mom/teacher/`
- Replace `diagnose`, `choose_scenes` and `ArgonPolicy.act` (SPEC §3.3). Model `gemini-4-argon` via the `google-genai` SDK; key from `GEMINI_API_KEY` in `.env`.
- Input is only the head-camera JPEGs, the instruction, skill history and the validator's reason. Never import `isaacs_mom.sim` or call `ground_truth` (rule 1).
- `choose_scenes` must keep calling `assert_not_heldout` (rule 2). Enforce `ARGON_CALL_BUDGET` and emit an `argon_call` event per call with latency (rule 5). When Argon is unreachable or over budget, `act` returns `reobserve` a few times, then `None`.
- Done: SPEC §6 criterion 4 (Gate D). Tests use a fake Argon client; no live calls in CI.

## 3. Validator and dataset — `isaacs_mom/validator/`
- Owns `rules.validate`, `Recorder`, `training.build_training_set` (working foundation versions with tests in `tests/test_foundation.py`). Add `replay(sim, manifest_path) -> bool` and set `replay_valid` in the manifest.
- Keep rules 2, 3 and 4 tested; tighten `validate` for the real scene (block at rest, contacts) once workstream 1 lands.
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
