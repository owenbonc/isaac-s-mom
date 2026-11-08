# Pitch data: where each number comes from

Showcase run: `learn-box_to_bin_fixed-57288f663c83-040045`. Instruction: "Stay where you are and put the box in the bin beside you, even if someone moves the bin."

All times are UTC, as recorded:
- **Event times** are the `t` epoch field, converted to UTC.
- **Transcript times** are the JSONL `timestamp` field, already in UTC.
- **Times printed by PowerShell** on the RTX laptop (05:26:56, 05:50:24, 06:31:26, 06:44:03) are the laptop's local clock. That clock matches UTC to within 1 s: the last exam event is 05:50:23.586Z and the script printed "done … 05:50:24".

The run used two machines:
- **The Mac** ran the spec and the demonstrations.
- **The RTX 5070 Ti laptop** ran the fine-tune and the exams. Between the two, the batch was copied to the laptop and waited 69.8 min for its GPU.

Nothing was re-run for this file.

Path shorthands used in the CSVs:
- `SP` = `<ws1-scratchpad>` (temporary, so copy anything you need to keep)
- `RUN` = `SP/ws1-learn/runs/learn-box_to_bin_fixed-57288f663c83-040045` (the Mac copy; `SP/ws1-from/runs/…` holds only the same checkpoint, byte-identical)
- `TASKS` = `<ws1-session>/tasks` (the console output of that session's background jobs)
- `TR line N` = line N of `<session-logs>/932f67aa-da2c-45c0-ab36-1897570e6594.jsonl`
- `SUB` = `<session-logs>/932f67aa-da2c-45c0-ab36-1897570e6594/subagents/agent-a716d62a8f31ce6d9.jsonl` (the agent that ran showcase-compare)

## Workflow timeline

**Instruction, 04:00:21.493Z.** This is the `timestamp` of TR line 8348. That line is the tool call that ran `uv run python -m isaacs_mom.tasks.learn "<sentence>" --demos 60 --workers 8 --collect-only` on the Mac, as background task `bpto1v1qm`. The same sentence is the `instruction` field of `RUN/spec.json` and of all 60 demo manifests. The sentence was passed as a CLI argument by the team's Claude Code session, not typed into a UI.

**Task spec, 23.5 s (04:00:21.493Z → 04:00:44.998Z); 2 Argon calls.**
- `SP/ws1-learn/runs/learn-generator/events.jsonl` line 5 is the `startup` check, which ends at 04:00:40.548Z after `latency_ms` 10643.1, so it started at 04:00:29.905Z. Line 6 is the `task` call (the spec proposal), which ends at 04:00:44.998Z after `latency_ms` 4449.5.
- That makes 15.1 s of Argon time. The other 8.4 s is process start-up.
- `RUN/collected.json` and `SP/old/report.json` both give `generator_calls: 2`. The run dir's suffix `040045` is the Mac's local time, which is GMT, so UTC.
- **Excluded:** lines 1–4 of that file (03:59:20Z–04:00:04Z). They were a dry run of the generator on this sentence and the walking sentence (TR lines 8332 and 8337), not part of this run.

**Teacher demonstrations, 952.7 s (15.9 min).**
- The span runs from the first `episode_started` (`RUN/workers/w7/events.jsonl` line 1, 04:00:46.862Z) to the last `validation` (`w3` line 331, 04:16:39.565Z). The batch process's exit was notified at 04:16:40.529Z (TR line 8456).
- **Demos:** 60 attempted, 56 accepted. "Accepted" means `task_success` and `physics_valid` are both true in the 60 `RUN/episodes/*/manifest.json`, and in the 60 `validation` events. The 4 rejections are all "box not inside the bin".
- **Bin moves:** 30 demos had a `perturbation` event (`kind: move_bin`) and non-empty manifest `perturbations`, on the odd seeds 6001–6059 listed in `collected.json`. 28 of the 30 were accepted.
- **Replays:** `replay_valid` is true in 56 of the 56 accepted manifests, and `report.json` gives `replayed: 56`.
- **Argon calls:** 587 `argon_call` events (all `purpose: act`), matching `report.json` `argon_calls: 587`. 584 have `ok: true`. The 3 failures are `ServerError: 504 DEADLINE_EXCEEDED` after 58.1–58.3 s (w1 at 04:05:34Z, w6 at 04:05:47Z and 04:08:30Z).
- **Decisions:** 643 `decision` events. 584 carry a skill and come from Argon. 59 are local stops (`skill: null`, about 0.1 ms) with no call.
- **Steps:** the manifests hold 584 executed steps, and every one has `controller: teacher`.
- **Per demo:** each demo starts with `episode_started` and ends with `validation`, and `argon_call` events are attributed to the episode running in their worker. That gives a median of 8 calls per demo (mean 9.78) and a median of 89.6 s per demo.

**Copy to the laptop, 13.8 s, and the GPU queue, 69.8 min.**
- **Copy:** TR line 8464 (04:16:55.166Z) is a failed scp. TR line 8475 (04:17:08.941Z) shows "episodes: 60; collected: True".
- **Queue:** `--resume` was queued at TR line 8478 (04:17:13.961Z). It was re-queued at TR lines 9014–9018 (04:52:45Z–04:52:52Z), dropping the base exam. `TASKS/b2m56y8lt.output` prints "GPU free at 05:26:56". The GPU had been busy with the fixed_feet v2 `mom eval`, a different experiment.

**Training, bracket 1271.7 s; recorded 1265.6 s and 1255.3 s.**
- **Bracket:** from the resume launch (`TASKS/b2m56y8lt.output`: "resume … 05:26:56") to the first exam event (`SP/old/exam_events.jsonl` line 1, 05:48:07.726Z). The fine-tune's own start and end aren't logged.
- `RUN/checkpoints/ckpt-r1/meta.json` `seconds: 1265.6` times the whole `finetune` call (`student/tuning.py`, timer around `trainer(...)`). The same file gives `examples: 605` and `frames: 549`.
- `metrics.json` `seconds: 1255.3` times `lora_train`: model load, 5 epochs, a 64-example accuracy check and the save. The same file gives `peak_gpu_memory_gb: 2.0` (`torch.cuda.max_memory_allocated / 2**30`, so GiB), `trainable_params: 8683520`, `epoch_losses` 0.1563 → 0.0522 and `train_skill_accuracy` 54/64.
- `TASKS/b2m56y8lt.output` prints the same figures: "fine-tune: 1255.3 s, peak 2.0 GiB".
- The laptop ran code 65a6e97 ("learn worktree at 65a6e97").

**Evaluation, 135.9 s.**
- The plain exam's 20 episodes ran 05:48:07.726Z → 05:49:17.597Z (69.9 s) in `SP/old/exam_events.jsonl`. The bin-moved exam's 20 ran 05:49:17.803Z → 05:50:23.586Z (65.8 s) in `SP/old/exam_moved_events.jsonl`.
- Both files are copies of the laptop's `exam/…/events.jsonl` and `exam-bin-moved/…/events.jsonl`, fetched at 05:58Z (`SUB` line 121).
- **Scores:** 18/20 and 17/20, from the `validation` events and `SP/old/report.json` `exam` / `exam_bin_moved`. The second block gives `adaptation_tests: 20`, `adapted: 17`, `errored: 0`.
- **Decisions:** 165 + 161 = 326 `decision` events, all with `controller: student`, and 0 `argon_call` events.

**Untrained baseline and re-exam, 757 s.** This was a separate run, `showcase-compare`, on the same laptop.
- `SP/job2/job2.out` prints "start 06:31:26 … end 06:44:03", plus per-exam durations of 77, 69, 305 and 305 s. `SP/job2/compare_results.json` gives trained 18/20 and 17/20 (adapted 17/20), and base 0/20 and 0/20 (`trigger_not_reached: 20`).
- The code was 41daaee. The same failing seeds as the original exam are reported in PR #34, https://github.com/owenbonc/scaffld/pull/34#issuecomment-6718403706.

**Totals.**
- **Pipeline work:** 2,383.8 s (39.7 min), the sum of the four stage rows above.
- **Wall clock as run:** 6,602.5 s (110.0 min), from TR line 8348 to "done … 05:50:24".

## Impact comparison

**Human involvement.** The pipeline took 1 sentence of input. There were 0 teleoperated steps: the manifests have `driver: teacher` and every step has `controller: teacher` (584 steps), and the exam decisions are `controller: student` (326). There were 0 human labels: verdicts come from 60 + 40 `validation` events written by the spec's rule in code (`tasks/rules.py` `judge`), not by a person or an LLM. The operator ran 2 commands and 1 copy (TR lines 8348, 8464–8478 and 9018), only because the run was split across the Mac and the laptop.

**Teacher calls.**
- **Total:** 589 = 587 + 2 (`report.json`).
- **Per accepted demo:** 587 / 56 = 10.48, or 589 / 56 = 10.52 with the spec calls.
- **Teacher at deployment (b):** 9.78 calls per episode on average, median 8, p95 17, range 6–20 (60 episodes). The student made 0 calls in its 40 exam episodes.

**Seconds per decision and per episode.**
- **Student:** median 0.312 s and p95 0.557 s per decision (nearest rank, 326 decisions, `decision.latency_ms` in `SP/old/exam_*events.jsonl`). Median 3.17 s per episode (40 episodes).
- **Teacher in this batch:** median 8.29 s and p95 21.41 s per decision (584 Argon-made decisions, `decision.latency_ms` where the skill isn't null). Median 89.6 s per episode (60 episodes).
- **Already-measured figures,** as given:
  - Argon over all batches: 8.1 s median, 19.5 s p95, n=3,551 (teacher latency analysis, technical summary data.json; `<workspace>/technical-summary/data.json` `latency.rows[2]`: 8066 ms, 19473 ms).
  - Student: about 0.48 s median in short runs (n=277).
  - Student: about 1.08 s median over 35-minute exams (https://github.com/owenbonc/scaffld/pull/22#issuecomment-6718403477: 1075–1081 ms).

**Network.**
- **Teacher:** `SP/ws1-learn/isaacs_mom/teacher/argon.py` lines 91–98 need `GEMINI_API_KEY` and build a `google.genai` client for `gemini-4-argon` (line 19), so every teacher decision needs the network. 3 of the 587 calls in this batch timed out (see above).
- **Student:** its exam logged no Argon calls.
- **HF Hub:** loading the model printed "You are sending unauthenticated requests to the HF Hub" (`TASKS/b2m56y8lt.output`). So offline start-up was not shown.

**Training cost.** 1255.3 s, i.e. 20.9 GPU-minutes on one RTX 5070 Ti laptop GPU (1265.6 s for the whole fine-tune call), a 2.0 GiB peak, and 605 examples. The sources are as in the timeline.

**Learning curve.** https://github.com/owenbonc/scaffld/pull/22#issuecomment-6719281987, with the raw JSON at TR line 10420 (08:29:01Z):
- 14 demos: 404.1 s, 152 examples, 0/20 and 0/20.
- 28 demos: 672.4 s, 293 examples, 0/20 and 0/20.
- 14 and 28 were trained by `SP/curve.py` (runs `curve-k14` at 07:56:40Z and `curve-k28` at 08:10:47Z) on the same laptop with code 41daaee. `git diff 65a6e97 41daaee -- isaacs_mom/student isaacs_mom/validator/training.py` is empty, so the fine-tune code is the same as the showcase's.
- **The 56-demo point was not re-run.** It is the showcase's own fine-tune (1255.3 s), and `curve.py`'s docstring says so.

**Success.**
- **Trained student:** 18/20 plain and 17/20 with the bin moved (move applied 20/20).
- **Untrained, (a):** 0/20 and 0/20. Same seeds and laptop; `showcase-compare` ran both checkpoints with the 41daaee exam code.
- **Teacher, (b):** 56/60 (93.3%; 28/30 plain, 28/30 bin moved), but on dev seeds 6000–6059. The held-out seeds are sealed from the teacher, so this is not a same-conditions comparison.

**Human teleoperation, (c), DROID.** These quotes were checked today against https://arxiv.org/abs/2403.12945 (v2, revised 22 Apr 2025):
- "a diverse robot manipulation dataset with 76k demonstration trajectories or 350 hours of interaction data, collected across 564 scenes and 84 tasks by 50 data collectors in North America, Asia, and Europe over the course of 12 months". The same abstract says collection "requires substantial investments in hardware and human labour".
- The project page (droid-dataset.github.io) says the rig includes "an Oculus Quest 2 headset with controllers for teleoperation".
- **Version caution:** the RSS 2024 proceedings abstract (roboticsproceedings.org/rss20/p120.html, DOI 10.15607/RSS.2024.XX.120) says "65k demonstration trajectories or 350h of interaction data, collected across 564 scenes and 86 tasks". The project page says "76k … 350h … 86 tasks". "RSS 2024 + 76k" mixes versions. Cite arXiv v2 for 76k, or RSS 2024 for 65k.
- No per-demo human time is derived.

## What could not be measured, and why

1. **Argon tokens and cost.** `argon_call` events log only purpose, model, call, budget, ok, error and latency_ms, and `argon.py` never reads the response's usage data.
2. **Exact fine-tune start and end times.** These aren't logged. Only the 1271.7 s bracket and the recorded durations (1265.6 s and 1255.3 s) exist. The training-set build and process start-up fit inside the remaining ≤ 6.1 s, but aren't timed separately.
3. **The student's GPU memory in this exam.** The `--resume` exam doesn't record it. 2.01 GiB comes from other runs (35-minute `mom eval`, `data.json` `other.student_peak_gib`).
4. **Teacher vs student under the same conditions.** The teacher never runs on the held-out seeds 1–20, by design. Its numbers come from dev seeds, from the Mac, and from 8 teachers running at once; the student's come from the RTX laptop. So the ratios (about 27–28× per decision and per episode with this exam's student timings, or about 7.5× per decision against the 1.08 s 35-minute median) compare different conditions.
5. **Offline operation of the student.** It wasn't tested, and the model load contacted the HF Hub.
6. **One-command, one-machine end-to-end time.** The showcase was split into `--collect-only` on the Mac and `--resume` on the laptop, with a 69.8 min GPU queue. Without that queue the stages add up to 39.7 min, but no single uninterrupted run was timed.
7. **Human operator effort, as time or attention.** Nothing logs it. The commands came from an AI coding session acting for the team.
8. **Per-demo human time for teleoperation.** DROID's abstract gives totals only, and we don't derive a per-demo figure.
9. **Latency of the untrained student in showcase-compare.** Only its exam durations (305 s per 20 episodes) are on this machine. Its decision events stayed on the laptop.
10. **Energy and power.** They aren't logged.
