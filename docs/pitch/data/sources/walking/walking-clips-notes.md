# Walking clips (across_room), 8 Nov 2026

## 1. Teacher clip: made and verified
| | |
|---|---|
| Files | `teacher-argon-walk-seed2059.mp4` (1280×640, 2 fps, 12 frames, 6 s); `teacher-argon-walk-seed2059.gif` (960 px, 0.97 MB, last frame held 2.5 s) |
| Shows | Head camera (left) and third person (right). The G1 lines its hands up, hugs and lifts the box, steps back, turns left 60°, walks 85 cm and then 10 cm twice to the bin (a third 10 cm step is refused by the walk skill: close enough), and places the box. Each step's caption gives the skill and Argon's own reason, and the last frame gives the validator's verdict. |
| Seed and episode | 2059, `walking-trial/batch-walk-0203/c5/episodes/ep-2059-c8323b` (one of the 69 of 100 walking demos the validator accepted) |
| Controller | teacher (Argon), recorded at collection time. No Argon or API calls were made for this clip. |
| Outcome | accepted: box at rest inside bin |
| Method | Replay, no model. The recorded skill calls are re-run in MuJoCo from seed 2059 (ws1-walk code; across_room's sim and validator are the same as at collection commit `b57b563`). Head and third-person frames are captured before each step, as `orchestrator.demo._ThirdPersonCapture` does, plus one end-state frame. Every step's executed actions, timestamps and ok flag match the recording (tolerance 1e-5), the replayed verdict equals the stored one, and the repo's own `validator.replay.replay()` gives `replay_valid = true`. See `teacher-argon-walk-seed2059.replay.json`. |

Caption for the pitch:
> **Teacher (Argon) demonstration, walking task:** the robot lifts the box, turns, walks about a metre with it and puts it in the bin. This is a replay of an accepted demo (seed 2059). The validator accepted 69 of the 100 walking demos Argon produced in this batch.

## 2. Student clip: none. No valid student success could be recorded.
- **The 2 successful held-out episodes** of the old-prompt walking student (`batch-walk-1108-0254/ckpt-r1`: 1453 examples, 5 epochs, LoRA r16) are **seed 1** (11 steps) and **seed 20** (9 steps), both "box at rest inside bin". They ran on the Windows RTX 5070 Ti laptop, worktree `isaac-s-mom-v6` at `b57b563`, run dir `runs\batch-walk-1108-0254\eval\episodes`, on CUDA in bf16.
  - seed 1: reach_left 5, hug, lift, walk_back 30, turn_left 90, walk_forward 65, walk_forward 10 ×2, walk_forward 10 (refused), place, reobserve
  - seed 20: hug, lift, walk_back 30, turn_left 90, walk_forward 65, walk_forward 10, walk_forward 10 (refused), place, reobserve
  - Their manifests and frames were **never copied to this Mac**. Only the checkpoint was, to `ws1-walk/runs/batch-walk-1108-0254/checkpoints/ckpt-r1`, and its metrics confirm 1453 examples and 5 epochs. So they could not be replayed here.
- **Live takes instead**: the same checkpoint, student alone (no teacher takeover, no Argon), on fresh video seeds 900101–900112, which pass the repo's `refuse_seen_take` check. It ran on this Mac's MPS in float32. **Result: 0 of 12 delivered.** Every take is the exam's main failure: it lifts, steps back and turns, walks toward the bin, then keeps doing `walk_forward 10` past the bin until the 20-step limit and never places.

| take | seed | outcome (validator) | steps |
|---|---|---|---|
| 1 | 900101 | box past-right of the bin; box still moving | 20 |
| 2 | 900102 | box past-left of the bin; box still moving | 20 |
| 3 | 900103 | box past-left of the bin; box still moving | 20 |
| 4 | 900104 | box past-right of the bin; box still moving | 20 |
| 5 | 900105 | box past-left of the bin; box still moving | 20 |
| 6 | 900106 | box past the bin; box still moving | 20 |
| 7 | 900107 | box past-left of the bin; box still moving | 20 |
| 8 | 900108 | box past the bin; box still moving | 20 |
| 9 | 900109 | box past-right of the bin; box still moving | 20 |
| 10 | 900110 | box past-right of the bin; box still moving | 20 |
| 11 | 900111 | box past-right of the bin; box still moving | 20 |
| 12 | 900112 | box past-left of the bin; box still moving | 20 |

0/12 fits a roughly 10% success rate (2/20 held out; the chance of 0 in 12 at 10% is about 28%). It doesn't show that the Mac (MPS, fp32) behaves differently from the exam (CUDA, bf16), and it doesn't rule that out either. Each take's video and manifest are under `student/` (`student/attempts/*.mp4`, `student/runs/*`, `student/student_attempts.jsonl`). These are failures and aren't pitch material.

Honest wording if walking is mentioned in the pitch (no clip):
> The walking student delivered the box in 2 of 20 held-out scenes (the untrained model: 0 of 20). Walking stays far harder than the fixed-feet task.

Only if the seed-1 or seed-20 episode is fetched from the Windows laptop and replayed to the same verdict can a clip be shown, captioned: "The student delivered in 2 of 20 held-out scenes; this clip is one of them."

## How to reproduce
Use the `ws1-walk/.venv` python, `HF_HUB_OFFLINE=1`, and run from this folder:
- `scripts/student_live.py student 900101`: live takes, stops after 2 successes or 12 attempts
- `scripts/replay_render.py <copied episode dir> <out.mp4> "teacher (Argon)" "<footer>" --reasons`: replay, check, render
- `scripts/make_gif.sh <in.mp4> <out.gif> 960`
