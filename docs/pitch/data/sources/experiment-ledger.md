# Experiment ledger (Sun 8 Nov 2026)

Every result has a comparison on the same seeds and conditions, with one variable changed. Student results are held-out seeds 1–20, student alone, successes / (n − errored).

| # | Claim | Result | Comparison | Same conditions? | Status |
|---|---|---|---|---|---|
| 1 | Teaching works: the student learns the task | v1 (52 demos) 10/20 (9/20 re-run on phase-1 code) | base 0/20 | same 20 held-out seeds, fixed_feet | done |
| 2 | Recovery and box-moved demos make the student adapt | v2 (117 demos), box moved before step 1: 15/20 | v1: 7/20 | same seeds, same move | done |
| 3 | Bin moved right after the lift | v2 10/20 (moved in 20/20, adapted 10/20) | v1 12/20 (adapted 12/20); v2 unmoved 10/20; base 0/20 (never lifts) | same seeds; v1 vs v2 n.s. (McNemar p=0.73) | done |
| 4 | Unseen bin placements | v2 21/40 (ahead 14/20, far 7/20) | v1 14/40 (ahead 4/20, far 10/20); base 0/40 | same seeds; bin further ahead: v2 better, p=0.006 | done |
| 5 | One sentence → a new skill (tasks.learn) | trained **18/20**; bin moved **17/20** (adapted 17/20) | base on the same task: **0/20**; bin moved 0/20 (never lifts) | same seeds; every count traced to a saved manifest | done |
| 6 | Diagnose → teach → better (Gate E round 1, run gate-e-1008) | round 1: pending | round 0 = v2 re-examined in the same run: **10/20** (matches v2's earlier 10/20) | same seeds, same run | running since 10:08: 3/10 dev trials failed, all diagnosed as "twisted past the bin to the 150° limit"; 20 teacher demos being collected |
| 7 | Round-level diagnosis | round 2 | round 1 | different starting checkpoints: confounded, stated as such | PR #37 open (19/19 mutants caught) |
| 8 | Image-only walking rules make walking learnable | walking student on new-rule demos (86 accepted, 1302 examples, 5 epochs): **0/20** | old-rule demos (109 accepted, 1453 examples, 5 epochs): 2/20; base 0/20 | same seeds, same code and epochs; sizes close, stated | done: **no gain**. It loops on `turn 5` (can't judge centred); the old one looped on `walk 10`. A memoryless student can't learn a fine visual stopping rule from ~100 demos |
| 9 | Fast enough on a laptop | idle 5070, sustained 35-min exams: median ~1.08 s, p95 ~1.1 s (1 over 2 s: first decision, warm-up) | 2 s budget; short runs (same code, now): ~0.5 s | idle machine; old and new code both ~0.5 s in short runs, so not a code regression | done |
| 11 | The student acts on what it sees (perception-driven) | v2 sighted 9/20 (Mac) | v2 shown a grey image: **0/20** (twists 45° once and places) | same checkpoint, seeds, device; only the image differs | done |
| 12 | The showcase student acts on what it sees | showcase sighted **19/20** (Mac) | the same student shown a grey image: **0/20** | same checkpoint, seeds, device; only the image differs | done |
| 13 | How many demos a new skill needs (learning curve) | 56 accepted demos (605 examples): **18/20**; bin moved 17/20 | 28 demos (293): 0/20; 14 demos (152): 0/20; bin moved 0/20 for both | same task, seeds, code and settings; only the demo count changes (first K accepted by seed) | done: with 14 or 28 the student twists to the 150° limit and never places |
| 10 | The new walking rules don't hurt the teacher | new rules: 14/20 accepted (seeds 7020–7039) | old rules on the same seeds: 16/20 | same seeds | done |

## Notes
- v2's first exam ran while Argon collectors loaded the laptop. The idle re-run gave the same latency (~1.08 s), so that load wasn't the cause; short runs give ~0.5 s.
- The showcase's first exam (18/20, 17/20) ran with the pipeline code from before Owen's review. The re-run with the new code reproduced it seed for seed (fails on 3 and 10; moved: 4, 10 and 14), and every number traces to a saved manifest.
- v1 vs v2 paired by seed (exact McNemar): only fixed_feet_ahead differs significantly (v2 better, p=0.006).
- The trained fixed-feet student runs a near-fixed routine (reach, hug, lift, twist 45° twice, place). #11 shows its stopping decision comes from the image: blind, it stops after one twist every time. Fixed-feet bin moves stay inside the 97–112° arc, so a student needn't re-aim much; the clips mustn't be shown as a re-aim by the student (Argon's takeovers do re-aim).
- Walking v1 (2/20) and the first loop round (0/20, a prompt bug) aren't pitch material. They stay in the record.
