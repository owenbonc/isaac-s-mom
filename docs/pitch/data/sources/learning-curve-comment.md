@owenbonc, the learning curve I announced here earlier, run on the `tasks.learn` showcase task ("Stay where you are and put the box in the bin beside you, even if someone moves the bin"). For each K, the student was trained on the first K accepted demos by seed, from the same 56 accepted. The exam is held-out seeds 1–20, student only, 0 errored. Same code (41daaee), same fine-tune settings; only K changes.

| Accepted demos | Examples | Fine-tune | Plain exam | Bin moved right after the lift |
|---|---|---|---|---|
| 14 | 152 | 404 s | 0/20 | 0/20 (applied 20/20, adapted 0/20) |
| 28 | 293 | 672 s | 0/20 | 0/20 (applied 20/20, adapted 0/20) |
| **56** | 605 | 1255 s | **18/20** | **17/20** (applied 20/20, adapted 17/20) |

**What fails at 14 and 28:** every episode ends "box not inside the bin; box still moving". The student hugs, lifts and twists, but never decides to stop and place. It twists to the 150° limit, then repeats refused turns until the step cap. At 56 it has learned when to stop. So for this skill, the stopping decision is what needs the extra demos.

**Two more controls**, both on the same checkpoint, seeds and machine (a Mac), with only the head frame replaced by a fixed grey image:
- showcase student (56 demos): 19/20 sighted, **0/20** blind;
- v2: 9/20 sighted, **0/20** blind.

Blind, both twist once and place. So the stopping decision comes from what they see.

**Walking, a negative result.** A student trained on demos from a new walking teacher prompt, where every turn is chosen from the current image (90° while the bin is out of view, 30° until it's above the box, 5° to centre it), scored **0/20**. Old-prompt walking demos gave 2/20.
- It had 86 accepted demos and 1302 examples, against the old set's 109 and 1453. Same code, 5 epochs.
- It loops on 5° turns, unable to judge "centred", where the old student looped on 10 cm walks.
- On the same 20 dev seeds, the teacher accepted 14/20 demos with the new prompt and 16/20 with the old one.
- That prompt stays on an unmerged experiment branch, `ws1/walk-reactive-teacher`, with no PR.
