# Scaffld: pitch kit

The numbers, captions and sources behind the deck ([`../deck/deck.pdf`](../deck/deck.pdf)), the [simulation video](../media/scaffld-simulation-video.mp4) and the [technical summary](../technical-summary.pdf). **Every number is measured, with its n and source.**

## Headline numbers

| Slide idea | Number | Control |
|---|---|---|
| **Teach a robot a new skill in 39.7 minutes** | one sentence → spec 24 s → 60 demos 15.9 min → fine-tune 21.2 min → exam 2.3 min (wall clock; a 70-min GPU queue excluded) | DROID: 50 collectors × 12 months of human demos |
| **One sentence → a new skill** | **18/20** unseen scenes, student alone | untrained: 0/20 |
| **...even when the bin moves after the lift** | **17/20** | untrained: 0/20 |
| **It acts on what it sees** (on a Mac) | **19/20** with its camera | **0/20** with a grey image |
| **Demos teach the decision** | 56 demos: **18/20** | 14 or 28 demos: 0/20 |
| **Lessons aimed at failures work** (fixed-feet v2 vs v1) | box moved at start: **15/20** | 7/20 (separate earlier run) |
| **No humans in the loop** | **1 sentence; 0 teleoperated steps; 0 labels** | DROID: 50 collectors × 12 months |
| **Runs on the edge** | **0.31 s** per decision (showcase exam, n = 326), **0 cloud calls** | teacher: 8.3 s and 9.8 cloud calls per task (n = 60) |

## Data

| File | What |
|---|---|
| [`data/workflow_timeline.csv`](data/workflow_timeline.csv) | The showcase run stage by stage, with start and end times (UTC) and sources |
| [`data/impact_comparison.csv`](data/impact_comparison.csv) | Human involvement, teacher calls, training cost, speed and success against each baseline |
| [`data/notes.md`](data/notes.md) | Where each number comes from, and what could not be measured |
| [`data/summary-numbers.json`](data/summary-numbers.json) | Every number in the technical summary, with its source |
| [`data/evidence/`](data/evidence/) | The teacher-latency script and output (n = 3,551), and the simulator and walking-policy bench |
| [`data/sources/`](data/sources/) | Raw logs: the showcase exam's events, the demo workers' events, the camera vs grey-image control, the 125° bin-move test, the 30-scene scan, the experiment ledger |

## Don't claim

- **"The student re-aims when the bin moves."** It keeps its learned routine. Moved beyond its training range (125°) it delivers 1/20. Within the range it still delivers 17/20: say "copes with the move". **Argon, the teacher, does re-aim.**
- **"Fully offline."** No cloud calls per decision, and the live demo loads the student from the local cache with `HF_HUB_OFFLINE=1`; the exam run's model load did contact Hugging Face. Say "no cloud calls".
- **Mixing Mac and GPU numbers.** The grey-image control (19/20 vs 0/20) ran on a Mac; the 18/20 exam on the laptop GPU.
- **The walking student as reliable.** It has its first successes (2/20 against 0/20), which is not significant. The teacher completes walking demos (69/100 accepted); the submitted loop uses `fixed_feet`.
- **DROID figures:** quote arXiv v2 (76k trajectories, 350 h, 50 collectors, 12 months). The RSS 2024 version says 65k.
