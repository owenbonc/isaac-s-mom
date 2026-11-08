# Live demo and simulation video

The showcase student alone (no teacher, no Argon, no network) on scenes it has never seen (seeds ≥ 900000, enforced by the repo's `refuse_seen_take`). Each run plays in a window while it runs (`ffplay`) and saves an MP4.

## Setup (once)

1. `uv sync --frozen` at the repo root. macOS on Apple Silicon is fine; no GPU or key is needed.
2. Copy the showcase checkpoint here. It is gitignored, like every checkpoint:
   `docs/live-demo/runs/learn-box_to_bin_fixed-57288f663c83-040045/checkpoints/ckpt-r1/` with `adapter_config.json` and `adapter_model.safetensors`, taken from the showcase run `runs/learn-box_to_bin_fixed-57288f663c83-040045` on the RTX laptop.
3. `brew install ffmpeg` for `ffmpeg` and `ffplay`.

## Run

```sh
cd docs/live-demo
./run.sh grey     # same scene twice: camera, then a grey image instead (about 30 s)
./run.sh live     # the student alone, bin moved after the lift, with opening and closing cards
./run.sh moved    # a second scene with the bin moved
./run.sh video    # re-film the simulation video (0.5x speed, file only)
```

| Run | Steps | Verdict |
|---|---|---|
| `grey`, camera (seed 900301) | hug, lift, turn 45°, turn 45°, turn back 45°, turn 45°, place | succeeded |
| `grey`, grey image (seed 900301) | reach, hug, lift, turn 45°, place | failed: box not in the bin |
| `live` (seed 900407) | reach, hug, lift, **bin moved** (20°, 16 cm), turn 45°, turn 45°, place | succeeded |

Each scene was re-run 2 to 6 times in separate processes on a Mac (float32) and gave identical steps every time.

## Files

- `sim_video2.py`: films one continuous take from the simulator at 1920×1080. It shows the head camera (all the student sees) beside a moving scene view, each decision the student outputs and how long it took, a snapshot of the bin before it moved, the action timeline and the validator's verdict. `--live` also plays it in a window; `--pair` runs camera then grey image.
- `live_demo.py`: the 2 fps step-by-step version (`--mode camera-vs-grey | bin-moved | normal`).
- `bin_move.py`, `fixed_move.py`: the episode helpers. The bin move is the repo's own `MujocoSim.perturb("move_bin")`, triggered right after the lift by `run_episode`; `fixed_move.py` only adds a fixed-target variant used in the 125° test.
- `showcase-spec.json`: the showcase task spec (`box_to_bin_fixed`).

## Honest framing

- The bin moves here stay within the arc the student trained on (85–112°), and the student keeps its learned two-twist routine; say "still delivers", not "re-aims". Moved to 125°, outside that arc, it delivered 1/20 (`docs/pitch/data/sources/bin-move-125-analysis.txt`).
- The scenes for the video and gallery were picked for large, visible moves from a scan of 30, where the student delivered 25, and in one of the 5 misses the bin never moved (`docs/pitch/data/sources/sim-video-scan.log`). The held-out rate is 17/20.
