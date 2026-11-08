"""Live demo for the presenter (Mac): the showcase student alone (no teacher, no Argon, no API calls) on
box_to_bin_fixed, on a video seed (>= 900000, enforced by the repo's refuse_seen_take). Prints each step's skill as it
happens and writes an MP4 (head camera | third person, caption strip, 2 fps as `mom demo`).

  --mode normal         one take, camera, the bin stays put
  --mode bin-moved      one take, camera, the bin moved right after the lift by the spec's own redraw
                        (same 85-112 deg x 0.40-0.50 m arc it was trained on: the routine still fits; NOT a re-aim)
  --mode camera-vs-grey the same scene twice: camera, then a fixed grey image instead of the camera; stitched side by
                        side, labelled "camera" | "grey image instead of the camera"

Run from this directory with the ws1-learn venv's python, e.g.
  PY=<ws1-scratchpad>/ws1-learn/.venv/bin/python
  $PY live_demo.py --mode camera-vs-grey --seed 900301

Every run records into a fresh run directory runs/live-<mode>-<seed>-<time>/ and a fresh MP4 (default videos/live/).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the model is in the local cache; no network calls
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.dont_write_bytecode = True

from bin_move import CKPT, GREY, SPEC, Grey, Narrated, episode  # noqa: E402
from fixed_move import FixedBinMoveSim  # noqa: E402

from isaacs_mom.orchestrator.demo import _captions, _ThirdPersonCapture, refuse_seen_take  # noqa: E402
from isaacs_mom.orchestrator.events import EventLog  # noqa: E402
from isaacs_mom.orchestrator.video import record  # noqa: E402
from isaacs_mom.student import StudentPolicy  # noqa: E402
from isaacs_mom.tasks.spec import TaskSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
CAPTION_CKPT = "showcase ckpt-r1"  # the full id runs past the caption strip
LEFT, RIGHT = "camera", "grey image instead of the camera"


def take(spec: TaskSpec, student: StudentPolicy, seed: int, *, grey: bool, move: bool, run_id: str, out: Path) -> dict:
    run_dir = HERE / "runs" / run_id
    refuse_seen_take(seed, run_dir)  # seed >= 900000, not held out, fresh run directory
    if out.exists():
        raise SystemExit(f"{out} exists")
    run_dir.mkdir(parents=True)
    # bearing None: a move is the repo's own MujocoSim.perturb redraw, unchanged
    capture = _ThirdPersonCapture(FixedBinMoveSim(spec.task_config(), None, None))
    sim = Narrated(capture)
    policy = Grey(student) if grey else student
    started = time.perf_counter()
    manifest, row = episode(
        sim, policy, spec, seed, run_dir, EventLog(run_dir, run_id), move=move, split="dev", demo=True
    )
    episode_dir = run_dir / "episodes" / manifest.episode_id
    (run_dir / "third_person").mkdir()
    for step_no, jpeg in capture.third_person.items():
        (run_dir / "third_person" / f"step_{step_no:04d}.jpg").write_bytes(jpeg)
    captions = [c.replace(CKPT, CAPTION_CKPT) for c in _captions(manifest)]
    out.parent.mkdir(parents=True, exist_ok=True)
    n = record(
        (
            (GREY if grey else (episode_dir / s.frame).read_bytes(), capture.third_person[s.step], c)
            for s, c in zip(manifest.steps, captions, strict=True)
        ),
        out,
        2,
    )
    row.update(
        view="grey image" if grey else "camera",
        video=str(out),
        frames=n,
        run_dir=str(run_dir),
        captions=captions,
        take_seconds=round(time.perf_counter() - started, 1),
    )
    verdict = "SUCCEEDED (accepted by the spec's judge)" if row["accepted"] else f"FAILED: {row['reason']}"
    print(
        f"  => {verdict}; turns {row['turn_seq'] or 'none'} (net {row['net_twist_deg']:g} deg); {n} frames -> {out}",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["normal", "bin-moved", "camera-vs-grey"])
    ap.add_argument("--seed", type=int, required=True, help="a video seed >= 900000")
    ap.add_argument("--out", type=Path, help="the MP4 to write (default videos/live/<mode>-<seed>-<time>.mp4)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    stamp = datetime.now().strftime("%H%M%S")
    out = args.out or HERE / "videos" / "live" / f"{args.mode}-{args.seed}-{stamp}.mp4"
    spec = TaskSpec.model_validate_json(SPEC.read_text())
    student = StudentPolicy(CKPT, runs_dir=HERE / "runs")
    print(
        f"student {CKPT} alone (no teacher), Mac float32; mode {args.mode}; seed {args.seed}\n"
        f"instruction: {spec.instruction}",
        flush=True,
    )
    rows = []
    if args.mode == "camera-vs-grey":
        halves = []
        for grey in (False, True):
            print(f"\n--- {RIGHT if grey else LEFT} ---", flush=True)
            half = out.with_name(out.stem + ("-grey" if grey else "-camera") + ".mp4")
            tag = "grey" if grey else "camera"
            rows.append(
                take(
                    spec,
                    student,
                    args.seed,
                    grey=grey,
                    move=False,
                    run_id=f"live-cvg-{tag}-{args.seed}-{stamp}",
                    out=half,
                )
            )
            halves.append(half)
        subprocess.run(
            [sys.executable, str(HERE / "stitch.py"), str(halves[0]), LEFT, str(halves[1]), RIGHT, str(out)], check=True
        )
    else:
        move = args.mode == "bin-moved"
        print(f"\n--- camera{', bin moved after the lift (spec redraw)' if move else ''} ---", flush=True)
        rows.append(
            take(
                spec, student, args.seed, grey=False, move=move, run_id=f"live-{args.mode}-{args.seed}-{stamp}", out=out
            )
        )
    total = round(time.perf_counter() - t0, 1)
    with (HERE / "logs" / "live.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps({"mode": args.mode, "total_seconds": total, **row}) + "\n")
    print(f"\nvideo: {out}\ntotal {total} s", flush=True)


if __name__ == "__main__":
    main()
