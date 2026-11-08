"""The showcase student alone (no teacher, no Argon, no API calls) on box_to_bin_fixed, with the bin moved right after
the lift to one fixed bearing/distance (or not at all, or by the spec's own redraw).

Two modes:

  eval  - many seeds, one JSON line per episode (accepted by the spec's judge, full skill sequence, turns):
      python bin_move.py eval --seeds heldout --move 125@0.45 --label moved-125 --out-dir runs/exp-moved-125

  demo  - the live-demo command: one seed (>= 900000, via the repo's refuse_seen_take), prints each step's skill as it
          happens, writes the MP4 (head camera | third person, caption strip; 2 fps, as mom demo):
      python bin_move.py demo --seed 900311 --move 125@0.45 --out videos/live-125-900311.mp4

--move: "none" (the bin stays), "spec" (the repo's own redraw from the spec's [85,112] deg x [0.40,0.50] m arc), or
"<bearing_deg>@<distance_m>" (the bin is put there, facing the robot; same refusals as the repo's move).
--grey: the student is shown a fixed grey image instead of the head camera (the control); the video's head panel then
shows that grey image, i.e. what the student saw.

Run with the ws1-learn venv's python from this directory; checkpoint ids resolve against ./runs (a copy of the
checkpoint, so nothing is read from or written to another worktree's runs/).
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

from fixed_move import FixedBinMoveSim
from PIL import Image

from isaacs_mom.contracts import ExecResult, GroundTruth, Observation, Perturbation, SkillCall
from isaacs_mom.orchestrator.demo import _captions, _ThirdPersonCapture, refuse_seen_take
from isaacs_mom.orchestrator.episode import run_episode
from isaacs_mom.orchestrator.events import EventLog
from isaacs_mom.orchestrator.video import record
from isaacs_mom.seeds import heldout_seeds
from isaacs_mom.student import StudentPolicy
from isaacs_mom.tasks.rules import judge
from isaacs_mom.tasks.spec import TaskSpec

HERE = Path(__file__).resolve().parent
CKPT = "learn-box_to_bin_fixed-57288f663c83-040045/ckpt-r1"
SPEC = HERE / "showcase-spec.json"
_buf = io.BytesIO()
Image.new("RGB", (512, 384), (128, 128, 128)).save(_buf, format="JPEG")
GREY = _buf.getvalue()  # the same grey frame as blind/blind_vs_sighted.py


class Grey:
    """The student shown a fixed grey frame instead of the head camera (as blind/blind_vs_sighted.py)."""

    controller = "student"

    def __init__(self, student: StudentPolicy) -> None:
        self.student, self.checkpoint = student, student.checkpoint

    def act(self, obs: Observation, instruction: str, feedback: str) -> SkillCall | None:
        return self.student.act(obs.model_copy(update={"head_rgb_jpeg": GREY}), instruction, feedback)


class Narrated:
    """Passes everything to the wrapped sim; prints each executed skill and the bin move, and keeps them."""

    def __init__(self, sim, quiet: bool = False) -> None:
        self._sim, self.quiet = sim, quiet
        self.sim_version, self.robot_config = sim.sim_version, sim.robot_config
        self.steps: list[dict] = []
        self.move: dict | None = None

    def _say(self, text: str) -> None:
        if not self.quiet:
            print(text, flush=True)

    def reset(self, seed: int) -> str:
        self.steps, self.move = [], None
        return self._sim.reset(seed)

    def observe(self) -> Observation:
        return self._sim.observe()

    def execute(self, call: SkillCall) -> ExecResult:
        result = self._sim.execute(call)
        params = " ".join(f"{v:g}" for v in call.params.values())
        self.steps.append({"skill": call.skill, "params": dict(call.params), "ok": result.ok, "error": result.error})
        outcome = "ok" if result.ok else f"REFUSED: {result.error}"
        self._say(f"  step {len(self.steps) - 1:2d}  student: {call.skill} {params}".rstrip() + f"   -> {outcome}")
        return result

    def ground_truth(self) -> GroundTruth:
        return self._sim.ground_truth()

    def perturb(self, perturbation: Perturbation) -> bool:
        ok = self._sim.perturb(perturbation)
        gt = self._sim.ground_truth()
        x, y = gt.bin_xy
        self.move = {
            "applied": ok,
            "before_step": len(self.steps),
            "bin_xy": [round(x, 3), round(y, 3)],
            "bin_bearing_deg": round(math.degrees(math.atan2(y, x)), 1),
            "bin_dist_m": round(math.hypot(x, y), 3),
        }
        if ok:
            self._say(
                f"  *** bin moved (after the lift, before step {len(self.steps)}): now at "
                f"{self.move['bin_bearing_deg']:.0f} deg, {self.move['bin_dist_m']:.2f} m ***"
            )
        else:
            self._say("  *** bin move refused by the sim; nothing moved ***")
        return ok

    def render_third_person(self) -> bytes:
        return self._sim.render_third_person()


def parse_move(text: str) -> tuple[bool, float | None, float | None]:
    """(move at all, bearing, distance); bearing None with move=True is the spec's own redraw."""
    if text == "none":
        return False, None, None
    if text == "spec":
        return True, None, None
    bearing, dist = text.split("@")
    return True, float(bearing), float(dist)


def skill_text(steps: list[dict]) -> str:
    return " ".join(
        s["skill"] + "".join(f" {v:g}" for v in s["params"].values()) + ("" if s["ok"] else "(refused)") for s in steps
    )


def turns(steps: list[dict]) -> dict:
    calls = [s for s in steps if s["skill"] in ("turn_left", "turn_right")]
    net = sum((1 if s["skill"] == "turn_left" else -1) * s["params"].get("angle_deg", 30.0) for s in calls if s["ok"])
    return {
        "turn_calls": len(calls),
        "turns_ok": sum(s["ok"] for s in calls),
        "net_twist_deg": net,
        "turn_seq": " ".join(
            f"{'L' if s['skill'] == 'turn_left' else 'R'}{s['params'].get('angle_deg', 30.0):g}"
            + ("" if s["ok"] else "x")
            for s in calls
        ),
    }


def episode(
    sim: Narrated,
    policy,
    spec: TaskSpec,
    seed: int,
    run_dir: Path,
    events: EventLog,
    *,
    move: bool,
    split: str,
    demo: bool,
):
    t0 = time.perf_counter()
    manifest = run_episode(
        sim,
        policy,
        run_dir=run_dir,
        events=events,
        seed=seed,
        split=split,
        round_=0,
        instruction=spec.instruction,
        move_bin_after_lift=move,
        demo=demo,
        verdict=lambda gt: judge(spec, gt),
    )
    gt = sim.ground_truth()
    verdict = judge(spec, gt)  # the spec's own judge, on the end state (also what the manifest holds)
    return manifest, {
        "seed": seed,
        "accepted": verdict.accepted,
        "reason": verdict.reason,
        "skills": skill_text(sim.steps),
        **turns(sim.steps),
        "n_steps": len(sim.steps),
        "move": sim.move,
        "box_xyz": [round(c, 3) for c in gt.box_xyz],
        "bin_xy": [round(c, 3) for c in gt.bin_xy],
        "seconds": round(time.perf_counter() - t0, 1),
        "episode_id": manifest.episode_id,
        "steps": sim.steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "demo"])
    ap.add_argument("--move", default="none", help='"none", "spec", or "<bearing_deg>@<distance_m>", e.g. 125@0.45')
    ap.add_argument("--grey", action="store_true", help="show the student a fixed grey image instead of the camera")
    ap.add_argument("--seeds", default="heldout", help='eval: "heldout" (1-20) or a comma list')
    ap.add_argument("--seed", type=int, help="demo: one seed >= 900000")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out-dir", type=Path, help="eval: run directory (must not exist)")
    ap.add_argument("--out", type=Path, help="demo: the MP4 to write")
    ap.add_argument("--run-id", help="demo: run directory name under ./runs (default from seed and move)")
    ap.add_argument("--jsonl", type=Path, help="append one JSON line per episode here")
    ap.add_argument("--quiet", action="store_true", help="eval: don't print each step")
    ap.add_argument("--caption-ckpt", default="showcase ckpt-r1", help=f"demo: the name captions give {CKPT}")
    args = ap.parse_args()

    spec = TaskSpec.model_validate_json(SPEC.read_text())
    move, bearing, dist = parse_move(args.move)
    base = FixedBinMoveSim(spec.task_config(), bearing, dist)
    student = StudentPolicy(CKPT, runs_dir=HERE / "runs")
    policy = Grey(student) if args.grey else student
    view = "grey image" if args.grey else "camera"
    print(f"student {CKPT} alone (no teacher), {view}; bin move: {args.move}; device float32 on this Mac", flush=True)

    if args.mode == "eval":
        seeds = sorted(heldout_seeds()) if args.seeds == "heldout" else [int(s) for s in args.seeds.split(",")]
        split = "heldout" if all(s in heldout_seeds() for s in seeds) else "dev"
        run_dir = args.out_dir
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists: each condition gets a run directory of its own")
        run_dir.mkdir(parents=True)
        events = EventLog(run_dir, args.label)
        sim = Narrated(base, quiet=args.quiet)
        wins = 0
        for seed in seeds:
            print(f"seed {seed}:", flush=True)
            _, row = episode(sim, policy, spec, seed, run_dir, events, move=move, split=split, demo=False)
            row = {"label": args.label, "move_arg": args.move, "grey": args.grey, **row}
            wins += row["accepted"]
            line = json.dumps(row)
            print(
                "RESULT "
                + json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "label",
                            "seed",
                            "accepted",
                            "skills",
                            "turn_seq",
                            "turns_ok",
                            "net_twist_deg",
                            "reason",
                            "seconds",
                        )
                    }
                ),
                flush=True,
            )
            if args.jsonl:
                with args.jsonl.open("a") as f:
                    f.write(line + "\n")
        print(
            f"SUMMARY {args.label}: {wins}/{len(seeds)} accepted by the spec's judge "
            f"(move {args.move}, {view}, Mac float32)",
            flush=True,
        )
        return

    # demo: one recorded take, the repo's refusals for a judged take (seed >= 900000, not held out, fresh run dir)
    if args.seed is None or args.out is None:
        raise SystemExit("demo needs --seed and --out")
    tag = "none" if not move else ("spec" if bearing is None else f"{bearing:g}at{dist:g}")
    run_id = args.run_id or f"take-{tag}{'-grey' if args.grey else ''}-{args.seed}"
    run_dir = HERE / "runs" / run_id
    refuse_seen_take(args.seed, run_dir)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    run_dir.mkdir(parents=True)
    capture = _ThirdPersonCapture(base)
    sim = Narrated(capture)
    events = EventLog(run_dir, run_id)
    print(f"seed {args.seed}: {spec.instruction}", flush=True)
    manifest, row = episode(sim, policy, spec, args.seed, run_dir, events, move=move, split="dev", demo=True)
    episode_dir = run_dir / "episodes" / manifest.episode_id
    third_dir = run_dir / "third_person"  # kept, so the take can be re-rendered without re-running it
    third_dir.mkdir()
    for step_no, jpeg in capture.third_person.items():
        (third_dir / f"step_{step_no:04d}.jpg").write_bytes(jpeg)
    # The repo's captions name the full checkpoint id, which runs past the 1280 px strip; shorten only that name.
    captions = [c.replace(CKPT, args.caption_ckpt) for c in _captions(manifest)]
    frames = (
        (GREY if args.grey else (episode_dir / step.frame).read_bytes(), capture.third_person[step.step], caption)
        for step, caption in zip(manifest.steps, captions, strict=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = record(frames, args.out, 2)
    row = {
        "label": args.label,
        "move_arg": args.move,
        "grey": args.grey,
        "video": str(args.out),
        "frames": n,
        "run_dir": str(run_dir),
        "captions": captions,
        **row,
    }
    if args.jsonl:
        with args.jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
    verdict = "ACCEPTED" if row["accepted"] else "NOT accepted"
    print(
        f"{verdict} by the spec's judge: {row['reason']}\nturns: {row['turn_seq'] or 'none'} "
        f"(net {row['net_twist_deg']:g} deg)\nvideo {args.out} ({n} frames at 2 fps), run saved to {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
