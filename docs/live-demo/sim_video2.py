"""The simulation video, v2: one continuous take (about 50 s), filmed from the simulator's own frames at 1920x1080.

The showcase student works alone (no teacher, no Argon, no network) on a scene it has never seen. Right after it lifts
the box, the simulator moves the bin, unannounced: the student learns of it only through its camera. On screen: the
head camera (all the student sees) beside a moving scene view, each decision the student outputs and how long it took,
a snapshot of the bin before it moved, the action timeline, and the validator's verdict. Played at a labelled speed.

  film:  python sim_video2.py --seed 900407                      # videos/sim-video2-<seed>.mp4, 0.5x speed
  live:  python sim_video2.py --seed 900407 --live --speed 1     # also plays in a window while it runs
  pair:  python sim_video2.py --seed 900301 --pair --live        # camera, then a grey image, in one window
"""

import argparse
import contextlib
import json
import math
import os
import subprocess
import textwrap
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the model is in the local cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io  # noqa: E402

import mujoco  # noqa: E402
from bin_move import CKPT, GREY, SPEC, Narrated, episode  # noqa: E402
from fixed_move import FixedBinMoveSim  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from isaacs_mom.orchestrator.demo import refuse_seen_take  # noqa: E402
from isaacs_mom.orchestrator.events import EventLog  # noqa: E402
from isaacs_mom.student import StudentPolicy  # noqa: E402
from isaacs_mom.tasks.spec import TaskSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
FPS, W, H = 25, 1920, 1080
SCENE = (40, 104, 1216, 684)  # x, y, w, h
HEAD = (1296, 130, 584, 438)
SUP = "/System/Library/Fonts/Supplemental/"
BG, PANEL, INK, MUTED = (13, 16, 22), (29, 33, 42), (242, 242, 238), (148, 151, 160)
BLUE, ORANGE, GREEN, RED = (64, 140, 230), (235, 104, 52), (38, 180, 116), (225, 70, 70)
GREY_IMG = Image.open(io.BytesIO(GREY)).convert("RGB")
_fonts: dict = {}


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold, mono)
    if key not in _fonts:
        path = "/System/Library/Fonts/Menlo.ttc" if mono else SUP + ("Arial Bold.ttf" if bold else "Arial.ttf")
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def skill_label(call) -> str:
    unit = {"angle_deg": "°", "distance_cm": " cm"}
    return (call.skill + " " + " ".join(f"{v:g}{unit.get(k, '')}" for k, v in call.params.items())).strip()


class Sink:
    """Frames to an MP4 and, live, to a window that plays them as they are made."""

    def __init__(self, out: Path, live: bool) -> None:
        args = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
        self.procs = [
            subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    *args,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(out),
                ],
                stdin=subprocess.PIPE,
            )
        ]
        if live:
            self.procs.append(
                subprocess.Popen(
                    [
                        "ffplay",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        f"{W}x{H}",
                        "-framerate",
                        str(FPS),
                        "-x",
                        "1440",
                        "-y",
                        "810",
                        "-window_title",
                        "Scaffld: live simulation",
                        *(["-nodisp"] if os.environ.get("NODISP") else []),
                        "-i",
                        "-",
                    ],
                    stdin=subprocess.PIPE,
                )
            )
        self.frames = 0

    def write(self, img: Image.Image) -> None:
        data = img.tobytes()
        for p in self.procs:
            with contextlib.suppress(BrokenPipeError):  # the live window was closed; the MP4 carries on
                p.stdin.write(data)
        self.frames += 1

    def close(self, keep_window: float = 0.0) -> None:
        for p in self.procs:
            with contextlib.suppress(BrokenPipeError):
                p.stdin.close()
        self.procs[0].wait()
        if len(self.procs) > 1:
            time.sleep(keep_window)
            self.procs[1].terminate()


class Film:
    """One take: frames from the sim's physics ticks at the chosen speed, plus held frames while the student decides."""

    def __init__(
        self, base: FixedBinMoveSim, sink: Sink, spec: TaskSpec, seed: int, speed: float, grey: bool, header: str
    ) -> None:
        self.base, self.sink, self.spec, self.seed, self.speed, self.grey, self.header = (
            base,
            sink,
            spec,
            seed,
            speed,
            grey,
            header,
        )
        base.model.vis.global_.offwidth, base.model.vis.global_.offheight = W, H
        self.head = mujoco.Renderer(base.model, 240, 320)
        self.scene = mujoco.Renderer(base.model, SCENE[3], SCENE[2])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (0.16, 0.18, 0.78)
        self.active, self.last_t, self.start_frame, self.t0 = False, -1.0, sink.frames, None
        self.doing, self.json, self.decide_s, self.thinking = "", "", None, False
        self.log: list[tuple[str, tuple]] = []
        self.marks: list[tuple[float, str, tuple]] = []  # (sim time, label, colour) on the timeline
        self.banner: tuple[str, tuple] | None = None
        self.banner_until = 0
        self.inset: Image.Image | None = None
        self.inset_until = 0
        self.card: list | None = None  # [(text, size, bold, colour)] centred over the scene
        self.verdict: tuple[str, tuple] | None = None
        base.on_tick = self.tick

    # ---- time -------------------------------------------------------------------------------------------------
    def t(self) -> float:
        return (self.sink.frames - self.start_frame) / FPS

    def tick(self) -> None:
        if self.active and self.base.data.time - self.last_t >= self.speed / FPS - 1e-9:
            self.last_t = self.base.data.time
            self.emit()

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds * FPS))):
            self.emit()

    def emit(self) -> None:
        self.sink.write(self.compose())

    # ---- drawing ----------------------------------------------------------------------------------------------
    def render_scene(self) -> Image.Image:
        t = self.t()
        self.cam.azimuth = -132.8 + 20 * math.sin(2 * math.pi * t / 40)
        self.cam.elevation = -21 + 3 * math.sin(2 * math.pi * t / 53)
        self.cam.distance = 2.05 + 0.25 * math.cos(2 * math.pi * t / 40)
        self.scene.update_scene(self.base.data, camera=self.cam)
        return Image.fromarray(self.scene.render())

    def compose(self) -> Image.Image:
        d = self.base.data
        img = Image.new("RGB", (W, H), BG)
        g = ImageDraw.Draw(img)
        # header
        g.text((40, 26), "Scaffld", font=font(40, True), fill=INK)
        g.text((210, 38), self.header, font=font(24), fill=MUTED)
        x = W - 40
        for text, colour in ((f"{self.speed:g}× speed", MUTED), ("no teacher · no cloud", GREEN)):
            w = g.textlength(text, font=font(20, True)) + 32
            g.rounded_rectangle((x - w, 28, x, 70), 21, fill=PANEL)
            g.text((x - w + 16, 38), text, font=font(20, True), fill=colour)
            x -= w + 12
        # scene view
        sx, sy, sw, sh = SCENE
        img.paste(self.render_scene(), (sx, sy))
        g.rounded_rectangle((sx + 14, sy + 14, sx + 14 + 470, sy + 50), 8, fill=(0, 0, 0))
        g.text((sx + 26, sy + 22), "SCENE VIEW  ·  for you; never shown to the robot", font=font(18, True), fill=INK)
        if self.inset is not None and self.sink.frames < self.inset_until:
            iw, ih = 384, 216
            ix, iy = sx + sw - iw - 16, sy + 16
            g.rectangle((ix - 4, iy - 4, ix + iw + 4, iy + ih + 34), fill=ORANGE)
            img.paste(self.inset.resize((iw, ih)), (ix, iy))
            g.text((ix + 8, iy + ih + 5), "the bin BEFORE it was moved", font=font(19, True), fill=(255, 255, 255))
        if self.banner and self.sink.frames < self.banner_until:
            text, colour = self.banner
            g.rounded_rectangle((sx + 14, sy + 64, sx + 14 + 700, sy + 140), 10, fill=colour)
            g.multiline_text((sx + 30, sy + 74), text, font=font(24, True), fill=(255, 255, 255), spacing=6)
        if self.verdict:
            text, colour = self.verdict
            g.rounded_rectangle((sx + 14, sy + sh - 112, sx + sw - 14, sy + sh - 14), 10, fill=colour)
            g.multiline_text((sx + 32, sy + sh - 100), text, font=font(28, True), fill=(255, 255, 255), spacing=10)
        if self.card:
            overlay = Image.new("RGBA", (sw, sh), (10, 12, 18, 215))
            img.paste(overlay, (sx, sy), overlay)
            g = ImageDraw.Draw(img)
            y = sy + 120
            for text, size, bold, colour in self.card:
                for line in text.split("\n"):
                    w = g.textlength(line, font=font(size, bold))
                    g.text((sx + (sw - w) / 2, y), line, font=font(size, bold), fill=colour)
                    y += int(size * 1.45)
                y += 14
        # head camera: what the student sees
        hx, hy, hw, hh = HEAD
        g.text((hx, 100), "WHAT THE STUDENT SEES", font=font(19, True), fill=BLUE)
        g.text((hx + 262, 101), "head camera, its only sensor", font=font(19), fill=MUTED)
        if self.grey:
            head = GREY_IMG
            g.text((hx, hy + hh + 2), "", font=font(14))
        else:
            self.head.update_scene(d, camera="head")
            head = Image.fromarray(self.head.render())
        img.paste(head.resize((hw, hh), Image.NEAREST), (hx, hy))
        if self.grey:
            g.rounded_rectangle((hx + 14, hy + 14, hx + 380, hy + 52), 8, fill=(0, 0, 0))
            g.text((hx + 26, hy + 22), "GREY IMAGE instead of the camera", font=font(19, True), fill=ORANGE)
        # the student's decision
        g.rounded_rectangle((hx, 586, hx + hw, 788), 12, fill=PANEL)
        g.text((hx + 20, 600), "STUDENT'S DECISION", font=font(18, True), fill=MUTED)
        g.text((hx + 222, 601), "SmolVLM2-500M, on this laptop", font=font(18), fill=MUTED)
        if self.thinking:
            g.text((hx + 20, 640), "reading the camera frame…", font=font(34, True), fill=MUTED)
        elif self.doing:
            g.text((hx + 20, 636), self.doing, font=font(40, True), fill=INK)
            g.text((hx + 20, 700), self.json, font=font(17, mono=True), fill=(170, 200, 255))
            if self.decide_s is not None:
                g.text((hx + 20, 744), f"decided in {self.decide_s:.2f} s", font=font(19), fill=MUTED)
        # instruction
        g.text((40, 808), "INSTRUCTION", font=font(18, True), fill=MUTED)
        g.text((180, 806), f"“{self.spec.instruction}”", font=font(22), fill=INK)
        # actions
        x, y = 40, 862
        g.text((x, y + 4), "ACTIONS", font=font(18, True), fill=MUTED)
        x += 104
        for text, colour in self.log:
            w = g.textlength(text, font=font(22, True)) + 32
            if x + w > W - 40:
                x, y = 144, y + 52
            g.rounded_rectangle((x, y - 4, x + w, y + 38), 19, fill=colour)
            g.text((x + 16, y + 3), text, font=font(22, True), fill=(255, 255, 255))
            x += w + 10
        # timeline (sim seconds of the episode)
        tx0, tx1, ty = 40, W - 40, 1010
        g.text((tx0, ty - 34), "TIMELINE", font=font(18, True), fill=MUTED)
        g.rounded_rectangle((tx0, ty, tx1, ty + 12), 6, fill=PANEL)
        now = 0.0 if self.t0 is None else d.time - self.t0
        span = max(10.0, now + 0.5)
        frac = min(1.0, now / span)
        g.rounded_rectangle((tx0, ty, tx0 + (tx1 - tx0) * frac, ty + 12), 6, fill=BLUE)
        for when, label, colour in self.marks:
            mx = tx0 + (tx1 - tx0) * min(1.0, (when - (self.t0 or 0.0)) / span)
            g.ellipse((mx - 9, ty - 3, mx + 9, ty + 15), fill=colour)
            g.text(
                (mx - g.textlength(label, font=font(16, True)) / 2, ty + 24), label, font=font(16, True), fill=colour
            )
        g.text((tx1 - 230, ty - 34), f"simulated time {now:4.1f} s", font=font(18), fill=MUTED)
        return img

    def snapshot(self) -> Image.Image:
        return self.render_scene()


class Filmed(Narrated):
    """Narrated, plus the film: what the student is doing, the bin move (with a 'before' snapshot), no frames while
    resetting."""

    def __init__(self, base: FixedBinMoveSim, film: Film) -> None:
        super().__init__(base, quiet=False)
        self.film = film
        self.start_bin: tuple[float, float] | None = None

    def reset(self, seed: int) -> str:
        self.film.active = False
        out = super().reset(seed)
        x, y = self._sim.ground_truth().bin_xy
        self.start_bin = (math.degrees(math.atan2(y, x)), math.hypot(x, y))
        self.film.active = True
        self.film.t0 = self._sim.data.time
        self.film.card = None
        self.film.hold(1.0)
        return out

    def execute(self, call):
        result = super().execute(call)
        self.film.log.append((skill_label(call) + ("" if result.ok else " (refused)"), BLUE if result.ok else RED))
        if call.skill == "place":
            self.film.marks.append((self._sim.data.time, "placed", GREEN))
        self.film.doing = ""
        return result

    def perturb(self, perturbation):
        before = self.film.snapshot()
        ok = super().perturb(perturbation)
        if ok:
            f = self.film
            f.log.append(("BIN MOVED", ORANGE))
            f.marks.append((self._sim.data.time, "bin moved", ORANGE))
            f.inset, f.inset_until = before, f.sink.frames + round(7 * FPS)
            f.banner = (
                "The simulator just moved the bin, unannounced.\nThe student only finds out through its camera.",
                ORANGE,
            )
            f.banner_until = f.sink.frames + round(6 * FPS)
            f.hold(2.0)
        return ok


class Watched:
    """The student, timed: frames held while it decides (scaled by the playback speed), then its output."""

    controller = "student"

    def __init__(self, student: StudentPolicy, film: Film, grey: bool) -> None:
        self.student, self.checkpoint, self.film, self.grey = student, student.checkpoint, film, grey

    def act(self, obs, instruction: str, feedback: str):
        if self.grey:
            obs = obs.model_copy(update={"head_rgb_jpeg": GREY})
        t0 = time.perf_counter()
        call = self.student.act(obs, instruction, feedback)
        dt = time.perf_counter() - t0
        f = self.film
        f.thinking = True
        f.hold(dt / f.speed)
        f.thinking = False
        f.doing = skill_label(call) if call else "done"
        f.json = json.dumps({"skill": call.skill, **call.params}) if call else '{"skill": null}'
        f.decide_s = dt
        f.hold(0.5)
        return call


def take(
    spec: TaskSpec,
    student: StudentPolicy,
    seed: int,
    sink: Sink,
    *,
    speed: float,
    grey: bool,
    intro: bool,
    outro: bool,
    header: str,
    move: bool = True,
) -> dict:
    stamp = datetime.now().strftime("%H%M%S")
    run_dir = HERE / "runs" / f"simvideo2-{'grey' if grey else 'cam'}-{seed}-{stamp}"
    refuse_seen_take(seed, run_dir)  # seed >= 900000, not held out, fresh run directory
    run_dir.mkdir(parents=True)
    base = FixedBinMoveSim(spec.task_config(), None, None)  # a move is the repo's own redraw from the spec's arc
    film = Film(base, sink, spec, seed, speed, grey, header)
    sim = Filmed(base, film)
    if intro:
        base.reset(seed)
        film.active = True
        film.card = [
            ("A robot teacher (Argon) taught this student a new skill", 38, True, INK),
            ("from one typed sentence, in 39.7 minutes. No human demonstrations.", 30, False, INK),
        ]
        film.hold(3.0)
        typed = f"“{spec.instruction}”"
        for i in range(0, len(typed) + 1, 3):
            film.card = film.card[:2] + [
                ("Now it works alone, on a scene it has never seen:", 26, False, MUTED),
                (textwrap.fill(typed[:i], 52), 30, True, (170, 200, 255)),
            ]
            film.emit()
        film.hold(2.0)
    _, row = episode(
        sim,
        Watched(student, film, grey),
        spec,
        seed,
        run_dir,
        EventLog(run_dir, run_dir.name),
        move=move,
        split="dev",
        demo=True,
    )
    m = sim.move or {}
    row["move_deg"] = round(m["bin_bearing_deg"] - sim.start_bin[0], 1) if m.get("applied") else None
    ok = row["accepted"]
    film.verdict = (
        ("Validator: the box is in the bin. Done by the student alone.", GREEN)
        if ok
        else (f"Validator: {row['reason']}.", RED)
    )
    film.hold(3.0)
    if outro:
        film.card = [
            ("Taught from one sentence in 39.7 minutes, then on its own:", 32, True, INK),
            ("18/20 unseen scenes  ·  17/20 with the bin moved after the lift", 30, False, INK),
            ("untrained: 0/20  ·  shown a grey image instead of its camera: 0/20", 26, False, MUTED),
            ("0.31 s per decision on a laptop GPU  ·  no cloud, no teacher", 26, False, MUTED),
            ("Scaffld: a robot teacher that finds the gaps and builds its own lessons", 28, True, (170, 200, 255)),
        ]
        film.hold(9.0)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=900407)
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed of the simulation (labelled on screen)")
    ap.add_argument("--live", action="store_true", help="also play it in a window while it runs")
    ap.add_argument("--pair", action="store_true", help="camera, then a grey image, on the same scene (no bin move)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plain", action="store_true", help="no opening or closing card (gallery clips)")
    args = ap.parse_args()
    spec = TaskSpec.model_validate_json(SPEC.read_text())
    student = StudentPolicy(CKPT, runs_dir=HERE / "runs")
    warm = FixedBinMoveSim(spec.task_config(), None, None)  # warm the model off camera, as when it stays loaded
    warm.reset(args.seed)
    student.act(warm.observe(), spec.instruction, "start")
    stamp = datetime.now().strftime("%H%M%S")
    out = args.out or HERE / "videos" / (
        f"pair-{args.seed}-{stamp}.mp4" if args.pair else f"sim-video2-{args.seed}.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = Sink(out, args.live)
    if args.pair:
        rows = [
            take(
                spec,
                student,
                args.seed,
                sink,
                speed=args.speed,
                grey=False,
                intro=False,
                outro=False,
                header="1 of 2: the student with its camera",
                move=False,
            ),
            take(
                spec,
                student,
                args.seed,
                sink,
                speed=args.speed,
                grey=True,
                intro=False,
                outro=False,
                header="2 of 2: the same scene, a grey image instead of the camera",
                move=False,
            ),
        ]
    else:
        rows = [
            take(
                spec,
                student,
                args.seed,
                sink,
                speed=args.speed,
                grey=False,
                intro=not args.plain,
                outro=not args.plain,
                header="the student alone, on a scene it has never seen",
            )
        ]
    sink.close(keep_window=3.0)
    for r in rows:
        print(
            f"seed {r['seed']}: {'SUCCEEDED' if r['accepted'] else 'failed: ' + r['reason']}; move {r['move_deg']}°; "
            f"[{r['skills']}]"
        )
    print(f"video {out}: {sink.frames} frames, {sink.frames / FPS:.1f} s")


if __name__ == "__main__":
    main()
