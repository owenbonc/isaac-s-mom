"""Stitch two takes side by side (hstack), each half labelled, the shorter take padded by holding its last frame.
Adapted from the other session's clips/stitch.py (labels are arguments; label PNGs go next to this script).

Usage: python stitch.py left.mp4 "left label" right.mp4 "right label" out.mp4
(This ffmpeg has no drawtext/freetype, so the labels are PNGs drawn with PIL and overlaid.)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
LABELS = Path(__file__).resolve().parent / "labels"
MARGIN = 16


def label(text: str) -> Path:
    LABELS.mkdir(exist_ok=True)
    path = LABELS / f"{re.sub(r'[^A-Za-z0-9]+', '-', text)}.png"
    font = ImageFont.load_default(size=30)
    left, top, right, bottom = font.getbbox(text)
    pad_x, pad_y = 14, 8
    image = Image.new("RGBA", (right - left + 2 * pad_x, bottom - top + 2 * pad_y), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=8, fill=(22, 24, 28, 210))
    draw.text((pad_x - left, pad_y - top), text, fill=(240, 240, 240, 255), font=font)
    image.save(path)
    return path


def probe(video: Path) -> tuple[int, str]:
    out = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["nb_read_frames"]), stream["r_frame_rate"]


def main() -> None:
    left, left_label, right, right_label, out = sys.argv[1:6]
    left, right, out = Path(left), Path(right), Path(out)
    (n0, fps0), (n1, fps1) = probe(left), probe(right)
    if fps0 != fps1:
        raise SystemExit(f"frame rates differ: {fps0} vs {fps1}")
    total = max(n0, n1)
    graph = (
        f"[0:v]tpad=stop_mode=clone:stop={total - n0}[a0];"
        f"[1:v]tpad=stop_mode=clone:stop={total - n1}[b0];"
        f"[a0][2:v]overlay={MARGIN}:{MARGIN}[a];"
        f"[b0][3:v]overlay={MARGIN}:{MARGIN}[b];"
        "[a][b]hstack=inputs=2[v]"
    )
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(left),
            "-i",
            str(right),
            "-i",
            str(label(left_label)),
            "-i",
            str(label(right_label)),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-r",
            fps0,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    n, _ = probe(out)
    print(
        f"{out}: left {n0} frames, right {n1} frames -> {n} frames at {fps0} fps"
        f" (padded {'left' if n0 < n1 else 'right' if n1 < n0 else 'neither'} by {abs(n1 - n0)})"
    )
    if n != total:
        raise SystemExit(f"expected {total} frames, got {n}")


if __name__ == "__main__":
    main()
