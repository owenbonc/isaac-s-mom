"""The demo video (SPEC §6 criterion 8): head camera | third-person side by side, a caption strip underneath."""

import io
from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_HEIGHT = 480
CAPTION_HEIGHT = 64
MACROBLOCK = 16  # H.264 wants dimensions divisible by 16
CAPTION_COLOUR = (240, 240, 240)
CAPTION_BACKGROUND = (22, 24, 28)


def _panel(jpeg: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    return image.resize((round(image.width * PANEL_HEIGHT / image.height), PANEL_HEIGHT))


def compose(head: bytes, third: bytes, caption: str) -> np.ndarray:
    left, right = _panel(head), _panel(third)
    width = -(-(left.width + right.width) // MACROBLOCK) * MACROBLOCK
    canvas = Image.new("RGB", (width, PANEL_HEIGHT + CAPTION_HEIGHT), CAPTION_BACKGROUND)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    ImageDraw.Draw(canvas).text(
        (16, PANEL_HEIGHT + CAPTION_HEIGHT // 2),
        caption,
        fill=CAPTION_COLOUR,
        font=ImageFont.load_default(size=24),
        anchor="lm",
    )
    return np.asarray(canvas)


def record(frames: Iterable[tuple[bytes, bytes, str]], out: Path, fps: int) -> int:
    """Write (head JPEG, third-person JPEG, caption) frames to an MP4 and return the frame count."""
    count = 0
    with imageio.get_writer(str(out), fps=fps, codec="libx264", macro_block_size=MACROBLOCK) as writer:
        for head, third, caption in frames:
            writer.append_data(compose(head, third, caption))  # pyright: ignore[reportAttributeAccessIssue]  # imageio's stub types the writer as its base class
            count += 1
    if count == 0:
        out.unlink(missing_ok=True)
        raise ValueError("no frames to record")
    return count
