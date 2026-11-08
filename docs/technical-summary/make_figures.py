"""Draws Figure 2 (results) from data.json as a vector PDF (build/fig_results.pdf) and a PNG."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = json.loads((HERE / "data.json").read_text())
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill), validated on white: blue/orange/aqua pass all-pairs CVD checks.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SHADE, LINK = "#e1e0d9", "#f3f2ee", "#c9c8c0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # laptop GPU, cloud, CPU: the same as Figure 1
LOC = {"gpu": BLUE, "cloud": ORANGE, "cpu": AQUA}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.3,
        "pdf.fonttype": 42,
        "xtick.major.width": 0.5,
        "xtick.major.size": 2,
        "xtick.color": MUTED,
        "xtick.labelcolor": INK2,
        "xtick.labelsize": 5.9,
        "ytick.major.width": 0.5,
        "ytick.major.size": 2,
        "ytick.color": MUTED,
        "ytick.labelcolor": INK2,
        "ytick.labelsize": 5.9,
    }
)

MM = 1 / 25.4
W, H = 186, 31  # figure size in mm (full text width)
TOP, BOTTOM = 5.4, 5.4  # title band and x-axis band, mm


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def title(fig, left_mm: float, text: str) -> None:
    fig.text(left_mm / W, 1 - 0.8 / H, text, ha="left", va="top", fontsize=7.2, fontweight="bold", color=INK)


def panel(fig, left_mm: float, label_mm: float, plot_mm: float, text: str):
    """An axes whose row labels sit in label_mm to its left; the title spans labels and plot."""
    ax = fig.add_axes([(left_mm + label_mm) / W, BOTTOM / H, plot_mm / W, (H - TOP - BOTTOM) / H])
    title(fig, left_mm, text)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.label_frac = label_mm / plot_mm
    return ax


def row_label(ax, y: float, text: str, sub: str = "", head: bool = False) -> None:
    tr = ax.get_yaxis_transform()
    if head:
        ax.text(-ax.label_frac, y, text, transform=tr, ha="left", va="center", color=INK, fontweight="bold")
    elif sub:
        ax.text(-0.05, y + 0.05, text, transform=tr, ha="right", va="bottom", color=INK)
        ax.text(-0.05, y - 0.03, sub, transform=tr, ha="right", va="top", color=MUTED, fontsize=5.7)
    else:
        ax.text(-0.05, y, text, transform=tr, ha="right", va="center", color=INK)


def comparisons(ax) -> None:
    """Each row: the control (hollow) and the result (filled; a diamond when the scene changed mid-run)."""
    layout = []
    for g in DATA["comparisons"]["groups"]:
        layout.append(("head", g["title"], None))
        layout += [("row", r["label"], r) for r in g["rows"]]
    ax.set_xlim(-4, 122)
    ax.set_ylim(-0.6, len(layout) - 0.4)
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)
    for i, (kind, text, r) in enumerate(layout):
        y = len(layout) - 1 - i
        if kind == "head":
            row_label(ax, y, text, head=True)
            continue
        row_label(ax, y, text)
        c, res = r["control"], r["result"]
        xc, xr = c["k"] / c["n"] * 100, res["k"] / res["n"] * 100
        ax.plot([xc, xr], [y, y], color=LINK, lw=1.6, solid_capstyle="round", zorder=1)
        ax.plot(xc, y, "o", ms=4.0, mfc="white", mec=MUTED, mew=0.9, zorder=3)
        marker = "D" if r["kind"] == "moved" else "o"
        ax.plot(xr, y, marker, ms=4.2 if marker == "D" else 4.8, mfc=BLUE, mec="white", mew=0.7, zorder=4)
        ax.text(xr + 4.5, y, f"{res['k']}/{res['n']}", ha="left", va="center", color=INK, fontsize=6.2)
        if c["k"] > 0:
            ax.text(xc - 4.5, y, f"{c['k']}/{c['n']}", ha="right", va="center", color=MUTED, fontsize=5.9)


def curve(fig, left_mm: float, width_mm: float, text: str) -> None:
    pad = 7.5  # y tick labels
    ax = fig.add_axes([(left_mm + pad) / W, BOTTOM / H, (width_mm - pad) / W, (H - TOP - BOTTOM) / H])
    title(fig, left_mm, text)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.5)
    c = DATA["curve"]
    n = c["n"]
    xs = [p["demos"] for p in c["points"]]
    ax.set_xscale("log", base=2)
    ax.set_xlim(10.5, 75)
    ax.set_xticks(xs, [str(x) for x in xs[:-1]] + [f"{xs[-1]} demos"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(-6, 106)
    ax.set_yticks([0, 50, 100], ["0", "50", "100%"])
    for yv in (0, 25, 50, 75, 100):
        ax.axhline(yv, color=GRID, lw=0.5, zorder=0)
    for key, marker, shift, ms in (("normal", "o", 0.955, 4.8), ("moved", "D", 1.045, 4.2)):
        px = [x * shift for x in xs]
        py = [p[key] / n * 100 for p in c["points"]]
        ax.plot(px, py, color=BLUE, lw=0.9, alpha=0.55, zorder=2)
        for x, p in zip(px, c["points"], strict=True):
            lo, hi = wilson(p[key], n)
            ax.plot([x, x], [lo * 100, hi * 100], color=BLUE, lw=0.9, solid_capstyle="round", zorder=2)
            ax.plot(x, p[key] / n * 100, marker, ms=ms, mfc=BLUE, mec="white", mew=0.7, zorder=4)
    last = c["points"][-1]
    ax.text(
        xs[-1] * 0.955 / 1.09,
        last["normal"] / n * 100,
        f"{last['normal']}/{n}",
        ha="right",
        va="center",
        color=INK,
        fontsize=6.2,
    )
    ax.text(
        xs[-1] * 1.045 * 1.09,
        last["moved"] / n * 100,
        f"{last['moved']}/{n}",
        ha="left",
        va="center",
        color=INK,
        fontsize=6.2,
    )
    for p in c["points"][:-1]:
        ax.text(p["demos"], 21, f"{p['normal']}/{n}", ha="center", va="bottom", color=INK, fontsize=6.2)
    # direct labels instead of a legend box: one series per marker shape
    ax.plot(11.6, 88, "o", ms=4.4, mfc=BLUE, mec="white", mew=0.7)
    ax.text(12.6, 88, "normal task", ha="left", va="center", color=INK2, fontsize=5.8)
    ax.plot(11.6, 72, "D", ms=3.9, mfc=BLUE, mec="white", mew=0.7)
    ax.text(12.6, 72, "bin moved after lift", ha="left", va="center", color=INK2, fontsize=5.8)


def latency(ax) -> None:
    lat = DATA["latency"]
    rows, budget = lat["rows"], lat["budget_s"]
    ax.set_xscale("log")
    ax.set_xlim(0.2, 45)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks, ["0.2", "0.5", "1", "2", "5", "10", "20 s"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for x in ticks:
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)
    ax.axvspan(budget, 45, color=SHADE, zorder=0, lw=0)
    ax.axvline(budget, color=INK, lw=0.8, zorder=1)
    ax.text(budget * 1.08, len(rows) - 0.3, "over the 2 s budget", ha="left", va="top", color=INK2, fontsize=5.8)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        colour = LOC[r["loc"]]
        med, p95 = r["median_ms"] / 1000, r["p95_ms"] / 1000
        note = f" · {r['note']}" if r["note"] else ""
        row_label(ax, y, r["label"], f"{r['where']}{note} · n = {r['n']:,}")
        ax.plot([med, p95], [y, y], color=colour, lw=1.0, solid_capstyle="round", zorder=2)
        ax.plot([p95, p95], [y - 0.13, y + 0.13], color=colour, lw=1.0, zorder=2)
        ax.plot(med, y, "o", ms=4.8, mfc=colour, mec="white", mew=0.7, zorder=4)
        # round half up, so 1075 ms reads 1.08 s as in the text
        text = (
            f"{r['median_ms']} ms"
            if r["median_ms"] < 1000
            else f"{int(r['median_ms'] / 10 + 0.5) / 100:.2f} s"
            if med < 2
            else f"{med:.1f} s"
        )
        ax.text(med, y + 0.2, text, ha="center", va="bottom", color=INK, fontsize=6.2)


def main() -> None:
    fig = plt.figure(figsize=(W * MM, H * MM))
    comparisons(panel(fig, 0, 29, 36, "(a) Student alone on sealed, unseen scenes"))
    curve(fig, 72, 45, "(b) Demos the new skill needs")
    latency(panel(fig, 124, 27, 35, "(c) Decision latency vs. edge budget"))
    fig.savefig(OUT / "fig_results.pdf")
    fig.savefig(OUT / "fig_results.png", dpi=300)


if __name__ == "__main__":
    main()
