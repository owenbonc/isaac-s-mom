"""Standalone charts for the README and the pitch deck, drawn by make_figures.py's code from data.json.

Usage: python3 make_media.py <out dir>   (writes PNG at 500 dpi and PDF of each chart)
"""

import sys
from pathlib import Path

import make_figures as mf
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "build/media")
OUT.mkdir(parents=True, exist_ok=True)

TEACHER = [  # Argon demos the validator accepted, out of n attempted on development seeds
    ("head", "Fixed feet", None),
    ("row", "scene unchanged", (52, 60, "after")),
    ("row", "box moved at start", (50, 60, "moved")),
    ("row", "bin moved after lift", (28, 30, "moved")),
    ("head", "One sentence → new task", None),
    ("row", "showcase task", (56, 60, "after")),
    ("head", "Walking", None),
    ("row", "scene unchanged", (69, 100, "after")),
]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=500, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(OUT / f"{name}.png")


def sized(w: float, h: float, top: float = 7.0, bottom: float = 6.0):
    mf.W, mf.H, mf.TOP, mf.BOTTOM = w, h, top, bottom
    return plt.figure(figsize=(w * mf.MM, h * mf.MM))


def teacher(ax) -> None:
    ax.set_xlim(-4, 122)
    ax.set_ylim(-0.6, len(TEACHER) - 0.4)
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=mf.GRID, lw=0.5, zorder=0)
    for i, (kind, text, v) in enumerate(TEACHER):
        y = len(TEACHER) - 1 - i
        if kind == "head":
            mf.row_label(ax, y, text, head=True)
            continue
        k, n, how = v
        lo, hi = mf.wilson(k, n)
        mf.row_label(ax, y, text)
        ax.plot([lo * 100, hi * 100], [y, y], color=mf.ORANGE, lw=1.0, solid_capstyle="round", zorder=2)
        marker = "D" if how == "moved" else "o"
        ax.plot(k / n * 100, y, marker, ms=4.2 if marker == "D" else 4.8, mfc=mf.ORANGE, mec="white", mew=0.7, zorder=4)
        ax.text(hi * 100 + 3, y, f"{k}/{n}", ha="left", va="center", color=mf.INK, fontsize=6.2)


IMPACT = [  # (title, teacher value, student value, label format) from pitch/data/impact_comparison.csv
    ("Cloud calls per task (mean)", 9.78, 0.0, "{:.1f}", "{:.0f}"),
    ("Seconds per decision (median)", 8.29, 0.312, "{:.1f} s", "{:.2f} s"),
    ("Seconds per task (median)", 89.6, 3.17, "{:.0f} s", "{:.1f} s"),
]


def impact(fig) -> None:
    n = len(IMPACT)
    left, gap, lab = 1.5, 5.0, 17.0
    width = (mf.W - left - (n - 1) * gap - 1.5) / n
    for i, (name, t, s_, ft, fs) in enumerate(IMPACT):
        x0 = left + i * (width + gap)
        ax = fig.add_axes(
            [(x0 + lab) / mf.W, mf.BOTTOM / mf.H, (width - lab) / mf.W, (mf.H - mf.TOP - mf.BOTTOM) / mf.H]
        )
        fig.text(
            x0 / mf.W,
            1 - (mf.TOP - 1.2) / mf.H,
            name,
            ha="left",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=mf.INK,
        )
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, t * 1.45)
        ax.set_ylim(-0.6, 1.6)
        ax.barh(1, t, height=0.62, color=mf.ORANGE, zorder=2)
        ax.barh(0, max(s_, t * 0.004), height=0.62, color=mf.BLUE, zorder=2)
        ax.text(t + t * 0.03, 1, ft.format(t), va="center", ha="left", fontsize=6.2, color=mf.INK)
        ax.text(
            max(s_, t * 0.004) + t * 0.03,
            0,
            fs.format(s_),
            va="center",
            ha="left",
            fontsize=6.2,
            color=mf.INK,
            fontweight="bold",
        )
        tr = ax.get_yaxis_transform()
        ax.text(-0.04, 1, "Teacher (Argon)", transform=tr, ha="right", va="center", fontsize=6.0, color=mf.INK2)
        ax.text(-0.04, 0, "Student", transform=tr, ha="right", va="center", fontsize=6.0, color=mf.INK2)
        ax.axvline(0, color=mf.MUTED, lw=0.5)


def main() -> None:
    fig = sized(96, 50)
    mf.comparisons(mf.panel(fig, 1.5, 33, 50, "Student alone on 20 sealed, unseen scenes"))
    fig.text(
        1.5 / mf.W,
        1 - 4.6 / mf.H,
        "Hollow: the control. Filled: the result. Diamond: the scene changed mid-run.",
        fontsize=5.6,
        color=mf.MUTED,
        ha="left",
        va="top",
    )
    save(fig, "results-student")

    fig = sized(84, 50)
    mf.curve(fig, 1.5, 78, "Demos the new skill needs (same 56, first K by seed)")
    save(fig, "learning-curve")

    fig = sized(108, 40)
    mf.latency(mf.panel(fig, 1.5, 36, 66, "Decision latency vs. the 2 s edge budget"))
    save(fig, "latency")

    fig = sized(96, 48)
    teacher(mf.panel(fig, 1.5, 33, 50, "Teacher demos the validator accepted"))
    save(fig, "teacher-acceptance")

    fig = sized(150, 30, top=11.5, bottom=6.5)
    mf.title(fig, 1.5, "Doing the task: the cloud teacher vs. the student it trained")
    impact(fig)
    fig.text(
        1.5 / mf.W,
        1.2 / mf.H,
        "Teacher: 60 demos on development seeds (Mac). Student: 40 held-out episodes (laptop GPU). "
        "Success: teacher 56/60; student 18/20 and 17/20, alone.",
        fontsize=5.3,
        color=mf.MUTED,
        ha="left",
        va="bottom",
    )
    save(fig, "impact")


if __name__ == "__main__":
    main()
