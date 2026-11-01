"""F1 and F2: conceptual workflow diagrams drawn with matplotlib patches."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.3))
    if text:
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, text, fontsize=8.5,
                color=color, va="center")


def f1():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
    box(ax, 0.3, 3.6, 3.1, 1.3,
        "Original prompt\n\"$120 after a 20% discount.\nWhat was the original price?\"",
        "#dceefb", fs=9)
    box(ax, 0.3, 0.7, 3.1, 1.3,
        "Generated scratchpad\n\"Let the original price be P.\nA 20% discount means ...\"",
        "#e6f4ea", fs=9)
    box(ax, 4.2, 2.1, 2.5, 1.2, "Context window\n(prompt + scratchpad)",
        "#fff3cd", fs=9, bold=True)
    box(ax, 7.3, 2.1, 2.4, 1.2, "Next-token\nprediction", "#f8d7da",
        fs=10, bold=True)
    arrow(ax, 3.4, 4.25, 4.3, 3.0)
    arrow(ax, 3.4, 1.35, 4.3, 2.4)
    arrow(ax, 6.7, 2.7, 7.3, 2.7)
    # feedback loop: prediction appends a token to the scratchpad
    ax.add_patch(FancyArrowPatch((8.5, 2.1), (3.4, 0.9),
                                 connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#555555", lw=1.1, linestyle="--"))
    ax.text(6.1, 0.35, "predicted token is appended to the context",
            fontsize=8.5, color="#555555", ha="center")
    ax.text(5, 5.25, "Figure F1. The context window as working memory",
            ha="center", fontsize=10.5, fontweight="bold")
    fig.savefig(os.path.join(FIG, "F1_context_window.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def f2():
    steps = [
        ("A. Recognize\nstructure", "\"20% discount\" is a\npercentage reduction"),
        ("B. Identify\nreference", "discount is relative\nto the baseline P"),
        ("C. Establish\nremainder", "Remaining = P - 0.20P"),
        ("D. Simplify", "P(1 - 0.20) = 0.80P"),
        ("E. Connect to\nobserved value", "customer paid $120"),
        ("Equation", "120 = 0.80P"),
    ]
    fig, ax = plt.subplots(figsize=(9.9, 2.9))
    ax.set_xlim(0, 12.75); ax.set_ylim(0, 3.6); ax.axis("off")
    w, h, gap = 1.92, 1.7, 0.18
    x = 0.15
    for i, (title, body) in enumerate(steps):
        fc = "#c8e6c9" if i == len(steps) - 1 else "#e1f0fa"
        box(ax, x, 1.15, w, h, f"{title}\n{body}", fc, fs=7.3)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.01, 2.0, x + w + gap - 0.02, 2.0)
        x += w + gap
    ax.text(6.35, 3.35, "Figure F2. The translation pipeline: from words to an equation",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(6.35, 0.55,
            "A conceptual decomposition of the computation. The network does not execute these as discrete symbolic steps;\n"
            "its hidden states progressively realise the same logical chain (Section 5).",
            ha="center", fontsize=8.3, color="#444444")
    fig.savefig(os.path.join(FIG, "F2_translation_pipeline.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    f1()
    f2()
    print("diagrams done")
