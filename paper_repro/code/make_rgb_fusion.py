"""Architecture diagram: state concatenated with a camera embedding.

The state+RGB skill actor encodes a 64x64 three-frame grayscale stack and
concatenates the 128-d embedding with the full state. The critic and the
meta-controller stay on state.

Writes figures/fig_rgb_fusion.pdf for the paper and a matching PNG for the site.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paths import FIGURES

OUT = FIGURES

INK = "#2c2c2c"
MUTED = "#5c5c5c"
GREEN = "#2ca02c"
BLUE = "#1f77b4"
EDGE = "#3a3a3a"
FILL_G = "#f4faf4"
FILL_B = "#f4f8fc"
FILL_N = "#fafafa"


def _cartpole(tilt: float) -> np.ndarray:
    """One 64x64 grayscale frame, the resolution the encoder actually sees."""
    img = np.full((64, 64), 0.16, dtype=np.float32)
    img[46:48, :] = 0.78
    img[38:46, 24:42] = 0.70
    img[44:46, 22:26] = 0.55
    img[44:46, 40:44] = 0.55
    cx, cy = 33.0, 38.0
    length = 26.0
    for t in np.linspace(0.0, 1.0, 18):
        x = cx + np.sin(tilt) * length * t
        y = cy - np.cos(tilt) * length * t
        xi, yi = int(round(x)), int(round(y))
        img[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2] = 0.92
    return img


def _box(ax, x, y, w, h, face, edge=EDGE):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        facecolor=face, edgecolor=edge, linewidth=0.9,
    ))


def _arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color=color, shrinkA=1, shrinkB=1,
    ))


def _state_bars(ax, x, y):
    lengths = (0.92, 0.55, 0.78, 0.34, 0.66, 0.48)
    for i, length in enumerate(lengths):
        yy = y - i * 0.09
        ax.add_patch(plt.Rectangle((x, yy), 0.78 * length, 0.05, color=BLUE, lw=0))


def draw() -> plt.Figure:
    fig = plt.figure(figsize=(7.16, 3.30))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 7.16)
    ax.set_ylim(0, 3.30)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Newest frame in front. Figure-fraction insets, square in pixels.
    side = 0.62
    for img, (x, y) in zip(
        (_cartpole(-0.42), _cartpole(-0.08), _cartpole(0.30)),
        ((0.16, 2.12), (0.30, 2.26), (0.44, 2.40)),
    ):
        inset = fig.add_axes((x / 7.16, y / 3.30, side / 7.16, side / 3.30))
        inset.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("#1a1a1a")
            spine.set_linewidth(0.5)

    ax.text(0.62, 1.92, "64×64 camera", ha="center", va="center", fontsize=8, color=INK)
    ax.text(0.62, 1.72, "3-frame grayscale stack", ha="center", va="center", fontsize=6.5, color=MUTED)

    _box(ax, 1.85, 1.72, 2.20, 1.12, FILL_G, edge=GREEN)
    ax.text(2.95, 2.60, "convolutional encoder", ha="center", va="center", fontsize=8, color=INK)
    ax.text(2.95, 2.34, "3×3 conv, stride 2", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(2.95, 2.12, "32 → 64 → 64, ReLU", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(2.95, 1.90, "Dense 128, LayerNorm, tanh", ha="center", va="center", fontsize=7, color=MUTED)

    _arrow(ax, 1.12, 2.30, 1.82, 2.30, GREEN)
    _arrow(ax, 4.10, 2.22, 4.62, 1.28, GREEN)
    ax.text(4.52, 1.88, r"$z$", ha="right", va="center", fontsize=9, color=GREEN)

    _box(ax, 0.10, 0.48, 1.22, 1.05, FILL_B, edge=BLUE)
    ax.text(0.71, 1.32, "full state", ha="center", va="center", fontsize=8, color=INK)
    _state_bars(ax, 0.28, 1.08)

    _arrow(ax, 1.36, 0.97, 4.55, 0.97, BLUE)

    _box(ax, 4.58, 0.55, 0.78, 0.78, FILL_N)
    ax.text(4.97, 0.94, "concat", ha="center", va="center", fontsize=8, color=INK)

    _arrow(ax, 5.40, 0.94, 5.56, 0.94)

    _box(ax, 5.58, 0.48, 1.00, 0.92, FILL_N)
    ax.text(6.08, 1.12, "skill MLP", ha="center", va="center", fontsize=8, color=INK)
    ax.text(6.08, 0.88, "256, 256", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(6.08, 0.66, "tanh", ha="center", va="center", fontsize=7, color=MUTED)

    _arrow(ax, 6.62, 0.94, 6.76, 0.94)
    ax.text(6.82, 0.94, "action", ha="left", va="center", fontsize=8, color=INK)

    ax.plot([0.71, 0.71], [0.48, 0.28], color=MUTED, lw=0.7, linestyle=(0, (2, 1.4)))
    ax.annotate(
        "",
        xy=(1.55, 0.28), xytext=(0.71, 0.28),
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7),
    )
    ax.text(
        1.62, 0.28,
        "critic and meta-controller read the state only",
        ha="left", va="center", fontsize=7, color=MUTED,
    )
    return fig


def main() -> None:
    draw()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "fig_rgb_fusion.pdf"
    png = OUT / "fig_rgb_fusion.png"
    plt.savefig(pdf)
    plt.savefig(png, dpi=220)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
