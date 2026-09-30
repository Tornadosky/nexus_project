"""Fig. 1: final return relative to a matched flat policy.

Each bar is the mean of the five listed seeds. Whiskers are the sample
standard deviation (ddof=1). Dots are the seeds. Neural, NeSy, and
Symbolic are shown for all six environments.

Emits figures/fig1_sweep.pdf, which paper.tex includes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paths import FIGURES

OUT = FIGURES

# Five seeds per arm, top-to-bottom on the figure.
ROWS = [
    ("Cartpole",
     [1.14, 1.08, 1.12, 1.09, 1.13],
     [1.18, 1.12, 1.21, 1.14, 1.17],
     [0.94, 1.01, 0.96, 0.92, 1.02]),
    ("Cheetah",
     [1.62, 1.48, 1.55, 1.39, 1.61],
     [1.19, 1.05, 1.22, 1.12, 1.13],
     [0.58, 0.76, 0.62, 0.81, 0.67]),
    ("Go1",
     [0.84, 0.68, 0.72, 0.79, 0.71],
     [0.36, 0.49, 0.41, 0.38, 0.47],
     [0.12, 0.31, 0.18, 0.08, 0.27]),
    ("Hopper",
     [1.38, 1.12, 1.41, 1.29, 1.11],
     [0.64, 0.88, 0.73, 0.81, 0.75],
     [0.28, 0.54, 0.35, 0.62, 0.31]),
    ("Panda",
     [0.98, 0.92, 1.01, 0.94, 0.95],
     [0.89, 0.96, 0.91, 0.88, 0.94],
     [0.61, 0.74, 0.65, 0.71, 0.64]),
    ("Walker",
     [1.16, 0.99, 1.12, 1.05, 1.08],
     [0.83, 0.94, 0.86, 0.91, 0.90],
     [0.45, 0.68, 0.59, 0.42, 0.62]),
]

ARMS = (
    ("Neural", 0.26, "#1f77b4"),
    ("NeSy", 0.00, "#2ca02c"),
    ("Symbolic", -0.26, "#ff7f0e"),
)

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def draw(ax):
    """Horizontal bars. y increases upward, so row 0 is the top environment."""
    n = len(ROWS)
    for i, (_name, neural, nesy, symbolic) in enumerate(ROWS):
        y = n - 1 - i
        for seeds, (label, offset, color) in zip((neural, nesy, symbolic), ARMS):
            mean, sd = _mean_sd(seeds)
            ax.barh(
                y + offset, mean, height=0.22, color=color, zorder=3,
                xerr=sd, error_kw={
                    "ecolor": "#333333", "elinewidth": 0.6,
                    "capsize": 1.6, "capthick": 0.6,
                },
                label=label,
            )
            jitter = np.linspace(-0.05, 0.05, len(seeds))
            ax.scatter(
                seeds, y + offset + jitter, s=8, color="#f4e4a8",
                edgecolor="#6b4c00", linewidth=0.35, zorder=4,
            )
    ax.axvline(1.0, color="#888888", lw=0.7, ls="--", zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([row[0] for row in ROWS][::-1])
    ax.set_xlim(0.0, 1.85)
    ax.set_xticks(np.arange(0.0, 1.61, 0.2))
    ax.set_xlabel("final return relative to the matched flat policy")
    ax.set_ylim(-0.55, n - 0.45)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    order = [name for name, _off, _col in ARMS]
    ax.legend(
        [seen[name] for name in order], order,
        ncol=3, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
    )


def main():
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    draw(ax)
    fig.savefig(OUT / "fig1_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig1_sweep.pdf")
    for name, neural, nesy, symbolic in ROWS:
        bits = []
        for label, seeds in (("Neural", neural), ("NeSy", nesy), ("Symbolic", symbolic)):
            mean, sd = _mean_sd(seeds)
            bits.append(f"{label} {mean:.2f} ± {sd:.2f}")
        print(name, " | ".join(bits))


if __name__ == "__main__":
    main()
