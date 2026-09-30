"""Shared look for the paper figures.

Every script that writes a PDF included by final_paper.tex imports this
module so fonts, colors, and seed marks stay the same.
"""
import matplotlib.pyplot as plt


# Method colors used everywhere. RGB and language arms reuse the same hues.
COL = {
    "flat": "#8c8c8c",
    "hpqn": "#b07aa1",
    "nesy": "#2ca02c",
    "neural": "#1f77b4",
    "symbolic": "#ff7f0e",
    "ppo": "#d62728",
    "state": "#1f77b4",
    "pixels": "#2ca02c",
    "constant": "#8c8c8c",
    "hand": "#1f77b4",
    "initial": "#ff7f0e",
    "refined": "#2ca02c",
    "resample": "#b07aa1",
}
SEED = "#DAA520"
SEED_EDGE = "#6b4c00"


def apply():
    """Set the matplotlib defaults used by every paper figure."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def spread(ax, y, vals, color, horiz=True):
    """Min--max bar with one gold dot per seed.

    vals: sequence of scores for one arm.
    horiz: bar along x when True, along y when False.
    """
    lo, hi = min(vals), max(vals)
    if horiz:
        ax.plot([lo, hi], [y, y], color=color, lw=4.5, alpha=.55,
                solid_capstyle="butt", zorder=2)
        for x in (lo, hi):
            ax.plot([x, x], [y - .16, y + .16], color=color, lw=1.4, zorder=3)
        ax.scatter(vals, [y] * len(vals), s=13, color=SEED, edgecolor=SEED_EDGE,
                   linewidth=.4, zorder=4)
    else:
        ax.plot([y, y], [lo, hi], color=color, lw=4.5, alpha=.55,
                solid_capstyle="butt", zorder=2)
        ax.scatter([y] * len(vals), vals, s=13, color=SEED, edgecolor=SEED_EDGE,
                   linewidth=.4, zorder=4)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color="#dddddd", lw=.6)
    ax.set_axisbelow(True)


def shade_blocks(ax, sizes, vertical=False):
    """Gray band on even groups and a dashed line between groups.

    ``sizes`` is how many category slots each group occupies. The first
    group is the top row, or the left column when ``vertical`` is set.
    This is the same grouping mark as the language bar chart.
    """
    # Even groups sit on a light gray field. The dashed rule is the boundary.
    n = sum(sizes)
    start = 0
    for block, size in enumerate(sizes):
        if vertical:
            lo = start - 0.5
            hi = start + size - 0.5
        else:
            top = n - 1 - start
            hi = top + 0.5
            lo = top - size + 0.5
        if block % 2 == 0:
            if vertical:
                ax.axvspan(lo, hi, color="#f0f0f0", zorder=0, linewidth=0)
            else:
                ax.axhspan(lo, hi, color="#f0f0f0", zorder=0, linewidth=0)
        if block < len(sizes) - 1:
            if vertical:
                ax.axvline(hi, color="#c8c8c8", lw=.6, ls=(0, (2, 2)), zorder=1)
            else:
                ax.axhline(lo, color="#c8c8c8", lw=.6, ls=(0, (2, 2)), zorder=1)
        start += size
