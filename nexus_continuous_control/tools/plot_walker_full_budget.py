"""WalkerWalk at the paper's full budget: state-only vs state+RGB.

The matched-budget campaign (2.05M steps) found the camera worth +7.2% on walker.
This asks whether that survives 25.6x more training. It does not survive intact:
the baseline gains far more from the extra budget than the camera ever gave it,
and the advantage shrinks to about a point.

The two arms are NOT equally sampled -- the baseline has 3 seeds, the extension 1
-- so they are drawn differently on purpose. A single run is a line, not a band,
and the figure says so rather than letting a lone trace pass for a mean.

    python tools/plot_walker_full_budget.py \
        --out results/rgb/state_plus_rgb_full/figures/walker_full_budget.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = "#4C72B0"   # state only
EXT = "#DD8452"    # state + RGB
WINDOW = 20        # updates averaged for the reported scalar


def curve(root: Path, arm: str, seed: int) -> list[float]:
    p = root / "walker" / f"{arm}_seed{seed}" / "training_curves.json"
    return json.loads(p.read_text())["curves"]["episode_return"]


def smooth(y, k: int, np):
    """Centred rolling mean, k odd. Edges shrink the window rather than pad."""
    out = np.empty(len(y))
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = float(np.mean(y[lo:hi]))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rgb/state_plus_rgb_full")
    ap.add_argument("--out", default="results/rgb/state_plus_rgb_full/figures/walker_full_budget.png")
    ap.add_argument("--smooth", type=int, default=101,
                    help="rolling-mean window in updates; 6400 updates is very noisy raw")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)

    base = [np.asarray(curve(root, "state_matched_full", s)) for s in (0, 1, 2)]
    ext = [np.asarray(curve(root, "state_plus_rgb_full", 0))]

    n = min(len(c) for c in base + ext)
    base = np.stack([c[:n] for c in base])
    ext = ext[0][:n]
    x = np.arange(n)

    base_mean = base.mean(axis=0)
    base_sd = base.std(axis=0, ddof=1)

    bm = smooth(base_mean, args.smooth, np)
    bs = smooth(base_sd, args.smooth, np)
    em = smooth(ext, args.smooth, np)

    # the reported scalar: mean of the last WINDOW updates
    b_final = base[:, -WINDOW:].mean(axis=1)
    e_final = float(ext[-WINDOW:].mean())

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.fill_between(x, bm - bs, bm + bs, color=BASE, alpha=0.18, linewidth=0)
    for c in base:
        ax.plot(x, smooth(c, args.smooth, np), color=BASE, lw=0.7, alpha=0.45)
    ax.plot(x, bm, color=BASE, lw=2.4,
            label=f"state only (baseline)  n=3 seeds")
    ax.plot(x, em, color=EXT, lw=2.4,
            label=f"state + RGB (extension)  n=1 seed")

    ax.axvspan(n - WINDOW, n - 1, color="0.55", alpha=0.18, linewidth=0)

    for val, col in ((b_final.mean(), BASE), (e_final, EXT)):
        ax.plot(n - 1, val, "o", color=col, ms=7, mec="k", mew=0.8, zorder=5)
    ax.annotate(f"{b_final.mean():.1f}", (n - 1, b_final.mean()), xytext=(10, -12),
                textcoords="offset points", color=BASE, fontweight="bold", fontsize=11)
    ax.annotate(f"{e_final:.1f}", (n - 1, e_final), xytext=(10, 4),
                textcoords="offset points", color=EXT, fontweight="bold", fontsize=11)

    ax.set_xlabel(f"training update      ({n} updates = {n * 128 * 64 / 1e6:.1f}M env steps)")
    ax.set_ylabel("episode return during training\n(mean over 128 parallel envs)")
    ax.set_title(
        "WalkerWalk at the full budget: does the camera still help?\n"
        f"nesy meta, {n} updates x 128 envs x 64 steps = "
        f"{n * 128 * 64 / 1e6:.1f}M env steps per arm; arms differ in one config key (RGB_ACTOR)",
        fontsize=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_ylim(bottom=0)

    delta = 100 * (e_final / b_final.mean() - 1)
    ax.text(
        0.30, 0.46,
        f"last-{WINDOW}-update mean\n"
        f"  state only   {b_final.mean():7.2f} +/- {b_final.std(ddof=1):.2f}   "
        f"({', '.join(f'{v:.1f}' for v in b_final)})\n"
        f"  state + RGB  {e_final:7.2f}            (single seed)\n"
        f"  difference   {e_final - b_final.mean():+7.2f}  ({delta:+.1f}%)",
        transform=ax.transAxes, va="top", ha="left", family="monospace", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.7"))

    fig.text(
        0.5, -0.02,
        "The two arms are NOT equally sampled: the baseline is a 3-seed mean with a +/-1 s.d. band "
        "(thin lines are its individual seeds), the extension is ONE run drawn as a single line. A "
        f"{e_final - b_final.mean():+.1f} gap against a baseline whose three seeds span "
        f"{b_final.max() - b_final.min():.1f} is suggestive, not established.\n"
        "Curves are a centred rolling mean over "
        f"{args.smooth} updates; the raw per-update signal is far noisier. The grey column is the "
        f"last-{WINDOW}-update window the scalars average. This is the TRAINER's return, collected "
        "with exploration noise -- not a deterministic evaluation.",
        ha="center", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out.resolve()}")
    print(f"  baseline  {b_final.mean():.2f} +/- {b_final.std(ddof=1):.2f}  {b_final.round(2)}")
    print(f"  extension {e_final:.2f}  (1 seed)")
    print(f"  delta     {e_final - b_final.mean():+.2f}  ({delta:+.1f}%)")


if __name__ == "__main__":
    main()
