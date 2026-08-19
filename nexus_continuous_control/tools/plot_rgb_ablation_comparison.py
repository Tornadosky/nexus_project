"""One figure for the whole pixel-dependence story, for any set of ablated runs.

Each run's conditions are normalized to ITS OWN intact baseline, so different
metrics (cartpole = upright fraction, locomotion = reward/step) sit on one
comparable axis: "how much performance survives when we corrupt the image".

Labels and colors are derived automatically from each run's rescored verdict
(tools/rescore_rgb_ablation.py must have been run first): green = SEES,
red = BLIND, grey = INCONCLUSIVE (near-zero baseline, e.g. hopper).

    python tools/plot_rgb_ablation_comparison.py \
        --tags cheetah/neural_seed0,cartpole/neural_blind,cartpole/neural_fixed \
        --out results/rgb/ablation/summary/comparison_neural.png
    python tools/plot_rgb_ablation_comparison.py \
        --tags cheetah/nesy_seed0,walker/nesy_blind,cartpole/nesy_blind,cartpole/nesy_fixed_seed0 \
        --out results/rgb/ablation/summary/comparison_nesy.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ["intact", "frozen_first", "random_replay", "shuffle_frames", "zeros", "const_action"]
NICE = {
    "intact": "intact\n(real pixels)",
    "frozen_first": "frozen\nfirst frame",
    "random_replay": "wrong-timestep\nframes",
    "shuffle_frames": "shuffled\nframe order",
    "zeros": "blank\nimage",
    "const_action": "no actor\n(constant action)",
}
VERDICT_COLOR = {True: "#55A868", False: "#C44E52", None: "#8C8C8C"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/rgb/ablation")
    ap.add_argument("--tags", default="cheetah/neural_seed0,cartpole/neural_blind,cartpole/neural_fixed")
    ap.add_argument("--out", default="results/rgb/ablation/summary/comparison_neural.png")
    ap.add_argument("--title", default="Does the in-loop pixel actor actually use its camera?")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = Path(args.runs)
    series, notes = [], []
    for tag in [t for t in args.tags.split(",") if t]:
        path = runs / tag / "pixel_ablation.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        d = json.loads(path.read_text())
        key = "upright_fraction_mean" if "upright_fraction_mean" in d["results"]["intact"] \
            else "reward_per_step_mean"
        base = d["results"]["intact"][key]
        if d.get("inconclusive"):
            notes.append(f"{d['env']} ({d['meta']}) omitted: intact score is "
                         f"indistinguishable from zero -- not measurable, not a verdict.")
            continue
        verdict = "SEES" if d.get("actor_uses_pixels") else "BLIND"
        # Same env+meta can appear twice (original vs the aux/fix run) -- the tag
        # is the only thing that disambiguates them, so fold it into the label.
        variant = " + FIX" if "fixed" in tag else ""
        label = f"{d['env']}{variant} ({d['meta']}) -- actor {verdict}"
        color = VERDICT_COLOR[d.get("actor_uses_pixels")]
        vals = [100.0 * d["results"][c][key] / base if c in d["results"] else np.nan
                for c in CONDITIONS]
        series.append((label, color, vals))

    if not series:
        raise SystemExit("no ablation results found")

    x = np.arange(len(CONDITIONS))
    width = min(0.75 / len(series), 0.30)
    fig, ax = plt.subplots(figsize=(max(11, 2.2 * len(series) + 6), 5))
    for i, (label, color, vals) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7.5)

    ax.axhline(100, ls="--", lw=1, color="grey", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("performance retained\n(% of intact run)")
    ax.set_ylim(0, 150)
    ax.set_title(f"{args.title}\n"
                "Same ablation protocol, corrupting ONLY the actor's image input.",
                fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95, ncol=1)
    if notes:
        fig.text(0.5, -0.06, "  ".join(notes), ha="center", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
