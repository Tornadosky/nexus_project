"""One figure for the whole pixel-dependence story: cartpole vs cheetah.

Each environment's conditions are normalized to ITS OWN intact baseline, so two
different metrics (cartpole = upright fraction, cheetah = reward/step) sit on one
comparable axis: "how much performance survives when we corrupt the image".

  CheetahRun       -> everything collapses  => the actor really uses its camera
  CartpoleBalance  -> nothing changes       => the actor is blind and the
                                               privileged meta does the work

    python tools/plot_rgb_ablation_comparison.py \
        --runs results/rgb/ablation --out results/rgb/ablation/comparison.png
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/rgb/ablation")
    ap.add_argument("--out", default="results/rgb/ablation/comparison.png")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = Path(args.runs)
    series = []
    for tag, label, color in [
        ("cheetah", "CheetahRun — actor SEES", "#4C72B0"),
        ("cartpole", "CartpoleBalance — actor BLIND", "#C44E52"),
        ("cartpole_aux", "CartpoleBalance + FIX — actor SEES", "#55A868"),
    ]:
        path = runs / tag / "pixel_ablation.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        d = json.loads(path.read_text())
        # Cartpole reports a bounded upright fraction; locomotion reports mean
        # reward/step. Each is normalized to its own intact run, so the bars mean
        # "fraction of normal performance retained".
        key = "upright_fraction_mean" if "upright_fraction_mean" in d["results"]["intact"] \
            else "reward_per_step_mean"
        base = d["results"]["intact"][key]
        vals = [100.0 * d["results"][c][key] / base if c in d["results"] else np.nan
                for c in CONDITIONS]
        series.append((label, color, vals, key, base))

    if not series:
        raise SystemExit("no ablation results found")

    x = np.arange(len(CONDITIONS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, color, vals, key, base) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8.5)

    ax.axhline(100, ls="--", lw=1, color="grey", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("performance retained\n(% of intact run)")
    ax.set_ylim(0, 150)
    ax.set_title(
        "Does the in-loop pixel actor actually use its camera?\n"
        "Same code, same budget — only the task differs. Corrupting ONLY the actor's image input.",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    # Caption goes BELOW the axes: inside the plot it covered the bars it explains.
    fig.text(0.5, -0.09,
             "Cheetah collapses without pixels -> vision is doing the control.   "
             "Cartpole is unchanged, and is even BETTER with no actor at all (118%) ->\n"
             "the privileged meta-policy alone solves that task, so the encoder was never "
             "under pressure to learn.   The FIX (aux pixel->state loss + META_DECISION_INTERVAL 4 + LR 3e-4) makes it see AND doubles\n"
             "performance: 0.514 -> 1.000 upright.   128 envs / 250 updates, 5 episodes per condition, single seed.",
             ha="center", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
