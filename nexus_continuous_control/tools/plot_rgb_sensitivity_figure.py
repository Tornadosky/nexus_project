"""How strongly does each trained pixel actor RESPOND to its camera?

The closed-loop ablation answers "does performance depend on pixels". This is the
open-loop counterpart and the cleanest single number for the failure mode: feed
each skill actor 128 genuinely different real frames and measure the spread of
its output action, as a percentage of the action range. A blind network is a
constant -> ~0%.

    blind cartpole   0.005-0.151%   (encoder never learned)
    fixed cartpole   7.8-26.0%      (aux pixel->state loss + committed meta)
    cheetah          30.3-38.1%     (learned from the policy gradient alone)

Log scale, because the interesting range spans four orders of magnitude.

    python tools/plot_rgb_sensitivity_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rgb/ablation")
    ap.add_argument("--out", default="results/rgb/ablation/pixel_responsiveness.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    runs = [
        ("cartpole", "CartpoleBalance (original) - BLIND", "#C44E52"),
        ("cartpole_aux", "CartpoleBalance + FIX", "#55A868"),
        ("cheetah", "CheetahRun", "#4C72B0"),
    ]
    series = []
    for tag, label, color in runs:
        p = root / tag / "pixel_sensitivity.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        d = json.loads(p.read_text())
        vals = [(k, 100.0 * v["relative_to_action_range"])
                for k, v in d["actor_sensitivity"].items()]
        series.append((label, color, vals))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    xticks, xlabels, pos, handles, labels = [], [], 0, [], []
    for label, color, vals in series:
        bar = None
        for name, v in vals:
            bar = ax.bar(pos, max(v, 3e-3), color=color, width=0.75)
            # 0.005% must not render as "0.00%": keep 3 decimals below 0.01.
            txt = f"{v:.3f}%" if v < 0.01 else (f"{v:.2f}%" if v < 1 else f"{v:.1f}%")
            ax.text(pos, max(v, 3e-3) * 1.18, txt, ha="center", va="bottom", fontsize=8)
            xticks.append(pos)
            xlabels.append(name.replace("_", " "))
            pos += 1
        handles.append(bar)
        labels.append(label)
        pos += 0.8

    ax.axhline(1.0, ls="--", lw=1.2, color="grey")
    ax.text(-0.55, 1.3, "1% of action range = responsiveness threshold",
            fontsize=8, ha="left", color="grey")
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 400)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8, rotation=22, ha="right")
    ax.set_ylabel("action spread across 128 different real frames"
                  "\n(% of action range, log scale)")
    ax.set_title("Does the trained encoder actually respond to what it sees?"
                 "\none bar per skill actor -- a blind network outputs a constant (~0%)",
                 fontsize=12)
    ax.legend(handles, labels, loc="upper left", fontsize=9, ncol=3, framealpha=0.95)
    fig.text(0.5, -0.08,
             "The original cartpole actor moves 0.005-0.151% of its action range across totally "
             "different images -- effectively a constant."
             "\nThe fix raises that ~170x, into the same regime as cheetah, which learned to see "
             "from the policy gradient alone.",
             ha="center", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
