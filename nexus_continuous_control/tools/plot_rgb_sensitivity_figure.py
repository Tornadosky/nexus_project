"""How strongly does each trained pixel actor RESPOND to its camera?

The closed-loop ablation answers "does performance depend on pixels". This is the
open-loop counterpart and the cleanest single number for the failure mode: feed
each skill actor 128 genuinely different real frames and measure the spread of
its output action, as a percentage of the action range. A blind network is a
constant -> ~0%.

Log scale, because the interesting range spans four orders of magnitude.

    python tools/plot_rgb_sensitivity_figure.py \
        --tags cartpole/neural_blind,cartpole/neural_fixed,cheetah/neural_seed0 \
        --out results/rgb/ablation/summary/pixel_responsiveness_neural.png
    python tools/plot_rgb_sensitivity_figure.py \
        --tags cartpole/nesy_blind,cartpole/nesy_fixed_seed0,cheetah/nesy_seed0,walker/nesy_blind \
        --out results/rgb/ablation/summary/pixel_responsiveness_nesy.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PALETTE = ["#C44E52", "#55A868", "#4C72B0", "#8172B2", "#CCB974", "#64B5CD"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rgb/ablation")
    ap.add_argument("--tags", default="cartpole/neural_blind,cartpole/neural_fixed,cheetah/neural_seed0")
    ap.add_argument("--out", default="results/rgb/ablation/summary/pixel_responsiveness_neural.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    runs = []
    tags = [t for t in args.tags.split(",") if t]
    for i, tag in enumerate(tags):
        sp = root / tag / "pixel_sensitivity.json"
        ap_ = root / tag / "pixel_ablation.json"
        if not sp.exists():
            print(f"[skip] {sp} missing")
            continue
        env = json.loads(sp.read_text()).get("env", tag)
        meta = json.loads(sp.read_text()).get("meta", "")
        verdict = None
        if ap_.exists():
            verdict = json.loads(ap_.read_text()).get("actor_uses_pixels")
        tag_note = " (inconclusive)" if verdict is None else ""
        variant = " + FIX" if "fixed" in tag else ""
        runs.append((tag, f"{env}{variant} ({meta}){tag_note}" if meta else f"{tag}{tag_note}",
                    PALETTE[i % len(PALETTE)]))

    series = []
    for tag, label, color in runs:
        p = root / tag / "pixel_sensitivity.json"
        d = json.loads(p.read_text())
        vals = [(k, 100.0 * v["relative_to_action_range"])
                for k, v in d["actor_sensitivity"].items()]
        series.append((label, color, vals))

    fig, ax = plt.subplots(figsize=(max(10, 1.9 * sum(len(v) for _, _, v in series)), 5.4))
    xticks, xlabels, pos, handles, labels = [], [], 0, [], []
    for label, color, vals in series:
        bar = None
        for name, v in vals:
            bar = ax.bar(pos, max(v, 3e-3), color=color, width=0.75)
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
    ax.legend(handles, labels, loc="upper left", fontsize=8.5,
              ncol=min(3, len(labels)), framealpha=0.95)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
