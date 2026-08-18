"""Headline RGB figures: how well does each method control from pixels?

Complements the ablation charts (which answer "does the actor USE its camera")
with the question a reader asks first: "how well does it actually perform?"

Panel 1 CartpoleBalance -- upright fraction, the interpretable metric:
    privileged state teacher / distilled pixels / in-loop pixels / in-loop + fix
Panel 2 CheetahRun -- mean per-step task reward (locomotion has no upright rate):
    privileged state teacher / distilled pixels / in-loop pixels

Sources: results/rgb/combined.json (distillation, upright, 3 seeds),
results/rgb/multienv/*.json (distillation, reward/step, 3 seeds),
results/rgb/ablation/*/pixel_ablation.json (in-loop, this campaign).

    python tools/plot_rgb_summary_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rgb")
    ap.add_argument("--out", default="results/rgb/ablation/method_comparison.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    jl = lambda p: json.loads(Path(p).read_text())

    # --- cartpole: everything on the upright fraction ---
    distill = {d["meta_policy"]: d for d in jl(root / "combined.json")}
    cp_state = distill["neural"]["state_hierarchy_success_mean"]      # 1.000
    cp_distill = distill["neural"]["pixel_hierarchy_success_mean"]    # 0.520
    cp_inloop = jl(root / "ablation/cartpole/pixel_ablation.json")["results"]["intact"][
        "upright_fraction_mean"]
    aux_path = root / "ablation/cartpole_aux/pixel_ablation.json"
    cp_fixed = (jl(aux_path)["results"]["intact"]["upright_fraction_mean"]
                if aux_path.exists() else None)

    # --- cheetah: everything on mean per-step task reward ---
    ch = jl(root / "multienv/rgb_cheetah_neural.json")
    ch_state = ch["state_hierarchy_success_mean"]                     # 0.618
    ch_distill = ch["pixel_hierarchy_success_mean"]                   # 0.155
    ch_inloop = jl(root / "ablation/cheetah/pixel_ablation.json")["results"]["intact"][
        "reward_per_step_mean"]

    STATE, PIX, GOOD, BAD = "#8C8C8C", "#DD8452", "#55A868", "#C44E52"
    panels = [
        ("CartpoleBalance", "upright fraction (250 steps)",
         [("privileged\nstate (cheats)", cp_state, STATE),
          ("distilled\npixels", cp_distill, PIX),
          ("in-loop\npixels", cp_inloop, BAD)]
         + ([("in-loop pixels\n+ FIX", cp_fixed, GOOD)] if cp_fixed is not None else []),
         "in-loop actor was BLIND here (red) until the aux pixel->state loss;\n"
         "the privileged meta alone could solve the task"),
        ("CheetahRun", "mean task reward / step",
         [("privileged\nstate (cheats)", ch_state, STATE),
          ("distilled\npixels", ch_distill, PIX),
          ("in-loop\npixels", ch_inloop, GOOD)],
         "in-loop actor genuinely SEES here (verified by ablation):\n"
         "81% of privileged performance, 3.2x the distilled policy"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (env, ylab, bars, note) in zip(axes, panels):
        names = [b[0] for b in bars]
        vals = [b[1] for b in bars]
        cols = [b[2] for b in bars]
        rects = ax.bar(range(len(bars)), vals, color=cols, width=0.62)
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9.5)
        ax.axhline(vals[0], ls="--", lw=1, color="grey", zorder=0)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_title(env, fontsize=12)
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))

    fig.suptitle("NEXUS skills from RGB: how well does each method control the robot?\n"
                 "dashed line = privileged upper bound (reads exact simulator state)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
