"""Headline RGB figures: how well does each method control from pixels?

Complements the ablation charts (which answer "does the actor USE its camera")
with the question a reader asks first: "how well does it actually perform?"

Panel 1 CartpoleBalance -- upright fraction, the interpretable metric:
    privileged state teacher / distilled pixels / in-loop pixels / in-loop + fix
Panel 2 CheetahRun -- mean per-step task reward (locomotion has no upright rate):
    privileged state teacher / distilled pixels / in-loop pixels

--meta selects which in-loop/distillation meta-variant to plot (the ablation
tags for env X under meta Y are assumed to be named "<env>" for neural and
"<env>_nesy" for nesy, matching this campaign's naming).

Sources: results/rgb/combined.json (distillation, upright, 3 seeds),
results/rgb/multienv/*.json (distillation, reward/step, 3 seeds),
results/rgb/ablation/*/pixel_ablation.json (in-loop, this campaign).

    python tools/plot_rgb_summary_figures.py --meta neural
    python tools/plot_rgb_summary_figures.py --meta nesy \
        --out results/rgb/ablation/method_comparison_nesy.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rgb")
    ap.add_argument("--meta", default="neural", choices=["neural", "nesy"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out_default = ("results/rgb/ablation/method_comparison.png" if args.meta == "neural"
                   else "results/rgb/ablation/method_comparison_nesy.png")
    out = Path(args.out or out_default)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    jl = lambda p: json.loads(Path(p).read_text())
    suffix = "" if args.meta == "neural" else "_nesy"

    # --- cartpole: everything on the upright fraction ---
    distill = {d["meta_policy"]: d for d in jl(root / "combined.json")}
    cp_state = distill[args.meta]["state_hierarchy_success_mean"]
    cp_distill = distill[args.meta]["pixel_hierarchy_success_mean"]
    cp_path = root / f"ablation/cartpole{suffix}/pixel_ablation.json"
    cp_inloop = jl(cp_path)["results"]["intact"]["upright_fraction_mean"] if cp_path.exists() else None
    aux_path = root / f"ablation/cartpole_aux{suffix}/pixel_ablation.json"
    cp_fixed = jl(aux_path)["results"]["intact"]["upright_fraction_mean"] if aux_path.exists() else None

    # --- cheetah: everything on mean per-step task reward ---
    ch_key = "neural" if args.meta == "neural" else "nesy"
    ch_distill_path = root / f"multienv/rgb_cheetah_{ch_key}.json"
    ch = jl(ch_distill_path) if ch_distill_path.exists() else None
    ch_state = ch["state_hierarchy_success_mean"] if ch else None
    ch_distill = ch["pixel_hierarchy_success_mean"] if ch else None
    ch_path = root / f"ablation/cheetah{suffix}/pixel_ablation.json"
    ch_inloop = jl(ch_path)["results"]["intact"]["reward_per_step_mean"] if ch_path.exists() else None

    STATE, PIX, GOOD, BAD = "#8C8C8C", "#DD8452", "#55A868", "#C44E52"
    cp_bars = [("privileged\nstate (cheats)", cp_state, STATE),
              ("distilled\npixels", cp_distill, PIX)]
    if cp_inloop is not None:
        cp_bars.append(("in-loop\npixels", cp_inloop, BAD if cp_inloop < 0.9 else GOOD))
    if cp_fixed is not None:
        cp_bars.append(("in-loop pixels\n+ FIX", cp_fixed, GOOD))
    ch_bars = [("privileged\nstate (cheats)", ch_state, STATE),
              ("distilled\npixels", ch_distill, PIX)] if ch else []
    if ch_inloop is not None:
        ch_bars.append(("in-loop\npixels", ch_inloop, GOOD))

    panels = [
        ("CartpoleBalance", "upright fraction (250 steps)", cp_bars,
         f"meta = {args.meta}; in-loop pixels {'BLIND' if cp_inloop and cp_inloop < 0.9 else 'sees'} "
         "here -- see the ablation figures for proof"),
        ("CheetahRun", "mean task reward / step", ch_bars,
         f"meta = {args.meta}; in-loop pixels verified SEEING by ablation "
         f"({100 * ch_inloop / ch_state:.0f}% of privileged)" if ch_state else ""),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (env, ylab, bars, note) in zip(axes, panels):
        if not bars:
            ax.axis("off")
            ax.set_title(f"{env}\n(no data)", fontsize=12)
            continue
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

    fig.suptitle(f"NEXUS skills from RGB ({args.meta} meta): how well does each method "
                "control the robot?\ndashed line = privileged upper bound (reads exact simulator state)",
                fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
