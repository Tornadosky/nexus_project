#!/usr/bin/env python3
"""V6 / paper Fig. 8 (Q4): degradation under action-noise perturbation.

The paper's claim (§4.3) is that symbolic / neuro-symbolic steering degrades LESS than the
baselines under environment perturbation.

**The retention ratio needs an absolute-performance guard, and this is not a cosmetic detail.**
Retention = success(0.3) / success(0.0). On Go1 the `symbolic` arm scores 0.031 unperturbed and
0.062 at the highest noise level — a retention of **202%**, which would print as the most robust
arm in the table and would appear to confirm the paper's specific claim. It is an artifact: the
policy does not work at any noise level, and adding action noise merely perturbs a degenerate
policy into slightly-less-degenerate behaviour. A ratio is uninformative when its denominator is
near zero.

So an arm is only assigned a retention figure when its unperturbed success clears `--min-base`.
Below that it is reported as DEGENERATE with its raw numbers, and excluded from any ranking.
(Same failure mode as the per-skill-return spread in `analyze_skill_returns.py`: a statistic that
looks spectacular precisely because its denominator is meaningless.)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3)
COLORS = {"flat": "#8792a2", "neural": "#343C96", "symbolic": "#F59F00", "nesy": "#1B6B45"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/robustness")
    ap.add_argument("--out", default="runs/robustness")
    ap.add_argument("--min-base", type=float, default=0.05,
                    help="unperturbed success below this => retention is not meaningful")
    args = ap.parse_args()

    data: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(args.dir) / "*.csv"))):
        stem = Path(f).stem
        env, var = stem.split("_")[0], stem.split("_")[1]
        for r in csv.DictReader(open(f)):
            data[(env, var)][float(r["perturbation"])].append(float(r["primary_success_rate"]))

    summary: dict[str, dict] = {}
    print(f"{'env/arm':<20} " + "  ".join(f"{l:>6}" for l in LEVELS) + "   retention@0.3")
    for key in sorted(data):
        row = data[key]
        if 0.0 not in row:
            continue
        base = float(np.mean(row[0.0]))
        cells = [f"{np.mean(row[l]):6.3f}" if l in row else "     -" for l in LEVELS]
        env, var = key
        if base < args.min_base:
            note = f"  DEGENERATE (base {base:.3f} < {args.min_base}) — no retention"
            ret = None
        elif 0.3 in row:
            ret = float(np.mean(row[0.3]) / base)
            note = f"  {ret * 100:6.1f}%"
        else:
            ret, note = None, "       -"
        print(f"{env + '/' + var:<20} " + "  ".join(cells) + note)
        summary[f"{env}/{var}"] = {
            "base": base,
            "levels": {str(l): float(np.mean(row[l])) for l in LEVELS if l in row},
            "retention_at_0.3": ret,
            "degenerate": base < args.min_base,
        }

    ranked = {k: v for k, v in summary.items() if v["retention_at_0.3"] is not None}
    if ranked:
        print("\nranking by retention (degenerate arms excluded):")
        for k, v in sorted(ranked.items(), key=lambda kv: -kv[1]["retention_at_0.3"]):
            print(f"  {k:<20} {v['retention_at_0.3'] * 100:6.1f}%   (base {v['base']:.3f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = sorted({k.split("/")[0] for k in summary})
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.2), squeeze=False, dpi=140)
    for i, env in enumerate(envs):
        ax = axes[0][i]
        for k, v in summary.items():
            e, var = k.split("/")
            if e != env:
                continue
            xs = [float(x) for x in v["levels"]]
            ys = [v["levels"][x] for x in v["levels"]]
            ax.plot(xs, ys, marker="o", lw=1.7, color=COLORS.get(var, "#666"),
                    ls=":" if v["degenerate"] else "-",
                    label=var + (" (degenerate)" if v["degenerate"] else ""))
        ax.set_xlabel("action-noise level")
        ax.set_ylabel("primary success")
        ax.set_title(env, fontsize=11, loc="left")
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Q4 robustness — degradation under action noise (paper Fig. 8 analogue). "
        "Dotted = degenerate arm, retention ratio not meaningful.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "robustness.png", bbox_inches="tight")
    print(f"\nwrote {out / 'robustness.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
