#!/usr/bin/env python3
"""V1.3 — meta-mode ablation on CartpoleBalance: flat vs neural vs symbolic vs nesy.

Gate (docs/VERIFICATION_PLAN.md §V1.3): `nesy >= neural >= flat` on **deterministic** return,
3 seeds, on the corrected exploration schedule. A changed ordering is a finding, not a failure,
but it has to be explained.

Two metrics are read, and they are not interchangeable:

  * deterministic return + primary success, from `tools/robustness_eval.py --levels 0.0`
    (`runs/v13/<variant>_s<seed>.csv`) — this is what the gate is stated on;
  * the training return curve from each checkpoint — this is the shape evidence, still carrying
    exploration noise, and it is NOT the gate.

The ceiling matters as much as the ordering. CartpoleBalance's episode return saturates near
1000 at 1000 steps upright; if every variant reaches the ceiling then the ablation has no
resolving power left and says nothing about the meta-policy, which is a result about the
*experiment*, not about the hierarchy.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = ["flat", "neural", "symbolic", "nesy"]
COLORS = {
    "flat": "#8792a2",
    "neural": "#343C96",
    "symbolic": "#0F6C74",
    "nesy": "#1B6B45",
}


def read_eval(out_dir: Path, variant: str, seeds=(0, 1, 2)) -> dict[str, list[float]]:
    ret, succ = [], []
    for s in seeds:
        p = out_dir / f"{variant}_s{s}.csv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if abs(float(row["perturbation"])) > 1e-9:
                    continue
                ret.append(float(row["episode_return_mean"]))
                succ.append(float(row["primary_success_rate"]))
    return {"return": ret, "success": succ}


def read_curves(runs: Path, variant: str, tag: str, seeds=(0, 1, 2)) -> list[np.ndarray]:
    out = []
    for s in seeds:
        p = runs / f"cartpole_balance_{variant}_{tag}_s{s}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            ck = pickle.load(fh)
        out.append(np.asarray(ck["metrics"]["rollout/episode_return"]).reshape(-1))
    return out


def plot(curves: dict, evals: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=150)

    ax = axes[0]
    for v in VARIANTS:
        cs = curves.get(v) or []
        if not cs:
            continue
        n = min(len(c) for c in cs)
        arr = np.stack([c[:n] for c in cs])
        m, sd = arr.mean(0), arr.std(0)
        x = np.arange(n)
        ax.plot(x, m, color=COLORS[v], lw=1.6, label=f"{v} (n={len(cs)})")
        ax.fill_between(x, m - sd, m + sd, color=COLORS[v], alpha=0.16, lw=0)
    ax.set_xlabel("update")
    ax.set_ylabel("training episode return")
    ax.set_title("Training return — mean ± seed std (exploration noise ON)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.18, lw=0.6)

    for ax, key, label, ceil in (
        (axes[1], "return", "deterministic episode return", 1000.0),
        (axes[2], "success", "primary success rate", 1.0),
    ):
        labels, means, pts = [], [], []
        for v in VARIANTS:
            vals = evals.get(v, {}).get(key) or []
            if not vals:
                continue
            labels.append(v)
            means.append(float(np.mean(vals)))
            pts.append(vals)
        x = np.arange(len(labels))
        ax.bar(x, means, color=[COLORS[v] for v in labels], width=0.6)
        for i, vals in enumerate(pts):
            ax.scatter([i] * len(vals), vals, color="#151A23", s=20, zorder=3, alpha=0.85)
        if ceil:
            ax.axhline(ceil, ls="--", lw=1.1, color="#8A3227")
            ax.text(
                len(labels) - 0.4, ceil * 0.985, "ceiling", color="#8A3227", fontsize=8, ha="right",
                va="top",
            )
            ax.set_ylim(0, ceil * 1.12)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"V1.3 gate metric — {label} (dots = seeds)", fontsize=10, loc="left")
        ax.grid(alpha=0.18, lw=0.6, axis="y")

    fig.suptitle(
        "V1.3 meta-mode ablation — CartpoleBalance, 150 updates, noise 1.0→0.15, 3 seeds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/verify")
    ap.add_argument("--evals", default="runs/v13")
    ap.add_argument("--tag", default="explore")
    ap.add_argument("--out", default="runs/v13")
    args = ap.parse_args(argv)

    runs, ev_dir, out = Path(args.runs), Path(args.evals), Path(args.out)
    curves = {v: read_curves(runs, v, args.tag) for v in VARIANTS}
    evals = {v: read_eval(ev_dir, v) for v in VARIANTS}

    print("=" * 86)
    print("V1.3 META-MODE ABLATION — CartpoleBalance, noise 1.0->0.15")
    print("=" * 86)
    print(f"{'variant':<10} {'n':>2}  {'det. return':>12}  {'seeds':<28} {'success':>8}  seeds")
    summary: dict[str, Any] = {}
    for v in VARIANTS:
        r, s = evals[v]["return"], evals[v]["success"]
        if not r:
            print(f"{v:<10}  -  {'(no eval)':>12}")
            continue
        summary[v] = {
            "return_mean": float(np.mean(r)),
            "return_seeds": r,
            "success_mean": float(np.mean(s)),
            "success_seeds": s,
        }
        print(
            f"{v:<10} {len(r):2d}  {np.mean(r):12.1f}  "
            f"{str([round(x, 1) for x in r]):<28} {np.mean(s):8.3f}  {[round(x, 3) for x in s]}"
        )

    have = [v for v in VARIANTS if v in summary]
    verdict = "INCOMPLETE"
    notes: list[str] = []
    if {"flat", "neural", "nesy"} <= set(have):
        f, n, y = (summary[k]["return_mean"] for k in ("flat", "neural", "nesy"))
        ordered = y >= n >= f
        verdict = "PASS" if ordered else "ORDERING CHANGED"
        print(f"\n  gate: nesy >= neural >= flat  ->  {y:.1f} >= {n:.1f} >= {f:.1f}  :  {verdict}")

        # Seed spread vs between-variant gap. On a saturating task the gap can be smaller than
        # the noise floor, in which case the ordering is not a measurement of anything.
        spreads = {k: (max(summary[k]["return_seeds"]) - min(summary[k]["return_seeds"])) for k in have}
        gap = max(summary[k]["return_mean"] for k in have) - min(
            summary[k]["return_mean"] for k in have
        )
        worst = max(spreads.values())
        print(f"  between-variant spread {gap:.1f} vs worst within-variant seed spread {worst:.1f}")
        if gap < worst:
            notes.append(
                f"Between-variant spread ({gap:.1f}) is SMALLER than the worst within-variant "
                f"seed spread ({worst:.1f}). The ordering is inside the noise floor and must not "
                f"be quoted as a result."
            )
        ceil_hits = [k for k in have if summary[k]["success_mean"] >= 0.999]
        if len(ceil_hits) >= 2:
            notes.append(
                f"{len(ceil_hits)} of {len(have)} variants sit at the success CEILING "
                f"({', '.join(ceil_hits)} at 1.000). CartpoleBalance no longer discriminates "
                f"between meta-modes on this budget — the ablation needs a harder env, not "
                f"more seeds."
            )

    for note in notes:
        print(f"\n  NOTE: {note}")

    plot(curves, evals, out / "v13_ablation.png")
    (out / "v13_ablation.json").write_text(
        json.dumps({"cells": summary, "verdict": verdict, "notes": notes}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out / 'v13_ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
