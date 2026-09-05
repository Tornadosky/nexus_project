"""Collapse rate per (env, arm) with a Wilson 95% interval.

The runbook's claims discipline is "failure rates and means-with-seeds, no min/max separation
claims" — every separation claim this campaign made died under more seeds. This reports the
failure rate in the form that survives: a proportion with an interval that shows how much the
denominator is actually carrying.

A cell "collapses" on a seed when its primary success is below --floor (default 0.01), the
threshold the §4 pre-registration is written against.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("flat", "neural", "symbolic", "nesy")
SUCCESS_KEY = "policy_diag/primary_success_rate"


def tail_mean(metrics: dict, key: str) -> float | None:
    """Transcribed from tools/analyze_v2.py:_tail_mean — the repo's canonical definition.

    It averages the last **10% of updates** (15 of 152), not the last 10 entries. Using a
    different window here made this tool report the Go1 rough flat cell as collapsing on 1 of 6
    seeds while analyze_v2's numbers imply 2 of 6, because two seeds sit within noise of the
    0.01 floor. Two tools disagreeing about a headline failure rate is worse than either
    convention being wrong.
    """
    if key not in metrics:
        return None
    a = np.asarray(metrics[key]).reshape(-1)
    if a.size == 0:
        return None
    return float(a[-max(1, a.size // 10):].mean())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tag_from(stem: str, variant: str) -> str:
    m = re.search(rf"{re.escape(variant)}_(.*?)_?s\d+$", stem)
    return (m.group(1) if m else "").strip("_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["runs/verify", "runs/viper"])
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args()

    cells: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for d in args.dirs:
        for pkl in sorted(Path(d).rglob("*.pkl")):
            if pkl.stem.startswith("diag"):
                continue
            try:
                ck = pickle.load(open(pkl, "rb"))
            except Exception:
                continue
            cfg, metrics = ck.get("config", {}) or {}, ck.get("metrics", {}) or {}
            variant = str(cfg.get("META_POLICY_TYPE", "")).lower()
            if variant not in VARIANTS or SUCCESS_KEY not in metrics:
                continue
            env = str(cfg.get("ENV_NAME", "?"))
            tag = tag_from(pkl.stem, variant)
            arm = f"{variant}·{tag}" if tag else variant
            v = tail_mean(metrics, SUCCESS_KEY)
            if v is None:
                continue
            cells[(env, arm)][int(cfg.get("SEED", -1))] = v

    print(f"Collapse rate = fraction of seeds with primary success < {args.floor}")
    print("Wilson 95% interval; cells with fewer than "
          f"{args.min_seeds} seeds omitted.\n")
    last_env = None
    for (env, arm), by_seed in sorted(cells.items()):
        n = len(by_seed)
        if n < args.min_seeds:
            continue
        if env != last_env:
            print(env)
            last_env = env
        vals = np.array([by_seed[s] for s in sorted(by_seed)])
        k = int((vals < args.floor).sum())
        lo, hi = wilson(k, n)
        print(f"  {arm:22s} n={n:3d}  collapse {k:3d}/{n:<3d} = {k / n:5.1%} "
              f"[{lo:.1%}, {hi:.1%}]   mean success {vals.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
