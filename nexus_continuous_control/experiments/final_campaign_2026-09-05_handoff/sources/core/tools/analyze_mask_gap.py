"""Separate "the rule forbade the skill" from "the meta-policy did not want it".

The nesy arm selects `recover` on ~1% of go1 decisions while the otherwise-identical neural
arm selects it on ~16%. Usage alone cannot say why. The training loop records both halves:

  mask_available/<skill>                  fraction of steps the rule ADMITS the skill
  mask_selected_given_available/<skill>   fraction of ADMITTING steps where it was CHOSEN

so usage ~= available * selected_given_available. If `available` is near 1 and the gap lives
in `selected_given_available`, the mask is not the constraint and the difference is learned;
if `available` is small, the hand-written rule is what suppresses the skill — which is the
`rules2` hypothesis (runbook §5: the thresholds, not the architecture, are the weak part).
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


def tail(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    return float(np.mean(np.asarray(metrics[key]).reshape(-1)[-10:]))


def report(path: Path) -> None:
    metrics = pickle.load(open(path, "rb")).get("metrics", {})
    skills = sorted(
        {re.fullmatch(r"skill_usage/(\d+_.+)", k).group(1) for k in metrics if k.startswith("skill_usage/")}
    )
    print(f"\n{path.name}")
    print(f"  {'skill':20s} {'available':>10s} {'sel|avail':>10s} {'usage':>8s} {'violation':>10s}")
    for sk in skills:
        av = tail(metrics, f"mask_available/{sk}")
        sg = tail(metrics, f"mask_selected_given_available/{sk}")
        us = tail(metrics, f"skill_usage/{sk}")
        vi = tail(metrics, f"mask_violation/{sk}")
        fmt = lambda x: "     n/a" if x is None else f"{x:8.3f}"
        print(f"  {sk:20s} {fmt(av):>10s} {fmt(sg):>10s} {fmt(us):>8s} {fmt(vi):>10s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()
    for c in args.checkpoints:
        p = Path(c)
        if p.exists():
            report(p)
        else:
            print(f"MISSING {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
