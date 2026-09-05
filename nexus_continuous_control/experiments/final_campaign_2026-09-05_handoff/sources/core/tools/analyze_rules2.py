"""Did the revised mask change BEHAVIOUR, given that it did not change score?

`rules2` cuts `recover` admissibility from 44.4% of a working policy's visited states to 11.8%,
yet scores identically to the shipped rule (Welch p=0.73, docs/ROUGH_TERRAIN_FINDINGS.md §J).
Two readings are possible and they say different things:

  * usage unchanged  -> the mask was never binding; `recover` was rarely the argmax anyway, and
                        the threshold measurement, while real, describes a rule that does not
                        drive the policy.
  * usage shifted    -> the mask IS binding, and reallocating decisions away from `recover`
                        simply does not improve performance — which points at skill quality
                        rather than skill selection, matching §E.1 and §H.

Reads `runs/oos/r2_<cond>_<arm>_s<seed>.csv`, written by tools/_queue_rules2_eval.sh.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("nesy_v2", "nesy_rules2")
CONDS = (("indist", "in-distribution"), ("cmd20", "commands x2"))


def load(root: Path, cond: str, arm: str) -> tuple[list[float], dict[str, list[float]]]:
    succ: list[float] = []
    usage: dict[str, list[float]] = defaultdict(list)
    for f in sorted(glob.glob(str(root / f"runs/oos/r2_{cond}_{arm}_s*.csv"))):
        if not re.search(rf"r2_{cond}_{arm}_s\d+\.csv$", f):
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        row = rows[0]
        succ.append(float(row["primary_success_rate"]))
        for k, v in row.items():
            if k.startswith("skill_usage/"):
                usage[k.split("/", 1)[1]].append(float(v))
    return succ, usage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)

    for cond, label in CONDS:
        loaded = {a: load(root, cond, a) for a in ARMS}
        if not any(s for s, _ in loaded.values()):
            continue
        print(f"\n=== {label} [{cond}] — greedy eval, 64 episodes/seed ===")
        skills = sorted(next(u for _, u in loaded.values() if u))
        head = " ".join(f"{s.split('_', 1)[1][:12]:>13s}" for s in skills)
        print(f"{'arm':12s} {'n':>3s} {'success':>8s}  {head}")
        for arm in ARMS:
            succ, usage = loaded[arm]
            if not succ:
                continue
            means = " ".join(f"{np.mean(usage[s]):13.3f}" for s in skills)
            print(f"{arm:12s} {len(succ):3d} {np.mean(succ):8.3f}  {means}")
        a, b = loaded[ARMS[0]][1], loaded[ARMS[1]][1]
        if a and b:
            print(f"{'delta':12s} {'':3s} {'':8s}  " + " ".join(
                f"{np.mean(b[s]) - np.mean(a[s]):+13.3f}" for s in skills))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
