"""Score the O1 zero-shot out-of-sample suite (runbook §1a).

Two shift families, both evaluated with NO retraining:
  (a) terrain swap   — flat-terrain checkpoints run on Go1JoystickRoughTerrain
                       (`runs/oos/go1_rough0_<arm>_s<seed>.csv`)
  (b/c) command shift — same env, `command_config.a` scaled x1.5 / x2 / zeroed
                       (`runs/oos/go1_<cond>_<arm>_s<seed>.csv`)

Retention = OOS success / that same seed's in-distribution success. It is only computed for
seeds whose in-distribution success clears DEGENERATE_FLOOR: go1 `flat` and `symbolic` collapse
on most seeds (mean 0.038 over 12 flat seeds), and 0.01/0.005 is a ratio of two numbers that
are both noise. Degenerate seeds are reported as absolute numbers only, per the V6 caveat.
"""

from __future__ import annotations

import argparse
import csv
import glob
import pickle
import re
from pathlib import Path

import numpy as np

# Env-agnostic alias, recorded by BOTH flat and hierarchical checkpoints, and the same
# quantity the OOS CSVs report as `primary_success_rate`. The go1-specific alias
# `policy_diag/go1/tracking_success` (no `_rate`) is written by the hierarchical arms only —
# keying on it silently dropped every `flat` seed from this table, i.e. dropped the baseline
# the whole comparison exists to make.
SUCCESS_KEY = "policy_diag/primary_success_rate"
DEGENERATE_FLOOR = 0.05
ARMS = ["flat", "neural", "nesy", "symbolic"]
# Both halves of the suite share the command-shift conditions; only the terrain swap differs in
# direction. `--prefix go1` is the flat-trained family (swap -> rough), `--prefix rt` is the
# rough-trained one (swap -> flat, i.e. reverse transfer). Conditions with no CSVs are skipped.
CONDITIONS = [
    ("indist", "in-distribution (baseline)"),
    ("rough0", "terrain swap -> RoughTerrain"),
    ("flat0", "terrain swap -> FlatTerrain (reverse transfer)"),
    ("cmd15", "commands x1.5"),
    ("cmd20", "commands x2"),
    ("cmd00", "zero commands"),
]


def in_distribution(root: Path, prefix: str = "go1") -> dict[tuple[str, int], float]:
    """In-distribution baseline, measured the SAME way as the OOS numbers.

    This must come from the deterministic eval at perturbation 0.0
    (`runs/robustness/go1_<arm>_s<seed>.csv`), NOT from the checkpoint's training tail-mean.
    The two disagree badly — flat-terrain `neural` s0 is 0.427 in training and 0.258 under
    greedy eval — because the training rollout is epsilon-greedy and the eval is not, and on
    go1 that epsilon is what keeps the meta-policy out of the `recover` attractor (see
    docs/ROUGH_TERRAIN_FINDINGS.md §A). Dividing an eval OOS number by a training baseline
    mixes the train/eval gap into what is supposed to be a terrain effect, and reported
    neural retention as 58% when the like-for-like figure is ~96%.

    Preferred source is the suite's own `indist` condition (`runs/oos/go1_indist_*`): same
    episode count, same eval seed, same code path as every shifted condition, so the only thing
    that differs is the shift. `runs/robustness/` at perturbation 0.0 is the fallback.
    """
    res = oos_results(root, "indist", prefix)
    if res:
        return res
    out: dict[tuple[str, int], float] = {}
    for p in glob.glob(str(root / "runs/robustness/go1_*_s*.csv")):
        m = re.search(r"go1_(\w+?)_s(\d+)\.csv$", p)
        if not m:
            continue
        for row in csv.DictReader(open(p)):
            if float(row.get("perturbation", "nan")) != 0.0:
                continue
            if row.get("mode") != "action_noise":
                continue
            out[(m.group(1), int(m.group(2)))] = float(row["primary_success_rate"])
            break
    return out


def oos_results(root: Path, cond: str, prefix: str = "go1") -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for f in glob.glob(str(root / f"runs/oos/{prefix}_{cond}_*_s*.csv")):
        m = re.search(rf"{prefix}_{cond}_(\w+?)_v2_s(\d+)\.csv$", f)
        if not m:
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        # robustness_eval writes the env's own metric name; `primary_success_rate` is the
        # env-agnostic alias for the same number and is what the in-distribution tail-mean
        # above is taken from.
        row = rows[0]
        col = "primary_success_rate" if "primary_success_rate" in row else "go1/tracking_success_rate"
        out[(m.group(1), int(m.group(2)))] = float(row[col])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--prefix", default="go1", help="CSV family: go1 (flat-trained) or rt (rough-trained)")
    args = ap.parse_args()
    root = Path(args.root)

    ind = in_distribution(root, args.prefix)
    if not ind:
        print("no in-distribution eval baselines in runs/robustness/ — cannot score retention")
        return 1
    print("O1 zero-shot OOS suite — Go1, tracking success, no retraining")
    print("in-dist baseline = deterministic eval at perturbation 0.0 (same path as the OOS runs)")
    print(f"(retention suppressed for seeds with in-distribution success < {DEGENERATE_FLOOR})")

    for cond, label in CONDITIONS:
        res = oos_results(root, cond, args.prefix)
        if not res:
            continue
        print(f"\n=== {label}  [{cond}] ===")
        print(f"{'arm':9s} {'seed':>4s} {'in-dist':>8s} {'oos':>8s} {'retention':>11s}")
        for arm in ARMS:
            rets = []
            for seed in sorted({s for a, s in res if a == arm}):
                i = ind.get((arm, seed))
                o = res.get((arm, seed))
                if i is None or o is None:
                    continue
                if i >= DEGENERATE_FLOOR:
                    rets.append(o / i)
                    tag = f"{100 * o / i:10.1f}%"
                else:
                    tag = " degenerate"
                print(f"{arm:9s} {seed:4d} {i:8.3f} {o:8.3f} {tag:>11s}")
            if rets:
                print(
                    f"{'':9s} {'':>4s} {'':>8s} {'':>8s} "
                    f"mean {100 * float(np.mean(rets)):.1f}% over n={len(rets)} "
                    f"non-degenerate seed(s)"
                )

    skill_table(root, args.prefix)
    return 0


def skill_usage(root: Path, cond: str, prefix: str = "go1") -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for f in glob.glob(str(root / f"runs/oos/{prefix}_{cond}_*_s*.csv")):
        m = re.search(rf"{prefix}_{cond}_(\w+?)_v2_s(\d+)\.csv$", f)
        if not m:
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        shares = {
            k.split("/", 1)[1]: float(v)
            for k, v in rows[0].items()
            if k.startswith("skill_usage/")
        }
        if shares:
            out[(m.group(1), int(m.group(2)))] = shares
    return out


def skill_table(root: Path, prefix: str = "go1") -> None:
    """What the meta-policy DID, averaged over all evaluation episodes.

    Success says how well the hierarchy performed; this says what it chose. Both are needed:
    a single rendered episode of go1 neural s0 showed 90.4% `recover` where the 64-episode
    share is 24.6%, so behavioural claims must come from here, not from a rollout video.
    """
    print("\n\n=== skill usage (share of in-episode meta decisions, 64 episodes) ===")
    for cond, label in CONDITIONS:
        use = skill_usage(root, cond, prefix)
        if not use:
            continue
        print(f"\n--- {label} [{cond}] ---")
        header_done = False
        for arm in ARMS:
            seeds = sorted({s for a, s in use if a == arm})
            if not seeds:
                continue
            # Skill sets are per-ARM, not global: `flat` has a single `flat_actor` skill while
            # the hierarchical arms have four. Deriving one column set from whichever arm was
            # read first raised KeyError on the others.
            skills = sorted(use[(arm, seeds[0])])
            if not header_done and len(skills) > 1:
                print(f"{'arm':9s} " + " ".join(f"{s.split('_', 1)[1][:12]:>13s}" for s in skills))
                header_done = True
            means = [float(np.mean([use[(arm, s)][sk] for s in seeds])) for sk in skills]
            if len(skills) == 1:
                print(f"{arm:9s} single skill ({skills[0]}) — no meta decision   n={len(seeds)}")
            else:
                print(f"{arm:9s} " + " ".join(f"{m:13.3f}" for m in means) + f"   n={len(seeds)}")


if __name__ == "__main__":
    raise SystemExit(main())
