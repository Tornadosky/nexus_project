"""Compare skill selection between matched cells that differ only in the environment.

Prediction 2 of the pre-registered rough-terrain block (runbook §4) is about *behaviour*, not
reward: `recover` usage should rise on rough terrain relative to flat. That is checked here on
the `skill_usage/*` metrics of paired checkpoints — same policy module, same variant, same
seed, same budget; only ENV_NAME differs.

Reported per skill: tail-mean usage share on each side and the delta, plus the paired
per-seed deltas so a single-seed swing cannot be mistaken for the effect.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


def usage(path: Path) -> dict[str, float]:
    """Tail-mean usage share per skill, keyed by the skill's display name."""
    metrics = pickle.load(open(path, "rb")).get("metrics", {})
    out: dict[str, float] = {}
    for k, v in metrics.items():
        m = re.fullmatch(r"skill_usage/(\d+)_(.+)", k)
        if not m:
            continue
        arr = np.asarray(v).reshape(-1)
        out[f"{m.group(1)}_{m.group(2)}"] = float(np.mean(arr[-10:]))
    return out


def scalar(path: Path, key: str) -> float | None:
    metrics = pickle.load(open(path, "rb")).get("metrics", {})
    if key not in metrics:
        return None
    return float(np.mean(np.asarray(metrics[key]).reshape(-1)[-10:]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--variant", default="neural", help="neural | nesy")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base", default="go1_joystick_{v}_v2_s{s}.pkl")
    ap.add_argument("--shifted", default="go1_rough_{v}_v2_s{s}.pkl")
    args = ap.parse_args()

    verify = Path(args.root) / "runs/verify"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    pairs = []
    for s in seeds:
        b = verify / args.base.format(v=args.variant, s=s)
        r = verify / args.shifted.format(v=args.variant, s=s)
        if b.exists() and r.exists():
            pairs.append((s, b, r))
        else:
            print(f"  skip s{s}: missing {'base' if not b.exists() else 'shifted'} checkpoint")
    if not pairs:
        print("no matched pairs yet")
        return 0

    skills = sorted(usage(pairs[0][1]))
    print(f"\nSkill-usage shift, variant={args.variant}, n={len(pairs)} paired seed(s)")
    print("flat terrain -> rough terrain (tail-mean share of meta decisions)\n")
    print(f"{'skill':20s} {'flat':>8s} {'rough':>8s} {'delta':>8s}   per-seed deltas")
    for sk in skills:
        bs = [usage(b)[sk] for _, b, _ in pairs]
        rs = [usage(r)[sk] for _, _, r in pairs]
        deltas = [r - b for b, r in zip(bs, rs)]
        per = ", ".join(f"s{s}:{d:+.3f}" for (s, _, _), d in zip(pairs, deltas))
        print(
            f"{sk:20s} {np.mean(bs):8.3f} {np.mean(rs):8.3f} {np.mean(deltas):+8.3f}   {per}"
        )

    print()
    for key, label in [
        ("policy_diag/primary_success_rate", "tracking success"),
        ("policy_diag/go1/no_fall_rate", "no-fall rate"),
        ("rollout/episode_length", "episode length"),
    ]:
        bs = [scalar(b, key) for _, b, _ in pairs]
        rs = [scalar(r, key) for _, _, r in pairs]
        if any(x is None for x in bs + rs):
            continue
        print(f"{label:20s} {np.mean(bs):8.3f} {np.mean(rs):8.3f} {np.mean(rs) - np.mean(bs):+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
