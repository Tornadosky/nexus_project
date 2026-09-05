#!/usr/bin/env python3
"""V3.3 — per-skill episodic returns: the paper's Fig. 3 (Q1 disentanglement), NEXUS vs HPQN.

The paper's Q1 claim is that skill-specific rewards produce *disentangled* skills: every skill
learns something of its own, rather than the agent collapsing onto one or two. The control that
makes the claim testable is HPQN — the identical hierarchy with every skill trained on the env
reward instead of its own — which this repo added as `SHARED_SKILL_REWARD`.

**Read the caveat before quoting any number from this tool.** The obvious statistic — spread of
the per-skill *returns* — is **circular for HPQN**. `skill_return/{i}` accumulates
`skill_rewards[..., i]`, and under `SHARED_SKILL_REWARD` those are all copies of the env reward
*by construction*. HPQN's spread is therefore exactly 0 no matter what the skills do; it measures
the reward definition we imposed, not any property of the learned policy. It is retained in the
JSON under an explicitly ugly key so nobody quotes it by accident.

The non-circular measure is **skill usage entropy**: how often the meta-policy actually selects
each skill, normalised to [0, 1]. It is the meta-Q's revealed preference, and it is free to come
out any way at all under either arm.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

SKILL_COLORS = ["#4C6EF5", "#F59F00", "#0CA678", "#E03131", "#7048E8"]


def load(path: Path) -> tuple[list[str], np.ndarray] | None:
    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    m = ck.get("metrics", {}) or {}
    keys = sorted(k for k in m if k.startswith("skill_return/"))
    if not keys:
        return None
    names = [k.split("/", 1)[1] for k in keys]
    curves = np.stack([np.asarray(m[k]).reshape(-1) for k in keys])  # [skills, updates]
    return names, curves


def usage_entropy(path: Path) -> tuple[float, list[float]] | None:
    """How often the meta-policy actually SELECTS each skill, as normalised entropy in [0, 1].

    This exists because the per-skill *return* spread is tautological for HPQN: under
    `SHARED_SKILL_REWARD` every skill's reward is a copy of the env reward, so their episodic
    returns are identical by construction, not by measurement. Reporting that as evidence of
    entanglement would be circular.

    Skill *usage* is not circular. It is the meta-Q's revealed preference over skills, and it is
    free to differ between NEXUS and HPQN. 1.0 = the meta spreads selection evenly over skills;
    0.0 = it has collapsed onto exactly one.
    """
    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    m = ck.get("metrics", {}) or {}
    keys = sorted(k for k in m if k.startswith("skill_usage/"))
    if not keys:
        return None
    arrs = [np.asarray(m[k]).reshape(-1) for k in keys]
    n = max(1, len(arrs[0]) // 10)
    p = np.array([a[-n:].mean() for a in arrs], dtype=float)
    total = p.sum()
    if total <= 1e-9:
        return None
    p = p / total
    nz = p[p > 1e-12]
    ent = float(-(nz * np.log(nz)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    return ent, p.tolist()


def spread_metric(curves: np.ndarray, tail: float = 0.1) -> float:
    """Mean pairwise |difference| of final per-skill returns, scaled by mean |return|.

    Scale-free so hopper and Go1 are comparable. ~0 means the skills ended up interchangeable.
    """
    n = max(1, int(curves.shape[1] * tail))
    finals = curves[:, -n:].mean(axis=1)
    k = len(finals)
    if k < 2:
        return 0.0
    diffs = [abs(finals[i] - finals[j]) for i in range(k) for j in range(i + 1, k)]
    denom = np.mean(np.abs(finals))
    return float(np.mean(diffs) / denom) if denom > 1e-9 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=[
            "HopperHop:runs/verify/hopper_hop_neural_v2_s0.pkl:runs/verify/hopper_hop_neural_hpqn_s0.pkl",
            "Go1:runs/verify/go1_joystick_neural_v2_s0.pkl:runs/verify/go1_joystick_neural_hpqn_s0.pkl",
        ],
        help="LABEL:nexus_ckpt:hpqn_ckpt",
    )
    ap.add_argument("--out", default="runs/v3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for spec in args.pairs:
        label, nexus_p, hpqn_p = spec.split(":", 2)
        a, b = load(Path(nexus_p)), load(Path(hpqn_p))
        if a is None or b is None:
            print(f"  SKIP {label}: missing skill_return/* in one of the checkpoints")
            continue
        entries.append((label, a, b, usage_entropy(Path(nexus_p)), usage_entropy(Path(hpqn_p))))

    if not entries:
        print("nothing to plot")
        return 1

    fig, axes = plt.subplots(len(entries), 2, figsize=(12.5, 4.0 * len(entries)), dpi=140,
                             squeeze=False)
    summary: dict[str, Any] = {}
    for r, (label, (names, cn), (_, ch), un, uh) in enumerate(entries):
        for c, (curves, arm, usage) in enumerate(((cn, "NEXUS (per-skill rewards)", un),
                                                  (ch, "HPQN (shared env reward)", uh))):
            ax = axes[r][c]
            for i, nm in enumerate(names):
                ax.plot(curves[i], color=SKILL_COLORS[i % len(SKILL_COLORS)], lw=1.5, label=nm)
            sp = spread_metric(curves)
            ax.set_title(f"{label} — {arm}\nskill spread = {sp:.2f}", fontsize=10, loc="left")
            ax.set_xlabel("update")
            ax.set_ylabel("episodic skill return")
            ax.grid(alpha=0.18, lw=0.6)
            ax.legend(fontsize=7, frameon=False)
            summary.setdefault(label, {})["nexus" if c == 0 else "hpqn"] = {
                "skills": names,
                "spread_TAUTOLOGICAL_FOR_HPQN": sp,
                "usage_entropy": usage[0] if usage else None,
                "usage_fractions": usage[1] if usage else None,
                "finals": curves[:, -max(1, curves.shape[1] // 10):].mean(axis=1).tolist(),
            }

    fig.suptitle(
        "V3.3 / paper Fig. 3 — per-skill episodic returns. Do skill-specific rewards "
        "disentangle the skills?",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "skill_returns.png", bbox_inches="tight")
    print(f"wrote {out / 'skill_returns.png'}")

    print()
    print("NOTE: per-skill RETURN spread is tautological for HPQN (all skill rewards are copies")
    print("      of the env reward by construction). Usage entropy is the non-circular measure.")
    print()
    print(f"{'env':<12} {'arm':<8} {'usage_ent':>10} {'usage fractions':<34} per-skill final returns")
    for label, d in summary.items():
        for arm in ("nexus", "hpqn"):
            e = d[arm]
            ue = f"{e['usage_entropy']:.3f}" if e["usage_entropy"] is not None else "  n/a"
            uf = str([round(x, 3) for x in e["usage_fractions"]]) if e["usage_fractions"] else "-"
            print(f"{label:<12} {arm:<8} {ue:>10} {uf:<34} {[round(x, 1) for x in e['finals']]}")
    (out / "skill_returns.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'skill_returns.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
