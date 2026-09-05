#!/usr/bin/env python3
"""Render the per-environment training curves once, to PNGs the report can just embed.

``build_dashboard.py`` draws these inline every time it runs, which is why it takes minutes: the
curves live inside the run pickles, so producing them means unpickling the whole ``runs/verify``
and ``runs/viper`` trees. Splitting the render out means the report build stays at a second or
two and these only get regenerated when new cells land.

Three plots per environment, the same three the board carried:

  * **training return** -- the gameable quantity, methods overlaid, band = seed spread
  * **primary success** -- the metric the campaign is actually scored on
  * **per-skill returns** -- each skill's own hand-written reward, from the first hierarchical
    variant available

Run collection (de-duplication, probe exclusion, canonical-tag preference) is imported from
``build_dashboard`` rather than reimplemented -- those rules were each written to fix a specific
distortion in the seed bands and must not drift between the two tools.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dashboard import (  # noqa: E402
    SKILL_COLORS,
    VARIANT_COLORS,
    _series,
    collect_runs,
)

RETURN_KEY = "rollout/episode_return"
# NOT "primary_success_rate": the trainer logs it under the policy_diag namespace, and the bare
# key silently yields None for every run -- which once emptied this plot on the whole board.
SUCCESS_KEY = "policy_diag/primary_success_rate"


def _fig(figsize=(5.8, 3.4)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(figsize=figsize, dpi=150)


def _finish(plt, fig, ax, title: str, ylabel: str, out: Path) -> None:
    ax.set_xlabel("update")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Legend outside the data box: with four variants and a seed band the in-box legend on the
    # old board sat on top of the curves it was labelling.
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22),
              ncol=3, borderaxespad=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def curve(series: dict[str, list], title: str, ylabel: str, out: Path) -> bool:
    """Mean +/- seed spread, one line per variant."""
    plt, (fig, ax) = _fig()
    drew = False
    for variant, runs in sorted(series.items()):
        runs = [r for r in runs if r is not None and r.size > 1]
        if not runs:
            continue
        n = min(len(r) for r in runs)
        stack = np.stack([r[:n] for r in runs])
        mean, std = stack.mean(0), stack.std(0)
        x = np.arange(n)
        c = VARIANT_COLORS.get(variant)
        ax.plot(x, mean, label=f"{variant} (n={len(runs)})", color=c, lw=1.7)
        if len(runs) > 1:
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.16, lw=0)
        drew = True
    if not drew:
        plt.close(fig)
        return False
    _finish(plt, fig, ax, title, ylabel, out)
    return True


def skills(metrics: dict, title: str, out: Path) -> bool:
    keys = sorted(k for k in metrics if k.startswith("skill_return/"))
    if not keys:
        return False
    plt, (fig, ax) = _fig()
    drew = False
    for i, k in enumerate(keys):
        s = _series(metrics, k)
        if s is None or s.size < 2:
            continue
        ax.plot(s, label=k.split("/", 1)[1], color=SKILL_COLORS[i % len(SKILL_COLORS)], lw=1.5)
        drew = True
    if not drew:
        plt.close(fig)
        return False
    _finish(plt, fig, ax, title, "episodic skill return", out)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["runs/verify", "runs/viper"])
    ap.add_argument("--out", default="runs/env_curves")
    args = ap.parse_args(argv)

    out = Path(args.out)
    print(f"collecting from {', '.join(args.dirs)} ...")
    runs = collect_runs([Path(d) for d in args.dirs])
    print(f"{len(runs)} environments, "
          f"{sum(len(rs) for v in runs.values() for rs in v.values())} runs")

    n = 0
    for env in sorted(runs):
        variants = runs[env]
        print(f"{env}  ({', '.join(f'{v}x{len(rs)}' for v, rs in sorted(variants.items()))})")
        n += curve({v: [_series(r["metrics"], RETURN_KEY) for r in rs]
                    for v, rs in variants.items()},
                   f"{env} — training return", "episode return", out / f"{env}__return.png")
        n += curve({v: [_series(r["metrics"], SUCCESS_KEY) for r in rs]
                    for v, rs in variants.items()},
                   f"{env} — primary success", "success rate", out / f"{env}__success.png")
        # Per-skill reward exists only for the hierarchical arms; take the first that has it.
        for v in ("nesy", "neural", "symbolic"):
            if variants.get(v) and skills(variants[v][0]["metrics"],
                                          f"{env} [{v}] — per-skill returns",
                                          out / f"{env}__skills.png"):
                n += 1
                break

    print(f"\nwrote {n} figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
