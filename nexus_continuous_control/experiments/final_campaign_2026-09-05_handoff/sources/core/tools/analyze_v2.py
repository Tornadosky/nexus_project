#!/usr/bin/env python3
"""V2 matrix — the campaign gate, scored on PRIMARY SUCCESS, never on return.

Gate (docs/VERIFICATION_PLAN.md §V2): in >= 4 of 6 environments, `nesy` or `neural` beats
`flat` on the primary success metric. Return is explicitly not the gate: the campaign has
measured panda earning 655 return with 0.001 lift success, and walker's old 931-return
"success" was a vertical-axis artifact.

Three things this tool refuses to do, each because the campaign already got burned by it:

  * **Merge recipe variants.** `walker_flat` exists under both the dm-suite and locomotion
    recipes. They are different experiments and are kept as separate arms, keyed by the tag in
    the filename — reading ENV_NAME/META_POLICY_TYPE out of the config alone would silently
    average them into one bar.
  * **Report a mean without its seeds.** Every cell carries its per-seed list, and the gate
    comparison additionally reports whether the seed ranges overlap. Panda `flat` is bimodal
    (0.585 / 0.603 / 0.001); its mean is not a description of anything.
  * **Call a cell done when seeds are missing.** Cells below `--min-seeds` are marked partial
    and excluded from the gate count rather than quietly counted.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = ["flat", "neural", "symbolic", "nesy"]
COLORS = {"flat": "#8792a2", "neural": "#343C96", "symbolic": "#0F6C74", "nesy": "#1B6B45"}
SUCCESS_KEY = "policy_diag/primary_success_rate"
RETURN_KEY = "rollout/episode_return"


def _tail_mean(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    a = np.asarray(metrics[key]).reshape(-1)
    if a.size == 0:
        return None
    return float(a[-max(1, a.size // 10) :].mean())


def _tag_from(stem: str, variant: str) -> str:
    """Recipe/campaign tag sitting between the variant and the seed in the filename.

    Every tag is kept. An earlier version of this function collapsed `explore` and `v2` as
    "campaign labels with no experimental contrast" — which was wrong and produced a wrong
    result: `cartpole_balance_flat_s0` (V1.2, shipped noise 0.30->0.02) and
    `cartpole_balance_flat_explore_s0` (V1.3, corrected noise 1.0->0.15) collapsed onto the same
    (env, arm, seed) key, one silently overwrote the other, and cartpole `flat` was reported at
    the shipped-noise number while being compared against corrected-noise hierarchical arms.
    Tags are contrasts until proven otherwise; never collapse one to tidy up a label.
    """
    m = re.search(rf"{re.escape(variant)}_(.*?)_?s\d+$", stem)
    return (m.group(1) if m else "").strip("_")


def collect(
    dirs: list[Path], min_seeds: int, exclude: tuple[str, ...] = ("diag",)
) -> dict[str, dict[str, dict[str, Any]]]:
    cells: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for d in dirs:
        if not d.exists():
            continue
        for pkl in sorted(d.rglob("*.pkl")):
            try:
                with open(pkl, "rb") as fh:
                    ck = pickle.load(fh)
            except Exception:
                continue
            cfg = ck.get("config", {}) or {}
            metrics = ck.get("metrics", {}) or {}
            env = str(cfg.get("ENV_NAME") or "?")
            variant = str(cfg.get("META_POLICY_TYPE") or "?").lower()
            if variant not in VARIANTS:
                continue
            if any(pkl.stem.startswith(p) for p in exclude):
                continue
            seed = int(cfg.get("SEED", -1))
            tag = _tag_from(pkl.stem, variant)
            arm = f"{variant}·{tag}" if tag else variant
            succ = _tail_mean(metrics, SUCCESS_KEY)
            if succ is None:
                continue
            # Budget is part of the experiment's identity. A reduced-batch or short-budget probe
            # is not a seed of the gated cell, and averaging one into an arm silently fabricates
            # a result — the 512-env/50-update ROCm diagnostic landed in WalkerWalk `nesy` and
            # reported 0.224 as if it were a real walker cell.
            budget = (int(cfg.get("NUM_ENVS", 0)), int(cfg.get("TOTAL_TIMESTEPS", 0)))
            # A collision means two checkpoints claim the same (env, arm, seed). That is only
            # legitimate for a genuine re-run; if it happens because two different experiments
            # share a key, silently keeping the last one fabricates a comparison. Say so.
            if seed in cells[(env, arm)]:
                print(
                    f"  WARNING collision {env}/{arm}/s{seed}: "
                    f"{cells[(env, arm)][seed]['path']} <- {pkl}"
                )
            cells[(env, arm)][seed] = {
                "success": succ,
                "return": _tail_mean(metrics, RETURN_KEY),
                "path": str(pkl),
                "budget": budget,
            }

    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (env, arm), by_seed in cells.items():
        seeds = sorted(by_seed)
        succ = [by_seed[s]["success"] for s in seeds]
        rets = [by_seed[s]["return"] for s in seeds]
        budgets = {by_seed[s]["budget"] for s in seeds}
        if len(budgets) > 1:
            print(f"  WARNING {env}/{arm} mixes budgets (NUM_ENVS, TOTAL_TIMESTEPS): {sorted(budgets)}")
        out[env][arm] = {
            "seeds": seeds,
            "success": succ,
            "success_mean": float(np.mean(succ)),
            "return": rets,
            "return_mean": float(np.mean([r for r in rets if r is not None])) if any(
                r is not None for r in rets
            ) else None,
            "n": len(seeds),
            "partial": len(seeds) < min_seeds,
            "budgets": sorted(budgets),
            "mixed_budget": len(budgets) > 1,
        }
    return out


# Tags marking arms that changed a HYPERPARAMETER rather than a seed/recipe. They must never
# enter the gate: the gate compares hierarchical vs flat at the SAME settings, and every `flat`
# baseline was run at the shipped ones. Letting a tuned hierarchical arm face an untuned flat
# baseline manufactured a "GATE PASS 4 of 6" on 2026-08-11 — cartpole's `neural` at
# MAX_GRAD_NORM=20 was being scored against flat at MAX_GRAD_NORM=1.0. Different experiments.
EXPERIMENTAL_TAGS = frozenset(
    {"noclip", "clip20", "clip05", "clip50", "scaleclip", "commit10", "hpqn", "quarter",
     # budget-scaled arms (4x TOTAL_TIMESTEPS, some with exploration overrides on top) are a
     # different experiment: a hierarchical budget4x cell facing a 1x flat baseline is the same
     # false-pass shape the 2026-08-11 tooling bug manufactured. They are reported in the
     # experimental block, and may only be compared against each other.
     # Match the FAMILY, not each multiple. Listing "budget2x" and "budget4x" individually left
     # `budget8x` and `budget16x` unguarded when those cells were added on 2026-08-12, and
     # `flat·budget8x` (0.3454 at EIGHT TIMES the baseline budget) promptly entered the gate as
     # the best flat arm and flipped HopperHop from a matched hierarchical win into a loss — the
     # exact false comparison this set exists to prevent, arriving through a tag nobody had
     # thought to add. Substring "budget" catches every present and future multiple.
     "budget",
     # revised-rule policy modules (different POLICY => different experiment) and any PPO arm
     "rules2", "rules3", "ppo"}
)


def _is_experimental(arm: str) -> bool:
    """True if any tag component names a hyperparameter change.

    Substring match, not equality: composite tags like `dm_quarter` (walker at 1/4 budget) and
    `dm_clip50` (walker flat at MAX_GRAD_NORM=5) must be caught too. Those are `flat` arms, so
    they could not produce a false PASS — the gate takes flat's BEST arm, so an extra flat arm
    only ever raises the bar — but they are still different experiments and do not belong in a
    gate comparison in either direction.
    """
    tags = arm.split("·")[1:]
    return any(marker in t for t in tags for marker in EXPERIMENTAL_TAGS)


def _cell_budget(cell: dict) -> int | None:
    """A cell's env-step budget, or None if its own seeds disagree.

    `collect` records (num_envs, TOTAL_TIMESTEPS) per seed. A cell whose seeds were trained at
    different budgets has no single budget to compare, and returning None makes the caller say
    "unknown" instead of silently picking one.
    """
    steps = {b[1] for b in (cell.get("budgets") or []) if isinstance(b, (list, tuple)) and len(b) > 1}
    return next(iter(steps)) if len(steps) == 1 else None


def gate(matrix: dict, min_seeds: int) -> dict[str, Any]:
    """Per env: does any hierarchical arm beat every flat arm on primary success?

    Only arms at shipped hyperparameters are eligible; `EXPERIMENTAL_TAGS` arms are excluded
    and reported separately so a tuning result cannot be read as a gate result.
    """
    verdicts = {}
    for env, arms in sorted(matrix.items()):
        arms = {a: c for a, c in arms.items() if not _is_experimental(a)}
        flats = {a: c for a, c in arms.items() if a.split("·")[0] == "flat" and not c["partial"]}
        hiers = {
            a: c
            for a, c in arms.items()
            if a.split("·")[0] in ("neural", "nesy") and not c["partial"]
        }
        if not flats or not hiers:
            verdicts[env] = {
                "status": "incomplete",
                "why": f"{len(flats)} complete flat arm(s), {len(hiers)} complete hierarchical arm(s)",
            }
            continue
        best_flat_arm = max(flats, key=lambda a: flats[a]["success_mean"])
        # Pick the best hierarchical arm from those BUDGET-MATCHED to that flat arm, and only
        # fall back to the overall best (which then gets flagged) when no matched arm exists.
        #
        # Taking the highest mean first and flagging afterwards throws away real results: on
        # HopperHop the highest hierarchical mean is nesy·v2 at 52,428,800, so the env reported
        # "BUDGET MISMATCH — NOT a result" while neural·v2 sat right there at the baseline's own
        # 26,214,400 scoring 0.2201 against flat's 0.1224. A comparison that IS matched should
        # never be discarded because a mismatched arm happened to score higher.
        flat_budget = _cell_budget(flats[best_flat_arm])
        matched_hiers = (
            {a: c for a, c in hiers.items() if _cell_budget(c) == flat_budget}
            if flat_budget is not None else {}
        )
        pool = matched_hiers or hiers
        best_hier_arm = max(pool, key=lambda a: pool[a]["success_mean"])
        bf, bh = flats[best_flat_arm], hiers[best_hier_arm]
        beats = bh["success_mean"] > bf["success_mean"]
        # Separation: do the seed ranges overlap? A mean win inside overlapping seeds is not one.
        separated = min(bh["success"]) > max(bf["success"])
        # Budget parity. `EXPERIMENTAL_TAGS` keeps budget-SCALED arms (`budget2x`, `quarter`) out
        # of the gate, but it keys on the cell NAME, so an asymmetry carrying no tag passes
        # straight through: on 2026-08-12 `hopper_hop_nesy_v2` was found training at 52,428,800
        # against `hopper_hop_flat`'s 26,214,400, and this function reported it as
        # "nesy·v2 BEATS flat" with nothing said about the 2x. A win at twice the baseline's
        # environment steps is not a win, and the gate should be the thing that says so rather
        # than a separate audit someone has to remember to run.
        budget_ok = None
        if _cell_budget(bf) is not None and _cell_budget(bh) is not None:
            budget_ok = _cell_budget(bf) == _cell_budget(bh)
        verdicts[env] = {
            "status": "beats_flat" if beats else "loses_to_flat",
            "separated": bool(separated),
            "budget_matched": budget_ok,
            "best_flat_budget": _cell_budget(bf),
            "best_hier_budget": _cell_budget(bh),
            "best_flat": best_flat_arm,
            "best_flat_success": bf["success_mean"],
            "best_flat_seeds": bf["success"],
            "best_hier": best_hier_arm,
            "best_hier_success": bh["success_mean"],
            "best_hier_seeds": bh["success"],
        }
    passing = [e for e, v in verdicts.items() if v.get("status") == "beats_flat"]
    # A "win" whose seed ranges overlap is not a win. Reported separately because the gate as
    # written in the plan has no magnitude or overlap requirement, so a 0.0001 mean difference
    # on a saturated env satisfies its letter -- which is exactly what WalkerWalk did on
    # 2026-08-11 (neural 0.9646 vs flat 0.9645) and would have flipped the campaign to "PASS".
    separated = [
        e for e, v in verdicts.items()
        if v.get("status") == "beats_flat" and v.get("separated")
        and v.get("budget_matched") is not False
    ]
    # Reported on its own so the asymmetry is visible rather than merely subtracted.
    unmatched = [e for e, v in verdicts.items() if v.get("budget_matched") is False]
    scored = [e for e, v in verdicts.items() if v.get("status") in ("beats_flat", "loses_to_flat")]
    return {
        "per_env": verdicts,
        "n_pass": len(passing),
        "n_scored": len(scored),
        "n_separated": len(separated),
        "separated_envs": separated,
        "budget_unmatched_envs": unmatched,
        "gate_letter": "PASS" if len(passing) >= 4 else ("FAIL" if len(scored) >= 6 else "INCOMPLETE"),
        "gate": "PASS" if len(separated) >= 4 else ("FAIL" if len(scored) >= 6 else "INCOMPLETE"),
    }


def _panel(ax, env: str, arms: dict, xlabel: bool) -> None:
    """One environment, drawn as horizontal bars.

    Arm names are long (`nesy·budget4x·clip20`) and there are up to eighteen of them in one
    environment. On a vertical-bar chart that name has to fit a ~40px tick slot, so it wraps and
    then overprints its neighbours -- which is what this panel replaces. On the y-axis each name
    gets a full gutter and cannot collide with anything at any arm count.
    """
    import numpy as np

    names = sorted(arms, key=lambda a: (VARIANTS.index(a.split("·")[0]), a))
    vals = [arms[a]["success_mean"] for a in names]
    cols = [COLORS[a.split("·")[0]] for a in names]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=cols, height=0.66)

    for i, a in enumerate(names):
        pts = arms[a]["success"]
        # Seeds are the evidence, so every one has to stay individually visible: spread them
        # across the bar's thickness deterministically rather than stacking them on one line.
        off = np.linspace(-0.19, 0.19, len(pts)) if len(pts) > 1 else np.zeros(1)
        ax.scatter(pts, y[i] + off, color="#151A23", s=18, zorder=3, alpha=0.85,
                   linewidths=0.5, edgecolors="white")
        if arms[a]["partial"]:
            bars[i].set_hatch("///")
            bars[i].set_edgecolor("#8A3227")

    # Hairline between variant blocks -- flat / neural / symbolic / nesy read as groups without
    # needing a legend inside the plot box.
    fams = [n.split("·")[0] for n in names]
    for i in range(1, len(fams)):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color="#B9C1CD", lw=0.7, zorder=0)

    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(vals) * 1.12) if vals else 1.0)
    if xlabel:
        ax.set_xlabel("primary success rate", fontsize=9)
    ax.set_title(env, fontsize=11, loc="left", pad=6)
    ax.grid(alpha=0.18, lw=0.6, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_height(n_arms: int) -> float:
    """Inches for a panel holding `n_arms` bars: a fixed row pitch plus title and axis furniture."""
    return 0.30 * n_arms + 1.25


SUBTITLE = "primary success by env and variant (dots = seeds; hatched = incomplete)"


def plot(matrix: dict, out: Path) -> list[Path]:
    """Write one figure per environment, plus a stacked overview at `out`.

    Returns every path written, `out` first. Per-env files are `<out stem>_<env>.png` beside it;
    the dashboard embeds those so each environment is sized by its own arm count instead of every
    environment sharing one grid cell height.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = sorted(matrix)
    if not envs:
        return []
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    heights = [_panel_height(len(matrix[e])) for e in envs]
    fig, axes = plt.subplots(
        len(envs), 1, figsize=(7.6, sum(heights)), dpi=140,
        gridspec_kw={"height_ratios": heights},
    )
    for ax, env in zip(np.atleast_1d(axes).ravel(), envs):
        _panel(ax, env, matrix[env], xlabel=env == envs[-1])
    fig.suptitle(f"V2 matrix — {SUBTITLE}", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    written.append(out)
    print(f"wrote {out}")

    for env in envs:
        f, ax = plt.subplots(figsize=(7.4, _panel_height(len(matrix[env]))), dpi=140)
        _panel(ax, env, matrix[env], xlabel=True)
        ax.set_title(env, fontsize=12, loc="left", pad=8)
        f.text(0.0, 1.0, SUBTITLE, fontsize=8, color="#6B7686", va="bottom")
        f.tight_layout()
        p = out.with_name(f"{out.stem}_{env}{out.suffix}")
        f.savefig(p, bbox_inches="tight")
        plt.close(f)
        written.append(p)
        print(f"wrote {p}")

    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["runs/verify", "runs/viper"])
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument(
        "--exclude", nargs="*", default=["diag"],
        help="filename-stem prefixes to skip; probes are not seeds of a gated cell",
    )
    ap.add_argument("--out", default="runs/v2")
    args = ap.parse_args(argv)

    out = Path(args.out)
    matrix = collect([Path(d) for d in args.dirs], args.min_seeds, tuple(args.exclude))
    g = gate(matrix, args.min_seeds)

    print("=" * 92)
    print("V2 MATRIX — primary success (NOT return)")
    print("=" * 92)
    for env in sorted(matrix):
        print(f"\n{env}")
        for arm in sorted(matrix[env], key=lambda a: (VARIANTS.index(a.split('·')[0]), a)):
            c = matrix[env][arm]
            flag = "  [PARTIAL]" if c["partial"] else ""
            seeds = ", ".join(f"{x:.3f}" for x in c["success"])
            ret = f"{c['return_mean']:8.1f}" if c["return_mean"] is not None else "       -"
            print(f"  {arm:<18} n={c['n']}  success={c['success_mean']:.4f}  [{seeds}]  return={ret}{flag}")

    print("\n" + "=" * 92)
    print("GATE — >=4 of 6 envs where nesy or neural beats flat on primary success")
    print("=" * 92)
    for env, v in g["per_env"].items():
        if v["status"] == "incomplete":
            print(f"  {env:<26} INCOMPLETE  ({v['why']})")
        else:
            mark = "BEATS" if v["status"] == "beats_flat" else "loses to"
            sep = "" if v["status"] != "beats_flat" else (
                "  (seed-separated)" if v["separated"] else "  (WITHIN seed overlap — not a result)"
            )
            if v.get("budget_matched") is False:
                bh_b, bf_b = v["best_hier_budget"], v["best_flat_budget"]
                sep = (f"  (BUDGET MISMATCH {bh_b:,} vs flat {bf_b:,} "
                       f"= {bh_b / bf_b:.2g}x — NOT a result)")
            print(
                f"  {env:<26} {v['best_hier']:<14} {v['best_hier_success']:.4f}  {mark:>8}  "
                f"flat {v['best_flat_success']:.4f}{sep}"
            )
    print(f"\n  mean-win count : {g['n_pass']} of {g['n_scored']}  ->  gate BY THE LETTER: {g['gate_letter']}")
    print(f"  seed-separated : {g['n_separated']} of {g['n_scored']}  ->  GATE: {g['gate']}")
    print("  A mean win inside seed overlap is not a result. WalkerWalk meets the letter by")
    print("  0.0001 on a saturated env; the seed-separated count is the defensible one.")
    if g.get("budget_unmatched_envs"):
        print("")
        print("  BUDGET-UNMATCHED, excluded from the seed-separated count: "
              + ", ".join(g["budget_unmatched_envs"]))
        print("  These arms beat flat at a different number of environment steps. EXPERIMENTAL_TAGS")
        print("  cannot catch it — it keys on the cell name, and the asymmetry carries no tag.")
        print("  Run tools/audit_budgets.py for the full per-cell table.")

    plot(matrix, out / "v2_matrix.png")
    (out / "v2_matrix.json").write_text(
        json.dumps({"matrix": matrix, "gate": g}, indent=2), encoding="utf-8"
    )
    print(f"wrote {out / 'v2_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
