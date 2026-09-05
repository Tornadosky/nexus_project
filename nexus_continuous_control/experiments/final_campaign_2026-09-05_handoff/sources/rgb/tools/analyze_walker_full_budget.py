"""Does WalkerWalk's state+RGB advantage survive the paper's full budget?

THE QUESTION
------------
At the matched budget (2.05M env steps = 250 updates x 128 envs x 64 steps)
WalkerWalk was the ONE environment where giving an already-state-fed skill actor
a 64x64 grayscale 3-frame camera helped:

    training return (last-20-update mean)  189.51 +/- 1.80 -> 203.20 +/- 6.25
                                           +7.2%, 3/3 seeds, non-overlapping
    30-episode eval (reward/step)          0.7130 -> 0.8157  (+14.4%)
    camera actually used                   frozen_first costs 62-72%

A +7% edge at 4% of the paper's budget is exactly the shape of an optimisation
artefact: an extra input can act as a regulariser or an extra gradient path
early and be redundant once the state-only arm has trained to convergence. This
file re-runs the whole comparison at 52,428,800 steps (6400 updates x 128 envs
x 64 steps, 25.6x) and reports BOTH budgets together, because the interesting
quantity is not the gap at either budget -- it is how the gap MOVES with budget.

WHAT IT DOES NOT DO
-------------------
It does not touch results/rgb/state_plus_rgb/ or .../state_plus_rgb_eval30/ --
the committed 2.05M campaign -- and it does not modify any result JSON. Every
number is recomputed from raw files and written to a new JSON that records the
source path of every input.

STATISTICAL CONVENTIONS (identical to tools/analyze_state_plus_rgb.py, whose
helpers are imported rather than re-implemented so the two cannot drift):
  * SAMPLE s.d. (ddof=1) everywhere. np.std's ddof=0 default understates the
    spread by 1.22x at n=3.
  * PRIMARY metric = last-20-update mean of the 128-env training curve. The
    5-episode deterministic eval has ~0.16 achieved power and inverted walker
    once already; the 30-episode re-score is the eval metric that is quoted.
  * Both Welch (unpaired) and paired-by-seed tests are reported. Evaluation
    reset keys are PRNGKey(9000 + 97*episode + seed) -- a function of (episode,
    seed) and NOT of the arm -- so the arms are genuinely paired by seed.
  * Seed-range overlap is reported explicitly: at n=3 it carries more of the
    signal than any p-value.

    python tools/analyze_walker_full_budget.py
    python tools/analyze_walker_full_budget.py --figures-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ENV = "walker"
SEEDS = (0, 1, 2)
LAST_N = 20
CONDITIONS = ("intact", "frozen_first", "random_replay", "shuffle_frames",
              "zeros", "const_action")

# (label, steps, train+5ep root, 30ep root, arm-name suffix)
BUDGETS = [
    ("2.05M", 2_048_000,
     "results/rgb/state_plus_rgb", "results/rgb/state_plus_rgb_eval30", ""),
    ("52.4M", 52_428_800,
     "results/rgb/state_plus_rgb_full", "results/rgb/state_plus_rgb_full_eval30",
     "_full"),
]
ARMS = ["state_matched", "state_plus_rgb"]
ARM_LABEL = {"state_matched": "state only", "state_plus_rgb": "state + RGB"}

# Fixed categorical order, assigned to the ENTITY (the arm) and never cycled.
# Budget is carried by FACET, not by extra hues -- adding a third and fourth hue
# for "2.05M" and "52.4M" would encode two different things in one channel.
# Validated: adjacent-pair CVD dE 20.0 (protan) / 29.4 (tritan), normal 26.9.
# The orange sits at 2.73:1 on white, below the 3:1 bar, so every bar carries a
# visible direct label and the tables below are the required table view.
C_STATE = "#4C72B0"   # state only  (baseline)
C_SPRGB = "#DD8452"   # state + RGB (extension)
ARM_COL = {"state_matched": C_STATE, "state_plus_rgb": C_SPRGB}
C_INK = "#222222"
C_MUTED = "#666666"
CORR_COL = {"frozen_first": "#C44E52", "random_replay": "#DD8452",
            "shuffle_frames": "#8172B3", "zeros": "#937860",
            "const_action": "#8C8C8C"}


def _load_helpers(root: Path):
    """Import welch/paired/achieved_power from the campaign analyser."""
    p = root / "tools" / "analyze_state_plus_rgb.py"
    spec = importlib.util.spec_from_file_location("_aspr", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------ io
def arm_name(arm: str, suffix: str) -> str:
    return f"{arm}{suffix}"


def read_run(root: Path, arm: str, seed: int):
    d = root / ENV / f"{arm}_seed{seed}"
    abl, cur = d / "pixel_ablation.json", d / "training_curves.json"
    out = {"dir": str(d), "exists": abl.exists()}
    if abl.exists():
        out["ablation"] = json.loads(abl.read_text())
        out["ablation_path"] = str(abl)
    if cur.exists():
        out["curves"] = json.loads(cur.read_text())
        out["curves_path"] = str(cur)
    return out


def train_metric(run) -> float | None:
    """Last-20-update mean of the 128-env training curve. THE primary metric.

    Recomputed from training_curves.json rather than trusting the stored
    `final_train_return`, so a change in the trainer's own summary convention
    cannot silently move the headline. They agree on every 2.05M run.
    """
    c = run.get("curves")
    if c:
        curve = (c.get("curves") or c).get("episode_return")
        if curve:
            tail = curve[-LAST_N:]
            return float(sum(tail) / len(tail))
    a = run.get("ablation")
    return float(a["final_train_return"]) if a and a.get("final_train_return") is not None else None


def eval_metric(run) -> float | None:
    a = run.get("ablation")
    if not a:
        return None
    return float(a["results"]["intact"]["reward_per_step_mean"])


def n_episodes(run) -> int | None:
    a = run.get("ablation")
    if not a:
        return None
    return len(a["results"]["intact"].get("per_episode", [])) or a.get("episodes")


# ----------------------------------------------------------------- statistics
def summarise(vals):
    import numpy as np
    v = [x for x in vals if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(a.size), "per_seed": [float(x) for x in a],
            "mean": float(a.mean()),
            "sd_sample_ddof1": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def compare(base_vals, ext_vals, H):
    """Full two-arm comparison. base = state only, ext = state + RGB."""
    import numpy as np
    b = [x for x in base_vals if x is not None]
    e = [x for x in ext_vals if x is not None]
    if len(b) < 1 or len(e) < 1:
        return None
    out = {"state_matched": summarise(b), "state_plus_rgb": summarise(e)}
    out["abs_change"] = out["state_plus_rgb"]["mean"] - out["state_matched"]["mean"]
    out["pct_change"] = (100.0 * out["abs_change"] / out["state_matched"]["mean"]
                         if out["state_matched"]["mean"] else None)
    # Seed-range overlap: at n=3 this says more than any p-value.
    lo = max(min(b), min(e))
    hi = min(max(b), max(e))
    out["ranges_overlap"] = bool(lo <= hi)
    out["range_state_matched"] = [float(min(b)), float(max(b))]
    out["range_state_plus_rgb"] = [float(min(e)), float(max(e))]
    if len(b) > 1 and len(e) > 1:
        out["welch"] = H.welch(b, e)
        out["achieved_power"] = H.achieved_power(b, e)
    if len(b) == len(e) and len(b) > 1:
        out["paired_by_seed"] = H.paired(b, e)
    return out


def fmt_cmp(c, prec=3):
    if not c:
        return "    (incomplete)"
    a, x = c["state_matched"], c["state_plus_rgb"]
    L = []
    for nm, s in (("state only ", a), ("state + RGB", x)):
        L.append(f"    {nm}  n={s['n']}  per-seed "
                 + "[" + ", ".join(f"{v:.{prec}f}" for v in s["per_seed"]) + "]"
                 + f"  mean {s['mean']:.{prec}f}  sd(ddof1) {s['sd_sample_ddof1']:.{prec}f}"
                 + f"  range [{s['min']:.{prec}f}, {s['max']:.{prec}f}]")
    L.append(f"    change       {c['abs_change']:+.{prec}f} "
             f"({c['pct_change']:+.2f}%)   seed ranges "
             + ("OVERLAP" if c["ranges_overlap"] else "DO NOT OVERLAP"))
    if "welch" in c:
        w = c["welch"]
        L.append(f"    Welch        t={w['t']:+.3f}  p={w['p']:.4f}  "
                 f"Hedges g={w['hedges_g']:+.2f}  achieved power={c['achieved_power']:.2f}")
    if "paired_by_seed" in c:
        p = c["paired_by_seed"]
        pp = f"{p['paired_p']:.4f}" if p.get("paired_p") is not None else "n/a"
        ci = (f"[{p['ci95_low']:+.{prec}f}, {p['ci95_high']:+.{prec}f}]"
              if p.get("ci95_low") is not None else "n/a")
        L.append(f"    paired/seed  mean diff {p['mean_diff']:+.{prec}f}  p={pp}  "
                 f"95% CI {ci}  wins {p['wins_b']}/{p['n_pairs']}  "
                 f"sign-test p={p['sign_test_p']:.3f}")
    return "\n".join(L)


# -------------------------------------------------------------------- figures
BAR_W = 0.22      # thin marks; the bar is not the point, the seeds are
BAR_OFF = 0.13    # half the centre-to-centre gap between the two arms


def _seed_dots(ax, x, vals, col, spread=0.062):
    """Every seed as a visible point. At n=3 the raw points ARE the evidence.

    Drawn with a white face and a 2px surface ring so they stay legible where
    they overlap the bar fill of their own colour.
    """
    import numpy as np
    off = np.linspace(-spread, spread, len(vals)) if len(vals) > 1 else [0.0]
    for o, v in zip(off, vals):
        ax.plot(x + o, v, marker="o", ms=7, mfc="white", mec=col, mew=1.6,
                zorder=6, linestyle="none",
                path_effects=_ring())


def _ring():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2.6, foreground="white")]


def _no_data(ax, x):
    """Centred in the axes, never on the baseline where it collides with ticks."""
    ax.annotate("not yet run", (x, 0.5), xycoords=("data", "axes fraction"),
                ha="center", va="center", fontsize=10, color=C_MUTED,
                style="italic")


def fig_headline(data, out, plt, np):
    """Grouped bars faceted by METRIC, budget on the x, arm by colour.

    Budget is a position, not a hue: the same two hues mean the same two arms in
    every panel, which is what makes "did the gap move?" readable at a glance.
    """
    panels = [("train", "PRIMARY: training return\n(last-20-update mean, 128 envs)", 1),
              ("eval30", "30-episode eval\n(reward / step)", 3)]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))
    for ax, (key, title, prec) in zip(axes, panels):
        # Headroom for the per-budget verdict annotation, computed BEFORE
        # drawing so it can never ride up into the panel title.
        tops = [max(c[a]["mean"] + c[a]["sd_sample_ddof1"] for a in ARMS)
                for c in (data["budgets"][b[0]]["compare"].get(key) for b in BUDGETS)
                if c]
        ymax = max(tops) * 1.30 if tops else 1.0
        for bi, (blab, bsteps, *_ ) in enumerate(BUDGETS):
            c = data["budgets"][blab]["compare"].get(key)
            if not c:
                _no_data(ax, bi)
                continue
            for ai, arm in enumerate(ARMS):
                x = bi + (ai - 0.5) * 2 * BAR_OFF
                s = c[arm]
                # 2px surface gap between adjacent bars = white edge.
                ax.bar(x, s["mean"], BAR_W, color=ARM_COL[arm],
                       edgecolor="white", linewidth=2.0, zorder=3)
                if s["n"] > 1:
                    ax.errorbar(x, s["mean"], yerr=s["sd_sample_ddof1"],
                                fmt="none", ecolor=C_INK, elinewidth=1.4,
                                capsize=5, zorder=5)
                _seed_dots(ax, x, s["per_seed"], ARM_COL[arm])
                # Direct label on every bar: the orange is 2.73:1 on white, so
                # the validator's contrast WARN obliges visible labels. Anchored
                # near the BAR BASE in axes fraction, not offset from the bar
                # top -- the seed dots cluster around the top and an offset
                # label collides with the low one whenever a seed is an outlier.
                ax.annotate(f"{s['mean']:.{prec}f}", (x, 0.035),
                            xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", fontsize=10, fontweight="bold",
                            color="white", zorder=7)
            ax.annotate(
                f"{c['pct_change']:+.1f}%\n"
                + ("ranges overlap" if c["ranges_overlap"] else "ranges do not overlap"),
                (bi, max(c[a]["mean"] + c[a]["sd_sample_ddof1"] for a in ARMS)),
                textcoords="offset points", xytext=(0, 12), ha="center",
                fontsize=10, color=C_INK, fontweight="bold", zorder=7)
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels([f"{b[0]} steps\n({b[1]//(128*64)} updates)" for b in BUDGETS])
        ax.set_xlim(-0.5, len(BUDGETS) - 0.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("episode return")
    axes[1].set_ylabel("reward / step")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COL[a]) for a in ARMS]
    fig.legend(handles, [ARM_LABEL[a] for a in ARMS], loc="upper center",
               ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("WalkerWalk: does the state+RGB advantage survive 25.6x more training?",
                 fontsize=13.5, fontweight="bold", y=1.075)
    fig.text(0.5, -0.035,
             "Open circles are individual seeds (n=3). Bars are the mean, whiskers the SAMPLE s.d. (ddof=1). "
             "Both arms differ in exactly one config key, RGB_ACTOR.",
             ha="center", fontsize=9, color=C_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_curves(data, out, plt, np):
    """Learning curves, faceted by budget. Independent x-axes, both labelled in
    env steps -- never one panel with two x-scales."""
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(14.0, 5.6))
    for ax, (blab, bsteps, *_rest) in zip(np.atleast_1d(axes), BUDGETS):
        b = data["budgets"][blab]
        any_curve = False
        for arm in ARMS:
            curves = [np.asarray(c, float) for c in b["raw_curves"][arm] if c]
            if not curves:
                continue
            any_curve = True
            n = min(len(c) for c in curves)
            M = np.stack([c[:n] for c in curves])
            x = np.arange(n) * 128 * 64
            mu = M.mean(0)
            sd = M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(n)
            for c in M:
                ax.plot(x, c, color=ARM_COL[arm], lw=0.8, alpha=0.45, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=ARM_COL[arm], alpha=0.16,
                            lw=0, zorder=2)
            ax.plot(x, mu, color=ARM_COL[arm], lw=2.4, zorder=4,
                    label=f"{ARM_LABEL[arm]}  (n={M.shape[0]})")
            lo = max(0, n - LAST_N)
            ax.axvspan(x[lo], x[-1], color="#333333", alpha=0.09, zorder=1)
        ax.set_title(f"{blab} env steps  ({bsteps//(128*64)} updates x 128 envs x 64 steps)",
                     fontsize=11.5, fontweight="bold")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return")
        ax.grid(alpha=0.22, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if any_curve:
            ax.legend(frameon=False, fontsize=10, loc="lower right")
        else:
            ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color=C_MUTED)
    fig.suptitle("WalkerWalk learning curves at both budgets  "
                 "(shaded band at the right = the last-20-update window that defines the primary metric)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_effect(data, out, plt, np):
    """The actual headline: the state+RGB minus state-only gap, per budget, with
    the 95% CI of the paired-by-seed difference. One measure, one axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for ax, (key, title, unit) in zip(
            axes, [("train", "PRIMARY: training return", "episode return"),
                   ("eval30", "30-episode eval", "reward / step")]):
        for bi, (blab, *_r) in enumerate(BUDGETS):
            c = data["budgets"][blab]["compare"].get(key)
            if not c:
                _no_data(ax, bi)
                continue
            p = c.get("paired_by_seed") or {}
            d = c["abs_change"]
            col = C_SPRGB if d > 0 else C_STATE
            ax.bar(bi, d, 0.30, color=col, edgecolor="white", linewidth=2.0, zorder=3)
            if p.get("ci95_low") is not None:
                ax.errorbar(bi, p["mean_diff"],
                            yerr=[[p["mean_diff"] - p["ci95_low"]],
                                  [p["ci95_high"] - p["mean_diff"]]],
                            fmt="none", ecolor=C_INK, elinewidth=1.6, capsize=7,
                            zorder=5)
            if "diffs" in c:
                _seed_dots(ax, bi, c["diffs"], col, spread=0.085)
            # Label above the whisker (or below it for a negative effect) so it
            # never lands on the CI bar or a seed dot.
            hi = max([d] + ([p["ci95_high"]] if p.get("ci95_high") is not None else [])
                     + list(c.get("diffs", [])))
            lo = min([d] + ([p["ci95_low"]] if p.get("ci95_low") is not None else [])
                     + list(c.get("diffs", [])))
            ax.annotate(f"{d:+.3f}  ({c['pct_change']:+.1f}%)",
                        (bi, hi if d > 0 else lo), textcoords="offset points",
                        xytext=(0, 11 if d > 0 else -22), ha="center",
                        fontsize=10.5, fontweight="bold", color=C_INK, zorder=7)
        ax.axhline(0, color=C_INK, lw=1.2, zorder=4)
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels([b[0] + " steps" for b in BUDGETS])
        ax.set_xlim(-0.5, len(BUDGETS) - 0.5)
        ax.margins(y=0.22)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylabel(f"(state+RGB) - (state only),  {unit}")
        ax.grid(axis="y", alpha=0.22, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Does the camera's payoff grow, hold or vanish with budget?  "
                 "(bars above zero favour state+RGB; whiskers = 95% CI of the paired-by-seed difference)",
                 fontsize=12.0, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_camera(data, out, plt, np):
    """Per-seed, per-condition camera dependence -- NOT the binarised median.

    The 30% `actor_uses_pixels` bar was calibrated on RGB_PROPRIO: none actors,
    whose only input is the camera. An actor that also holds the state can ride
    the state through a corrupted frame and still be using the camera, so the
    boolean under-reads here and the tool itself prints a warning saying so.
    """
    conds = [c for c in CONDITIONS if c != "intact"]
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(14.5, 5.8), sharey=True)
    for ax, (blab, *_r) in zip(np.atleast_1d(axes), BUDGETS):
        drops = data["budgets"][blab]["camera"]
        w = 0.8 / len(conds)
        got = False
        for si, seed in enumerate(SEEDS):
            d = drops.get(str(seed)) or drops.get(seed)
            if not d:
                continue
            got = True
            for ci, cond in enumerate(conds):
                if cond not in d:
                    continue
                x = si + (ci - (len(conds) - 1) / 2) * w
                ax.bar(x, 100 * d[cond], w * 0.9, color=CORR_COL[cond],
                       edgecolor="white", linewidth=0.8, zorder=3,
                       label=cond if si == 0 else None)
        ax.axhline(30, ls="--", lw=1.5, color=C_INK, zorder=4)
        ax.annotate("30% `actor_uses_pixels` bar -- calibrated on pixels-ONLY\n"
                    "actors; it under-reads when the actor also has the state",
                    (len(SEEDS) - 0.55, 31), fontsize=8, color=C_INK,
                    ha="right", va="bottom")
        ax.axhline(0, lw=0.9, color=C_MUTED, zorder=4)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_title(f"{blab} env steps", fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not got:
            ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color=C_MUTED)
    np.atleast_1d(axes)[0].set_ylabel("performance drop vs intact (%)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=len(conds), frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Does the state+RGB actor still USE its camera at the full budget?",
                 fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", default="results/rgb/state_plus_rgb_full")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    H = _load_helpers(root)
    outdir = root / args.outdir
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    data = {"env": ENV, "seeds": list(SEEDS), "last_n_updates": LAST_N,
            "primary_metric": "training curve episode_return, last-20-update mean",
            "budgets": {}}

    for blab, bsteps, root_tr, root_30, suffix in BUDGETS:
        B = {"label": blab, "env_steps": bsteps,
             "updates": bsteps // (128 * 64),
             "roots": {"train_and_5ep": root_tr, "eval30": root_30},
             "arms": {}, "compare": {}, "camera": {}, "raw_curves": {a: [] for a in ARMS},
             "sources": {}}
        vals = {"train": {}, "eval5": {}, "eval30": {}}
        for arm in ARMS:
            an = arm_name(arm, suffix)
            tr, e5, e30, srcs = [], [], [], []
            for s in SEEDS:
                r_tr = read_run(root / root_tr, an, s)
                r_30 = read_run(root / root_30, an, s)
                tr.append(train_metric(r_tr))
                e5.append(eval_metric(r_tr))
                e30.append(eval_metric(r_30))
                B["raw_curves"][arm].append(
                    ((r_tr.get("curves") or {}).get("curves") or {}).get("episode_return"))
                srcs.append({"seed": s,
                             "train": r_tr.get("curves_path") or r_tr.get("ablation_path"),
                             "eval5": r_tr.get("ablation_path"),
                             "eval30": r_30.get("ablation_path"),
                             "n_eval30_episodes": n_episodes(r_30)})
                # camera conditions come from the state+RGB arm only
                if arm == "state_plus_rgb":
                    a30 = r_30.get("ablation") or r_tr.get("ablation")
                    if a30 and not a30.get("state_only"):
                        B["camera"][str(s)] = a30.get("performance_drop_fraction", {})
                        # All six conditions, not five drops: the intact score is
                        # the baseline the drops are relative to, and the absolute
                        # score of each corrupted condition is what makes a drop
                        # interpretable. Stored so the table stands alone.
                        B["camera"][f"{s}_absolute"] = {
                            c: a30["results"][c]["reward_per_step_mean"]
                            for c in CONDITIONS if c in a30.get("results", {})}
                        B["camera"][f"{s}_verdict_caveat"] = a30.get("verdict_caveat")
                        B["camera"][f"{s}_source"] = (r_30.get("ablation_path")
                                                      or r_tr.get("ablation_path"))
            B["arms"][arm] = {"config": f"configs/walker_walk_nesy_{an}.yaml",
                              "train": summarise(tr), "eval5": summarise(e5),
                              "eval30": summarise(e30)}
            B["sources"][arm] = srcs
            vals["train"][arm], vals["eval5"][arm], vals["eval30"][arm] = tr, e5, e30

        for key in ("train", "eval5", "eval30"):
            c = compare(vals[key]["state_matched"], vals[key]["state_plus_rgb"], H)
            if c and all(v is not None for v in vals[key]["state_matched"]) \
                    and all(v is not None for v in vals[key]["state_plus_rgb"]):
                c["diffs"] = [float(x - y) for x, y in
                              zip(vals[key]["state_plus_rgb"], vals[key]["state_matched"])]
            B["compare"][key] = c
        data["budgets"][blab] = B

    # --------------------------------------------------------------- printing
    print("=" * 92)
    print("WalkerWalk -- state-only vs state+RGB at TWO budgets")
    print("  one-variable pair: the two configs differ in exactly RGB_ACTOR")
    print("  s.d. is the SAMPLE s.d. (ddof=1) throughout")
    print("=" * 92)
    for blab, bsteps, *_ in BUDGETS:
        B = data["budgets"][blab]
        print(f"\n### {blab} env steps  ({B['updates']} updates x 128 envs x 64 steps)")
        for key, title, prec in [
                ("train", "PRIMARY -- training return, last-20-update mean (128 envs)", 3),
                ("eval30", "30-episode deterministic eval -- reward/step", 4),
                ("eval5", "5-episode eval (low power, 0.16; reported for continuity)", 4)]:
            print(f"\n  {title}")
            print(fmt_cmp(B["compare"].get(key), prec))
        if any(str(s) in B["camera"] for s in SEEDS):
            print("\n  camera dependence of the state+RGB actor, PER SEED, ALL SIX CONDITIONS")
            print("    reported as the per-condition table, NOT the binarised median rule:")
            print("    the 30% `actor_uses_pixels` bar was calibrated on RGB_PROPRIO: none")
            print("    actors and under-reads when the actor also holds the state (the")
            print("    ablation script itself prints a WARN saying so on every such run).")
            for s in SEEDS:
                d, ab = B["camera"].get(str(s)), B["camera"].get(f"{s}_absolute", {})
                if not d:
                    continue
                it = ab.get("intact")
                print(f"    seed {s}:  intact = {it:.4f} reward/step" if it is not None
                      else f"    seed {s}:")
                for c in CONDITIONS[1:]:
                    if c in d:
                        av = f"{ab[c]:.4f}" if c in ab else "  n/a "
                        print(f"        {c:<15s} {av}   drop {100*d[c]:+6.1f}%")

    # cross-budget movement of the effect
    print("\n" + "=" * 92)
    print("DID THE EFFECT MOVE WITH BUDGET?")
    for key, title, prec in [("train", "training return (primary)", 3),
                             ("eval30", "30-episode eval", 4)]:
        a = data["budgets"]["2.05M"]["compare"].get(key)
        b = data["budgets"]["52.4M"]["compare"].get(key)
        if not (a and b):
            print(f"  {title}: 52.4M not complete yet")
            continue
        print(f"  {title}: {a['pct_change']:+.2f}% at 2.05M  ->  "
              f"{b['pct_change']:+.2f}% at 52.4M   "
              f"(absolute {a['abs_change']:+.{prec}f} -> {b['abs_change']:+.{prec}f})")
        verdict = ("GREW" if b["pct_change"] > a["pct_change"] + 1 else
                   "REVERSED" if b["pct_change"] < 0 else
                   "SHRANK" if b["pct_change"] < a["pct_change"] - 1 else "HELD")
        print(f"      -> the effect {verdict}; "
              f"52.4M seed ranges "
              + ("OVERLAP" if b["ranges_overlap"] else "DO NOT OVERLAP"))
    print("=" * 92)

    jpath = outdir / "walker_full_budget_analysis.json"
    slim = json.loads(json.dumps(data))
    for b in slim["budgets"].values():
        b.pop("raw_curves", None)   # curves are large; they stay in their own files
    jpath.write_text(json.dumps(slim, indent=2))
    print(f"\nwrote {jpath}")

    if not args.no_figures:
        figdir.mkdir(parents=True, exist_ok=True)
        for fn, name in [(fig_headline, "fig1_budget_headline.png"),
                         (fig_curves, "fig2_budget_learning_curves.png"),
                         (fig_effect, "fig3_effect_vs_budget.png"),
                         (fig_camera, "fig4_camera_use_both_budgets.png")]:
            p = figdir / name
            fn(data, p, plt, np)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
