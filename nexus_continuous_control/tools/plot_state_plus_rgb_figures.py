"""Figures for the state-only vs state+RGB matched-budget campaign.

Reads the committed run JSON only (no GPU, no retraining) and writes the figure
set plus a README describing what each one does and does not claim.

Honesty rules are enforced here rather than left to the caller:
  * every panel states its METRIC, its BUDGET and its seed count;
  * cartpole (upright fraction) and walker/cheetah (reward per step) never share
    an axis -- each environment gets its own panel and its own y-label;
  * per-SEED points are always drawn, never only a mean. Camera use in
    particular is seed-dependent (walker returns 4.8% / 65.6% / 42.9% median
    pixel drop on three seeds of the identical config), so a mean would describe
    no run that actually happened;
  * a run the rescore guard marks INCONCLUSIVE is drawn grey and labelled, never
    given a verdict colour;
  * the 52.4M-step state baseline is plotted only if a run artifact exists; it
    is otherwise omitted with a visible note, never estimated.

    python tools/plot_state_plus_rgb_figures.py --figure all
    python tools/plot_state_plus_rgb_figures.py --figure curves --outdir /tmp/f
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# House palette (seaborn "deep"), matching tools/plot_rgb_ablation_comparison.py.
C_STATE = "#4C72B0"   # state-only  (baseline)
C_SPRGB = "#DD8452"   # state + RGB (extension)
C_PIXEL = "#937860"   # pixels-only variants
C_SEES = "#55A868"
C_BLIND = "#C44E52"
C_INCONC = "#8C8C8C"

ENVS = ["cartpole", "walker", "cheetah"]
ENV_TITLE = {"cartpole": "CartpoleBalance", "walker": "WalkerWalk",
             "cheetah": "CheetahRun"}
ENVCOL = {"cartpole": "#4C72B0", "walker": "#DD8452", "cheetah": "#55A868"}

NEW_ROOT = "results/rgb/state_plus_rgb"
OLD_ROOT = "results/rgb/ablation"

NEW_ARMS = {
    "state_matched": ("state only (baseline)", C_STATE),
    "state_plus_rgb": ("state + RGB (extension)", C_SPRGB),
}

OLD_VARIANTS = {
    "cartpole": [
        ("RGB pixels-only", ["nesy_blind", "nesy_blind_seed1", "nesy_blind_seed2"]),
        ("RGB aux-fix", ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
        ("RGB no-aux MDI4", ["nesy_noaux_mdi4_seed0", "nesy_noaux_mdi4_seed1",
                             "nesy_noaux_mdi4_seed2"]),
        ("RGB shared", ["nesy_shared_seed0"]),
        ("RGB shared no-aux", ["nesy_shared_noaux_mdi4_seed0"]),
        ("RGB shared+meta-z", ["nesy_shared_metaz_noaux_seed0",
                               "nesy_shared_metaz_noaux_seed1",
                               "nesy_shared_metaz_noaux_seed2"]),
    ],
    "walker": [
        ("RGB pixels-only", ["nesy_blind"]),
        ("RGB aux-fix", ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
        ("RGB no-aux MDI4", ["nesy_noaux_mdi4_seed0"]),
        ("RGB shared", ["nesy_shared_seed0"]),
        ("RGB shared no-aux", ["nesy_shared_noaux_mdi4_seed0"]),
        ("RGB shared+meta-z", ["nesy_shared_metaz_noaux_seed0"]),
    ],
    "cheetah": [("RGB pixels-only", ["nesy_seed0", "nesy_seed1", "nesy_seed2"])],
}

LADDER_PIXEL = {
    "cartpole": ["nesy_blind", "nesy_blind_seed1", "nesy_blind_seed2"],
    "walker": ["nesy_blind"],
    "cheetah": ["nesy_seed0", "nesy_seed1", "nesy_seed2"],
}
KNOWN_SEEING = {
    "cartpole": ("RGB aux-fix (pixels-only)",
                 ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
    "walker": ("RGB aux-fix (pixels-only)",
               ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
    "cheetah": ("RGB pixels-only", ["nesy_seed0", "nesy_seed1", "nesy_seed2"]),
}

MATCHED_BUDGET = "2.05M env steps (250 updates x 128 envs x 64 steps)"
SMOOTH_W = 11


# --------------------------------------------------------------------------- io
def load(root: Path, env: str, tag: str):
    p = Path(root) / env / tag / "pixel_ablation.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["_dir"] = p.parent
    return d


def load_curve(root: Path, env: str, tag: str, key="episode_return"):
    p = Path(root) / env / tag / "training_curves.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["curves"].get(key)


def metric_info(d):
    """(key, human label, value, per-episode std) for a run's intact condition."""
    i = d["results"]["intact"]
    if "upright_fraction_mean" in i:
        return ("upright_fraction_mean", "upright fraction",
                i["upright_fraction_mean"], i["upright_fraction_std"])
    return ("reward_per_step_mean", "reward / step",
            i["reward_per_step_mean"], i["reward_per_step_std"])


def aggregate(runs):
    """Mean plus an error bar whose MEANING is returned with it."""
    import numpy as np
    vals = [metric_info(d)[2] for d in runs]
    if len(runs) == 1:
        return float(vals[0]), float(metric_info(runs[0])[3]), 1, "5 eval episodes"
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(runs), \
        f"{len(runs)} seeds"


def collect(root, env, tags):
    return [d for d in (load(root, env, t) for t in tags) if d is not None]


def seed_tags(arm, seeds=(0, 1, 2)):
    return [f"{arm}_seed{s}" for s in seeds]


def smooth(y, w=SMOOTH_W):
    """Centred rolling mean, edge-padded so the length (and the x-axis) is kept."""
    import numpy as np
    y = np.asarray(y, float)
    if w <= 1 or y.size < w:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(w) / w,
                       mode="valid")[:y.size]


def env_metric_label(new_root, env):
    r = collect(new_root, env, seed_tags("state_matched")) or \
        collect(new_root, env, seed_tags("state_plus_rgb"))
    return metric_info(r[0])[1] if r else "score"


# ------------------------------------------------------- fig 1: LEARNING CURVES
def fig_curves(new_root, out, plt, np):
    """THE headline. Baseline vs extension as learning curves, seeds banded."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    found = False
    for ax, env in zip(axes, ENVS):
        ns, finals = {}, []
        for arm, (lab, col) in NEW_ARMS.items():
            curves = [c for c in (load_curve(new_root, env, t)
                                  for t in seed_tags(arm)) if c]
            if not curves:
                continue
            found = True
            n = min(len(c) for c in curves)
            raw = np.stack([np.asarray(c[:n], float) for c in curves])
            sm = np.stack([smooth(r) for r in raw])
            mu, sd = sm.mean(0), sm.std(0)
            x = np.arange(n)
            # Raw mean faintly behind, so heavy smoothing cannot hide instability.
            ax.plot(x, raw.mean(0), color=col, lw=0.8, alpha=0.28, zorder=2)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.20, lw=0,
                            zorder=3)
            ax.plot(x, mu, color=col, lw=2.4, label=f"{lab}  (n={len(curves)})",
                    zorder=4)
            ax.plot([x[-1]], [mu[-1]], marker="o", ms=7, color=col,
                    mec="#222222", mew=1.2, zorder=5)
            finals.append((x[-1], float(mu[-1]), col))
            ns[arm] = len(curves)
        # Annotate the end points AFTER both arms are known, pushing the higher
        # value up and the lower one down. Staggering by arm index instead would
        # push them together whenever the second arm finishes higher (WalkerWalk).
        for xf, yf, col in finals:
            dy = 0
            if len(finals) == 2:
                other = [v for _x, v, _c in finals if v != yf]
                if other:
                    dy = 9 if yf >= other[0] else -9
            ax.annotate(f"{yf:.1f}", (xf, yf), textcoords="offset points",
                        xytext=(9, dy), ha="left", va="center", fontsize=10,
                        fontweight="bold", color=col)
        if not ns:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.set_title(f"{ENV_TITLE[env]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("training update      (250 updates = 2.048M env steps)",
                      fontsize=9.5)
        ax.set_ylabel("episode return during training", fontsize=10)
        ax.set_xlim(0, 260)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    if not found:
        raise SystemExit("curves: nothing found")
    fig.suptitle("Baseline vs extension: does adding a camera to a skill actor "
                 "that ALREADY HAS the state help?\n"
                 f"nesy meta, matched budget ({MATCHED_BUDGET}); each pair of arms "
                 "differs in exactly ONE config key (RGB_ACTOR)",
                 fontsize=13.5, y=1.04)
    fig.text(0.5, -0.05,
             f"Bold line = mean over seeds of each seed's curve after a centred "
             f"rolling mean of {SMOOTH_W} updates. Shaded band = +/-1 s.d. ACROSS "
             "SEEDS of those smoothed curves (it is seed spread, not update "
             "noise).\nThe faint line behind is the unsmoothed mean, drawn so that "
             "genuine late-training instability stays visible rather than being "
             "smoothed away. The ringed marker is the final update, which is the "
             "checkpoint the evaluation scores.\nCAUTION: this is the TRAINER's "
             "episode return, collected with exploration noise across 128 envs. It "
             "is NOT the deterministic evaluation metric used in the bar and "
             "scatter figures, and the two can disagree.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------- fig 2: camera use (per seed)
def fig_camera(new_root, old_root, out, plt, np):
    data = []
    for env in ENVS:
        for tag in seed_tags("state_plus_rgb"):
            d = load(new_root, env, tag)
            if d is None or d.get("pixel_drop_median") is None:
                continue
            data.append((f"{ENV_TITLE[env]}   state+RGB, seed "
                         f"{tag.rsplit('seed', 1)[1]}",
                         100 * float(d["pixel_drop_median"]), 0.0, [],
                         bool(d.get("inconclusive")), True))
    for env in ENVS:
        lab, tags = KNOWN_SEEING[env]
        runs = collect(old_root, env, tags)
        vals = [r["pixel_drop_median"] for r in runs
                if r.get("pixel_drop_median") is not None]
        if not vals:
            continue
        data.append((f"{ENV_TITLE[env]}   {lab}", 100 * float(np.mean(vals)),
                     100 * float(np.std(vals)) if len(vals) > 1 else 0.0,
                     [100 * v for v in vals],
                     any(r.get("inconclusive") for r in runs), False))
    if not data:
        raise SystemExit("camera figure: nothing to plot")

    fig, ax = plt.subplots(figsize=(13.5, 0.62 * len(data) + 4.2))
    ypos = np.arange(len(data))[::-1]
    for y, (lab, v, e, seeds, inc, is_new) in zip(ypos, data):
        col = C_INCONC if inc else (C_SEES if v > 30 else C_BLIND)
        ax.barh(y, v, xerr=e if e else None, capsize=5, color=col, height=0.6,
                edgecolor="#333333" if is_new else "white",
                linewidth=1.8 if is_new else 0.8)
        for sv in seeds:
            ax.plot(sv, y, marker="o", ms=5, mfc="white", mec="#333333", mew=1.2,
                    ls="none", zorder=6)
        tag = "INCONCLUSIVE" if inc else ("SEES" if v > 30 else "BLIND")
        extra = "" if is_new else f"   (n={len(seeds)}, dots = seeds)"
        ax.text(108, y, f"{v:+.1f}%   {tag}{extra}", va="center", ha="left",
                fontsize=10.5, fontweight="bold" if is_new else "normal",
                color=col if not inc else "#555555")
    ax.axvline(30, ls="--", lw=2, color="#333333")
    ax.text(30.8, len(data) - 0.5, "verdict threshold 30%", fontsize=9.5,
            color="#333333", ha="left", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB"))
    ax.set_ylim(-0.7, len(data) - 0.15)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in data], fontsize=9.5)
    ax.set_xlabel("median performance lost when the actor's camera is corrupted\n"
                  "(median over frozen-first / wrong-timestep / blank-image, as % "
                  "of that run's own intact score)", fontsize=10)
    ax.set_xlim(min(-8, min(r[1] for r in data) - 8), 178)
    ax.set_xticks(list(range(0, 101, 20)))
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("Does the actor use the camera when it already has the state?\n"
                 "ONE ROW PER SEED -- the verdict is not consistent across seeds\n"
                 f"bold = this campaign's state+RGB arms; thin = committed "
                 f"pixels-only reference arms. All at {MATCHED_BUDGET}",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.02,
             "Higher = the actor depends on its camera; near 0% means blanking the "
             "image changes nothing, i.e. that actor ignored the camera and rode "
             "the state.\nSeeds are shown individually and deliberately NOT "
             "averaged: on WalkerWalk three seeds of the identical config give "
             "4.8%, 65.6% and 42.9%, so a mean would describe no run that "
             "happened.\nThe reference arms have NO state input, so they must use "
             "pixels or fail; they calibrate what a genuinely seeing actor looks "
             "like under this identical protocol.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------- fig 3: camera use vs performance
def fig_payoff(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    any_pt = False
    for ax, env in zip(axes, ENVS):
        base = collect(new_root, env, seed_tags("state_matched"))
        ylab = env_metric_label(new_root, env)
        if base:
            bm, bs, bn, bkind = aggregate(base)
            ax.axhline(bm, color=C_STATE, lw=2.2, zorder=2)
            if bn > 1:
                ax.axhspan(bm - bs, bm + bs, color=C_STATE, alpha=0.15, zorder=1)
            for r in base:      # the baseline's own seeds, so its spread is visible
                ax.plot(-4, metric_info(r)[2], marker="_", ms=13, color=C_STATE,
                        mew=2.5, ls="none", zorder=3)
            ax.text(0.985, bm, f" state-only mean {bm:.3f} (n={bn}) ",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=9, color=C_STATE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_STATE))
        for tag in seed_tags("state_plus_rgb"):
            d = load(new_root, env, tag)
            if d is None or d.get("pixel_drop_median") is None:
                continue
            any_pt = True
            x = 100 * float(d["pixel_drop_median"])
            y = metric_info(d)[2]
            sees = bool(d.get("actor_uses_pixels")) and not d.get("inconclusive")
            ax.scatter(x, y, s=230, color=C_SEES if sees else C_BLIND,
                       edgecolor="#222222", linewidth=1.4, zorder=5)
            ax.annotate(f"s{tag.rsplit('seed', 1)[1]}", (x, y),
                        textcoords="offset points", xytext=(0, -3), ha="center",
                        va="center", fontsize=8.5, color="white",
                        fontweight="bold", zorder=6)
        ax.axvline(30, ls="--", lw=1.8, color="#333333", zorder=1)
        ax.set_xlim(-10, 105)
        ax.set_xlabel("that seed's camera dependence (%)\n"
                      "dashed line = 30% verdict threshold", fontsize=9.5)
        ax.set_ylabel(f"{ylab}  (eval intact)", fontsize=10)
        ax.set_title(ENV_TITLE[env], fontsize=13, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    if not any_pt:
        raise SystemExit("payoff: nothing to plot")
    import matplotlib.lines as mlines
    fig.legend(handles=[
        mlines.Line2D([], [], marker="o", ls="none", ms=11, mfc=C_SEES,
                      mec="#222222", label="state+RGB seed whose actor USED the camera"),
        mlines.Line2D([], [], marker="o", ls="none", ms=11, mfc=C_BLIND,
                      mec="#222222", label="state+RGB seed that IGNORED the camera"),
        mlines.Line2D([], [], color=C_STATE, lw=2.5,
                      label="state-only mean (band = +/-1 s.d. across its seeds)"),
        mlines.Line2D([], [], marker="_", ls="none", ms=13, mec=C_STATE, mew=2.5,
                      label="individual state-only seeds"),
    ], loc="upper center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.0),
        framealpha=0.95)
    fig.suptitle("Does USING the camera pay off?   one point per state+RGB seed, "
                 f"against the state-only control of the same environment\n"
                 f"nesy meta, {MATCHED_BUDGET}", fontsize=13, y=1.10)
    fig.text(0.5, -0.05,
             "A point ABOVE the blue line beat the state-only baseline of its own "
             "environment; below it, it lost. Points RIGHT of the dashed line are "
             "the seeds whose actor genuinely depended on its camera.\n"
             "Units differ between panels (upright fraction vs reward per step), so "
             "heights are not comparable across panels -- only each point against "
             "its own panel's blue line.\nThis shows ASSOCIATION over a handful of "
             "seeds, not a causal effect of seeing on performance, and the "
             "within-environment ordering is not monotone.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------- fig 4: pixel sensitivity during training
def fig_sensitivity(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    found = False
    for ax, env in zip(axes, ENVS):
        for tag in seed_tags("state_plus_rgb"):
            c = load_curve(new_root, env, tag, "pixel_sensitivity")
            d = load(new_root, env, tag)
            if not c:
                continue
            found = True
            seed = tag.rsplit("seed", 1)[1]
            pd_ = d.get("pixel_drop_median") if d else None
            sees = bool(d and d.get("actor_uses_pixels")
                        and not d.get("inconclusive"))
            col = C_SEES if sees else C_BLIND
            lab = f"seed {seed}"
            if pd_ is not None:
                lab += f" -- {'USES' if sees else 'ignores'} ({100 * pd_:+.0f}%)"
            y = np.asarray(c, float)
            ax.plot(np.arange(y.size), y, color=col, lw=0.7, alpha=0.25)
            ax.plot(np.arange(y.size), smooth(y), color=col, lw=2.2, label=lab)
        if not ax.get_legend_handles_labels()[0]:
            ax.set_title(f"{ENV_TITLE[env]}\n(no pixel-sensitivity curve)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.set_yscale("log")
        ax.axhline(0.01, ls="--", lw=1.2, color="grey")
        ax.text(2, 0.0105, "0.01 = the blind/seeing rule of thumb", fontsize=8,
                color="grey", va="bottom")
        ax.set_title(ENV_TITLE[env], fontsize=13, fontweight="bold")
        ax.set_xlabel("training update      (250 = 2.048M env steps)", fontsize=9.5)
        ax.set_ylabel("pixel sensitivity  (log scale)\nhow much the action moves "
                      "when the image changes", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    if not found:
        raise SystemExit("sensitivity: no curves found")
    fig.suptitle("Does the actor's camera reliance DEVELOP during training?\n"
                 "state+RGB arms only, one line per seed "
                 f"({SMOOTH_W}-update rolling mean, raw faint behind)",
                 fontsize=13, y=1.03)
    fig.text(0.5, -0.05,
             "`train/rgb/pixel_sensitivity` is the trainer's own online probe: how "
             "far the actor's output moves when its pixel input changes. It is an "
             "OPEN-LOOP quantity measured during training, and it is NOT the same "
             "number as the closed-loop pixel drop\nquoted in the legend -- that "
             "one comes from the end-of-training ablation. They are shown together "
             "because the question is whether the end verdict was already visible "
             "in training.\nColour encodes the FINAL verdict, so a red line that "
             "climbs is an actor that developed some sensitivity and still did not "
             "depend on the camera by the end.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------- fig 5: summary bars
def fig_bars(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
    any_data = False
    for ax, env in zip(axes, ENVS):
        bars, labels, colors, errs, ns, seedvals = [], [], [], [], [], []
        ylab = None
        for arm, (lab, col) in NEW_ARMS.items():
            runs = collect(new_root, env, seed_tags(arm))
            if not runs:
                continue
            ylab = metric_info(runs[0])[1]
            m, e, n, _k = aggregate(runs)
            bars.append(m); errs.append(e); labels.append(lab.replace(" (", "\n("))
            colors.append(col); ns.append(n)
            seedvals.append([metric_info(r)[2] for r in runs])
        if not bars:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        any_data = True
        x = np.arange(len(bars))
        b = ax.bar(x, bars, 0.55, yerr=errs, capsize=6, color=colors,
                   edgecolor="white", linewidth=1.2)
        top = max(max(v + e for v, e in zip(bars, errs)),
                  max((max(sv) for sv in seedvals if sv), default=0.0))
        ax.set_ylim(0, top * 1.55)
        for i, sv in enumerate(seedvals):
            for j, val in enumerate(sv):
                # Jitter horizontally by seed: cartpole's state-only seeds are all
                # exactly 1.0000, so stacked dots and labels would overprint.
                xs = i + 0.16 + 0.075 * j
                ax.plot(xs, val, marker="o", ms=7, mfc="white",
                        mec="#222222", mew=1.5, ls="none", zorder=6)
                ax.annotate(f"s{j}", (xs, val), textcoords="offset points",
                            xytext=(0, 9), ha="center", va="bottom", fontsize=7.5,
                            color="#333333", zorder=6)
        for i, (v, e, n) in enumerate(zip(bars, errs, ns)):
            ax.text(i, v + e + top * 0.03, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
            ax.text(i, v + e + top * 0.125, f"n={n} seed{'s' if n > 1 else ''}",
                    ha="center", va="bottom", fontsize=8, color="#555555")
        if len(bars) == 2:
            w = int(np.argmax(bars))
            d = bars[w] - bars[1 - w]
            # Overlapping seed ranges are called out, not hidden behind an arrow.
            overlap = (min(seedvals[0]) <= max(seedvals[1])
                       and min(seedvals[1]) <= max(seedvals[0])
                       and len(seedvals[0]) > 1 and len(seedvals[1]) > 1)
            wl = "state only" if w == 0 else "state + RGB"
            msg = f"{wl} higher by {d:.3f}"
            if len(seedvals[0]) > 1 and len(seedvals[1]) > 1:
                import warnings
                from scipy import stats
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _t, _p = stats.ttest_ind(seedvals[1], seedvals[0],
                                             equal_var=False)
                msg += f"\nWelch p={_p:.2f} at n=3"
            if overlap:
                msg += "  --  seed ranges OVERLAP"
            # Axes coords: a data-coordinate y collided with the tick labels when
            # the panel's range was small (CheetahRun).
            ax.text(0.5, 0.995, msg, transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, fontweight="bold",
                    color=colors[w] if not overlap else "#555555")
            if not overlap:
                b[w].set_edgecolor("#333333"); b[w].set_linewidth(2.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(f"{ylab}  (eval, higher is better)", fontsize=10)
        ax.set_title(ENV_TITLE[env], fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    if not any_data:
        raise SystemExit("bars: no runs found")
    fig.suptitle("SECONDARY summary: final evaluation score, state-only vs "
                 f"state+RGB\nnesy meta, {MATCHED_BUDGET}; each pair differs in ONE "
                 "config key (RGB_ACTOR)", fontsize=13, y=1.04)
    fig.text(0.5, -0.05,
             "Metric per panel: CartpoleBalance = upright fraction (bounded 0-1); "
             "WalkerWalk / CheetahRun = task reward per step. Different metrics, so "
             "the panels share no axis.\nRinged dots are the INDIVIDUAL SEEDS. Error "
             "bars are +/-1 s.d. across seeds where several were run, otherwise "
             "+/-1 s.d. across the 5 evaluation episodes of the single seed; every "
             "bar is annotated with its n.\nWhere the two arms' seed ranges overlap "
             "the difference is labelled as such and no winner is highlighted -- at "
             "n=3 an overlapping difference is not a result.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------ fig 6: all variants
def fig_full(new_root, old_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    for ax, env in zip(axes, ENVS):
        entries = []
        for arm, (lab, col) in NEW_ARMS.items():
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab, m, e, n, col, False))
        for lab, tags in OLD_VARIANTS.get(env, []):
            runs = collect(old_root, env, tags)
            if not runs:
                continue
            m, e, n, _ = aggregate(runs)
            inc = any(r.get("inconclusive") for r in runs)
            entries.append((lab, m, e, n, C_INCONC if inc else C_PIXEL, inc))
        if not entries:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ylab = env_metric_label(new_root, env)
        entries.sort(key=lambda t: -t[1])
        x = np.arange(len(entries))
        vals = [e[1] for e in entries]; errs = [e[2] for e in entries]
        ax.bar(x, vals, 0.66, yerr=errs, capsize=4,
               color=[e[4] for e in entries], edgecolor="white", linewidth=1.0)
        top = max(v + er for v, er in zip(vals, errs)) or 1.0
        ax.set_ylim(0, top * 1.26)
        for i, e in enumerate(entries):
            ax.text(i, e[1] + e[2] + top * 0.02, f"{e[1]:.3f}", ha="center",
                    va="bottom", fontsize=8.5, fontweight="bold")
            ax.text(i, e[1] + e[2] + top * 0.075, f"n={e[3]}", ha="center",
                    va="bottom", fontsize=7, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] + ("\n(INCONCLUSIVE)" if e[5] else "")
                            for e in entries], rotation=32, ha="right", fontsize=8.5)
        ax.set_ylabel(f"{ylab}  (eval intact)", fontsize=10)
        ax.set_title(f"{ENV_TITLE[env]}   -- metric: {ylab}", fontsize=12,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    import matplotlib.patches as mpatches
    fig.legend(handles=[
        mpatches.Patch(color=C_STATE, label="state only (baseline)"),
        mpatches.Patch(color=C_SPRGB, label="state + RGB (extension)"),
        mpatches.Patch(color=C_PIXEL, label="pixels-only variants (committed)"),
        mpatches.Patch(color=C_INCONC, label="inconclusive under the rescore guard"),
    ], loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.005),
        framealpha=0.95)
    fig.suptitle("Every approach we have, on one axis per environment\n"
                 f"All bars: {MATCHED_BUDGET}, nesy meta, identical evaluation "
                 "protocol (5 deterministic episodes x 250 steps)",
                 fontsize=13, y=1.10)
    fig.text(0.5, -0.02,
             "Units are consistent WITHIN each panel and stated in each panel "
             "title; they differ BETWEEN panels, so bar heights must not be "
             "compared across panels.\nOnly the state-only / state+RGB pair is a "
             "controlled one-variable contrast -- the pixels-only variants differ "
             "from each other in more than one config key.\nn = seeds; error bars "
             "are +/-1 s.d. across seeds when n>1, else across the 5 eval episodes "
             "of the single seed.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------- fig 7: budget gap
def fig_budget(new_root, old_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, env in zip(axes, ENVS):
        entries = []
        for arm, lab, col in (("state_matched", "state only\n@ 2.05M", C_STATE),
                              ("state_plus_rgb", "state+RGB\n@ 2.05M", C_SPRGB)):
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab, m, e, n, col))
        runs = collect(old_root, env, LADDER_PIXEL[env])
        if runs:
            m, e, n, _ = aggregate(runs)
            entries.append(("pixels-only\n@ 2.05M", m, e, n, C_PIXEL))
        if not entries:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ylab = env_metric_label(new_root, env)
        x = np.arange(len(entries))
        vals = [e[1] for e in entries]; errs = [e[2] for e in entries]
        ax.bar(x, vals, 0.6, yerr=errs, capsize=5, color=[e[4] for e in entries],
               edgecolor="#333333", linewidth=1.0)
        top = max(v + er for v, er in zip(vals, errs)) or 1.0
        ax.set_ylim(0, top * 1.28)
        for i, e in enumerate(entries):
            ax.text(i, e[1] + e[2] + top * 0.02, f"{e[1]:.3f}", ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold")
            ax.text(i, e[1] + e[2] + top * 0.085, f"n={e[3]}", ha="center",
                    va="bottom", fontsize=7.5, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries], fontsize=9)
        ax.set_ylabel(f"{ylab}  (eval intact)", fontsize=10)
        ax.set_title(ENV_TITLE[env], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle("The budget caveat, closed\n"
                 "Every state baseline previously cited was trained at 52.4M steps "
                 "while the RGB arms got 2.05M -- a 25x gap. These are all 2.05M.",
                 fontsize=12.5, y=1.05)
    fig.text(0.5, -0.09,
             "EVERY bar here is at the 2.05M matched budget, so these comparisons "
             "are fair. No 52.4M bar is drawn: the 52.4M state baselines quoted in "
             "earlier prose have NO run artifact in `results/`\n(it contains only "
             "`results/rgb`), so that number cannot be verified and is OMITTED "
             "rather than estimated. The matched-budget state arms above replace it "
             "as the baseline to cite.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FFF4E5", ec="#DDAA66"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------- README
README_HEAD = """\
# state-only vs state+RGB at matched budget -- figures

Generated by `tools/plot_state_plus_rgb_figures.py` from the committed run JSON:

    python tools/plot_state_plus_rgb_figures.py --figure all

## The question

The NEXUS paper suggests using RGB inputs for the skill agents. Earlier work here
tested RGB *replacing* the state (`RGB_PROPRIO: none`). This campaign tests RGB
*added to* the state (`RGB_PROPRIO: full`) against a state-only control at the
same environment-step budget -- the first honest baseline-vs-extension comparison
in this project.

## How the comparison is kept honest

Both arms of every pair are generated from ONE base config by
`tools/gen_state_plus_rgb_configs.py`, which asserts the resolved configs differ
in exactly one key, `RGB_ACTOR`. That key gates only whether the skill actors
have a camera pathway; the ENVIRONMENT keeps `USE_RGB: true` in both arms and
still renders. This matters, because `USE_RGB` also switches the task: MuJoCo
Playground's CartpoleBalance keys `ctrl_dt`, `episode_length`, the reward
function and the termination rule on `vision`, and the vec wrapper changes the
actor's state vector. A `USE_RGB: false` baseline would differ in the
environment, the reward, the horizon and the state representation too.

Both arms are also scored by the same code: `rgb_pixel_ablation.py` runs the same
`rollout()`, the same metric keys and the same scoring loop for both. The
state-only arm scores the `intact` condition only, because pixel corruptions are
undefined without a pixel input.

## The figures

| file | shows | does NOT claim |
|---|---|---|
| `fig1_headline_learning_curves.png` | **PRIMARY.** Episode return over training, state-only vs state+RGB, one panel per env, mean over seeds with a +/-1 s.d. seed band. | It is the TRAINER's metric (exploration noise, 128 envs), not the deterministic eval. Curves show learning dynamics; the eval figures give the score. |
| `fig2_camera_use_per_seed.png` | Camera dependence, one row per seed, against the 30% verdict threshold and against pixels-only arms known to see. | Measures the ACTOR's pixel dependence only, not the hierarchy's. A low value does not show the camera *could not* help, only that this actor did not use it. |
| `fig3_camera_use_vs_performance.png` | Each state+RGB seed's camera dependence against its score, with the state-only control drawn as a reference line per env. | Association across a handful of seeds, NOT a causal effect of seeing on performance. The within-env ordering is not monotone. |
| `fig4_pixel_sensitivity_curves.png` | The trainer's online pixel-sensitivity probe over training, per seed, for the state+RGB arms. | Open-loop sensitivity during training is not the same quantity as the closed-loop pixel drop; they are shown together only to ask whether the end verdict was visible earlier. |
| `fig5_summary_bars.png` | Secondary summary of final eval scores with per-seed dots. | Where seed ranges overlap it says so and picks no winner. |
| `fig6_all_variants.png` | Every committed variant plus the two new arms, per env. | Only the state-only / state+RGB pair is a controlled contrast; the other variants differ in several keys. Units differ between panels. |
| `fig7_budget_gap.png` | That every bar here is at the matched 2.05M budget. | The 52.4M state baseline is OMITTED, not estimated: `results/` holds only `results/rgb`, so it has no verifiable artifact. |

## Findings that must not be summarised away

* **Camera use is environment- and seed-dependent, not uniform.** Three seeds of
  the identical WalkerWalk config give median pixel drops of 4.8%, 65.6% and
  42.9% -- one ignores the camera and two depend on it. Any statement that "the
  actor ignores the camera" as a blanket claim is false. Read the per-seed table
  below, never a mean over those numbers.
* **`walker/state_matched_seed0` has a contaminated evaluation.** Its five
  episodes scored [0.642, 0.788, 0.833, 0.763, 0.055]; one collapsed episode
  drags the mean to 0.6161 where the other four average 0.757. Carry this caveat
  wherever that number appears.
* **Differences at n=3 are mostly not separable.** Where the two arms' seed
  ranges overlap, the figures say so and highlight no winner. The table below
  reports Welch's t-test, and at n=3 per arm it has very little power -- treat
  it as a description of overlap, not as evidence of an effect.

## Reading rules

* Metrics differ by environment: CartpoleBalance uses the bounded upright
  fraction (0-1), computed geometrically from `qpos` and so independent of the
  env's reward function; WalkerWalk and CheetahRun use task reward per step.
* `n` is the number of SEEDS. Error bars are +/-1 s.d. across seeds when n > 1,
  and +/-1 s.d. across the 5 evaluation episodes of the single seed when n = 1.
* The evaluation scores the FINAL weights, so a training curve that collapses in
  its last updates produces an eval number that reflects the collapse.
* `const_action` is not a pixel-specific control for a `RGB_PROPRIO: full` actor:
  it removes the actor's state-driven variation too. Only the
  frozen/replay/blank conditions isolate the pixels.
* CartpoleBalance's base config uses `NUM_MINIBATCHES: 8` where walker and
  cheetah use 64, so cartpole takes 8x fewer gradient steps per env step. It is
  identical in both arms, so it cannot confound state vs state+RGB, but cartpole
  numbers are not comparable across environments.
"""


def write_readme(new_root, old_root, outdir):
    import numpy as np
    from scipy import stats

    lines = [README_HEAD, "\n## Per-seed results (the numbers to cite)\n",
             "| env | arm | seed | intact | train return | median pixel drop | "
             "camera verdict |", "|---|---|---|---|---|---|---|"]
    for env in ENVS:
        for arm, (lab, _c) in NEW_ARMS.items():
            for tag in seed_tags(arm):
                d = load(new_root, env, tag)
                if d is None:
                    continue
                _k, _yl, v, sd = metric_info(d)
                tr = d.get("final_train_return")
                trs = "n/a" if tr is None else f"{tr:.1f}"
                if d.get("state_only"):
                    drop, verdict = "n/a (no pixels)", "n/a"
                else:
                    drop = f"{100 * d['pixel_drop_median']:+.1f}%"
                    verdict = ("INCONCLUSIVE" if d.get("inconclusive")
                               else ("**SEES**" if d.get("actor_uses_pixels")
                                     else "ignores"))
                lines.append(f"| {ENV_TITLE[env]} | {lab} | "
                             f"{tag.rsplit('seed', 1)[1]} | {v:.4f} +/- {sd:.4f} | "
                             f"{trs} | {drop} | {verdict} |")

    lines += ["\n## Per-environment summary\n",
              "| env | metric | state only | state + RGB | difference | "
              "seed ranges overlap? | Welch t-test | camera use |",
              "|---|---|---|---|---|---|---|---|"]
    for env in ENVS:
        a = collect(new_root, env, seed_tags("state_matched"))
        b = collect(new_root, env, seed_tags("state_plus_rgb"))
        if not (a and b):
            continue
        av = [metric_info(r)[2] for r in a]
        bv = [metric_info(r)[2] for r in b]
        am, bm = float(np.mean(av)), float(np.mean(bv))
        asd, bsd = float(np.std(av)), float(np.std(bv))
        overlap = (min(av) <= max(bv) and min(bv) <= max(av))
        if len(av) > 1 and len(bv) > 1:
            # Suppress scipy's catastrophic-cancellation warning: it fires when an
            # arm's seeds are identical (cartpole state-only is exactly 1.0000 on
            # all three), which is a real property of the data, not an error.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, p = stats.ttest_ind(bv, av, equal_var=False)
            tt = f"t={t:.2f}, p={p:.2f}"
            if asd == 0.0 or bsd == 0.0:
                tt += " (one arm has ZERO seed variance; the test is degenerate)"
        else:
            tt = "not computed (n=1)"
        drops = [(r, r.get("pixel_drop_median")) for r in b]
        uses = sum(1 for r, dv in drops
                   if dv is not None and dv > 0.30 and not r.get("inconclusive"))
        cam = f"{uses}/{len(drops)} seeds use it"
        lines.append(
            f"| {ENV_TITLE[env]} | {env_metric_label(new_root, env)} "
            f"| {am:.4f} +/- {asd:.4f} (n={len(av)}) "
            f"| {bm:.4f} +/- {bsd:.4f} (n={len(bv)}) | {bm - am:+.4f} "
            f"| {'YES' if overlap else 'no'} | {tt} | {cam} |")
    lines += ["", "Welch's t-test compares the state+RGB seed means against the "
                  "state-only seed means (two-sided, unequal variance). At n=3 per "
                  "arm it has very little power: a large p-value is NOT evidence "
                  "the arms are equivalent, and a small one would need "
                  "replication.", ""]
    p = Path(outdir) / "README.md"
    p.write_text("\n".join(lines))
    print("wrote", p)


# ----------------------------------------------------------- legacy deprecation
LEGACY_FIG = Path("results/rgb/ablation/summary/method_comparison_nesy.png")
LEGACY_README = """\
# DEPRECATED: `method_comparison_nesy.png`

`method_comparison_nesy.png` (and its `--meta neural` sibling, produced by
`tools/plot_rgb_summary_figures.py`) draws a dashed "privileged upper bound"
that must not be presented. It is **superseded** by
`results/rgb/state_plus_rgb/figures/`.

## Why it is wrong

The "privileged state (cheats)" bar and the dashed upper-bound line are read
from `results/rgb/distill/combined.json`, i.e. from the DISTILLATION
experiment, and are then placed beside the in-loop pixel bars as if the two
were comparable. They are not:

* **Different environment.** Distillation trains its teacher on
  `configs/cartpole_balance_nesy.yaml`, which has `vision=False`. MuJoCo
  Playground's CartpoleBalance keys `ctrl_dt` (0.02 under vision),
  `episode_length` (250 under vision, 1000 otherwise), the REWARD FUNCTION
  (`_dense_vision_reward` vs `_dense_reward`) and the termination rule on
  `vision`. The in-loop bars beside it run the vision env. Different task.
* **Different observation.** The non-vision env feeds the actor the DM-suite
  featurised observation; the vision env feeds it `qpos+qvel`.
* **Different budget.** That teacher trained for 9,830,400 environment steps;
  the in-loop bars beside it had 2,048,000 -- roughly 5x fewer.
* **It is not even an upper bound.** The matched-budget state control measured
  in this campaign reaches an upright fraction of **1.000** at 2.05M steps,
  above the 0.743 the figure draws as the ceiling.

## What to use instead

`results/rgb/state_plus_rgb/figures/` -- `fig1_headline_learning_curves.png`
for baseline-vs-extension and `fig6_all_variants.png` for where every variant
lands. Those state baselines are measured in the SAME environment, at the SAME
budget, through the SAME evaluation code as the RGB arms, from configs that a
generator asserts differ in exactly one key.

The original image is preserved unaltered as
`method_comparison_nesy.SUPERSEDED.png`; the file under the original name now
carries a deprecation banner so it cannot be pasted into a talk by accident.
"""


def deprecate_legacy(root, plt):
    import shutil
    import matplotlib.image as mpimg

    fig_path = Path(root) / LEGACY_FIG
    out_dir = fig_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(LEGACY_README)
    print("wrote", out_dir / "README.md")
    if not fig_path.exists():
        print(f"[skip] {fig_path} missing; wrote the deprecation README only")
        return
    keep = fig_path.with_name("method_comparison_nesy.SUPERSEDED.png")
    if not keep.exists():
        shutil.copy2(fig_path, keep)
        print("preserved original ->", keep)
    img = mpimg.imread(keep)
    h, w = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(w / 140, h / 140), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), w, h, color="white", alpha=0.55, zorder=2))
    ax.text(w / 2, h * 0.34, "DEPRECATED", color="#C44E52", fontsize=52,
            fontweight="bold", ha="center", va="center", zorder=3, alpha=0.95)
    ax.text(w / 2, h * 0.63,
            "The dashed \"privileged upper bound\" is the DISTILLATION teacher:\n"
            "different environment, reward, observation and budget.\n"
            "The matched-budget state control beats it (1.000 vs 0.743).\n"
            "Use  results/rgb/state_plus_rgb/figures/  instead.\n"
            "Original preserved as  method_comparison_nesy.SUPERSEDED.png",
            color="#222222", fontsize=12.5, ha="center", va="center", zorder=3,
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#C44E52", lw=1.5,
                      alpha=0.92))
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print("stamped DEPRECATED ->", fig_path)


FIGS = {
    "curves": ("fig1_headline_learning_curves.png", "new"),
    "camera": ("fig2_camera_use_per_seed.png", "both"),
    "payoff": ("fig3_camera_use_vs_performance.png", "new"),
    "sensitivity": ("fig4_pixel_sensitivity_curves.png", "new"),
    "bars": ("fig5_summary_bars.png", "new"),
    "full": ("fig6_all_variants.png", "both"),
    "budget": ("fig7_budget_gap.png", "both"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", default=NEW_ROOT)
    ap.add_argument("--old-root", default=OLD_ROOT)
    ap.add_argument("--outdir", default="results/rgb/state_plus_rgb/figures")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--figure", default="all",
                    choices=["all"] + list(FIGS) + ["readme", "deprecate"])
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new_root, old_root = Path(args.new_root), Path(args.old_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    w = args.figure

    if w in ("all", "curves"):
        fig_curves(new_root, outdir / FIGS["curves"][0], plt, np)
    if w in ("all", "camera"):
        fig_camera(new_root, old_root, outdir / FIGS["camera"][0], plt, np)
    if w in ("all", "payoff"):
        fig_payoff(new_root, outdir / FIGS["payoff"][0], plt, np)
    if w in ("all", "sensitivity"):
        fig_sensitivity(new_root, outdir / FIGS["sensitivity"][0], plt, np)
    if w in ("all", "bars"):
        fig_bars(new_root, outdir / FIGS["bars"][0], plt, np)
    if w in ("all", "full"):
        fig_full(new_root, old_root, outdir / FIGS["full"][0], plt, np)
    if w in ("all", "budget"):
        fig_budget(new_root, old_root, outdir / FIGS["budget"][0], plt, np)
    if w in ("all", "readme"):
        write_readme(new_root, old_root, outdir)
    if w in ("all", "deprecate"):
        deprecate_legacy(args.repo_root, plt)


if __name__ == "__main__":
    main()
