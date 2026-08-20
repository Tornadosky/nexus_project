"""Figures for the state-only vs state+RGB matched-budget campaign.

Reads the committed run JSON only (no GPU, no retraining) and writes five
figures plus a README describing exactly what each does and does not claim.

Honesty rules baked in, not left to the caller:
  * every panel states its METRIC and its BUDGET;
  * cartpole (upright fraction) and walker/cheetah (reward per step) are NEVER
    put on a shared axis -- each env gets its own panel and its own y-label;
  * error bars are labelled by what they measure. With one seed they are +/-1
    s.d. over the 5 eval episodes and the bar is annotated "n=1 seed"; with
    several seeds they are +/-1 s.d. ACROSS SEEDS and annotated "n=<k> seeds".
    Nothing is drawn that was not measured;
  * a run the rescore guard marked INCONCLUSIVE is drawn grey and labelled
    inconclusive, never given a verdict colour;
  * the 52.4M-step state baseline is plotted ONLY if a run artifact exists.
    It is otherwise omitted with a visible note, never estimated.

    python tools/plot_state_plus_rgb_figures.py --figure all
    python tools/plot_state_plus_rgb_figures.py --figure headline --outdir /tmp/f
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# House palette (seaborn "deep"), matching tools/plot_rgb_ablation_comparison.py.
C_STATE = "#4C72B0"   # state-only
C_SPRGB = "#DD8452"   # state + RGB
C_PIXEL = "#937860"   # pixels-only (RGB_PROPRIO: none)
C_SEES = "#55A868"
C_BLIND = "#C44E52"
C_INCONC = "#8C8C8C"
C_ACCENT = "#8172B2"

ENVS = ["cartpole", "walker", "cheetah"]
ENV_TITLE = {"cartpole": "CartpoleBalance", "walker": "WalkerWalk", "cheetah": "CheetahRun"}

NEW_ROOT = "results/rgb/state_plus_rgb"
OLD_ROOT = "results/rgb/ablation"

# The two new matched-budget arms.
NEW_ARMS = {
    "state_matched": ("state-only\n(no camera)", C_STATE),
    "state_plus_rgb": ("state + RGB\n(camera added)", C_SPRGB),
}

# Every committed variant, per env, for the full-picture panel. Each entry maps a
# short label to the list of run tags that are seeds OF THE SAME variant.
OLD_VARIANTS = {
    "cartpole": [
        ("RGB pixels-only", ["nesy_blind", "nesy_blind_seed1", "nesy_blind_seed2"]),
        ("RGB aux-fix", ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
        ("RGB no-aux MDI4", ["nesy_noaux_mdi4_seed0", "nesy_noaux_mdi4_seed1",
                             "nesy_noaux_mdi4_seed2"]),
        ("RGB shared", ["nesy_shared_seed0"]),
        ("RGB shared no-aux", ["nesy_shared_noaux_mdi4_seed0"]),
        ("RGB shared+meta-z", ["nesy_shared_metaz_noaux_seed0", "nesy_shared_metaz_noaux_seed1",
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
    "cheetah": [
        ("RGB pixels-only", ["nesy_seed0", "nesy_seed1", "nesy_seed2"]),
    ],
}

# The pixels-only arm that shares a base config with the new arms (the third rung
# of the one-variable ladder), and the best-known SEEING arm, per env.
LADDER_PIXEL = {"cartpole": ("nesy_blind", ["nesy_blind", "nesy_blind_seed1", "nesy_blind_seed2"]),
                "walker": ("nesy_blind", ["nesy_blind"]),
                "cheetah": ("nesy_seed0", ["nesy_seed0", "nesy_seed1", "nesy_seed2"])}
KNOWN_SEEING = {"cartpole": ("RGB aux-fix (pixels-only)",
                             ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
                "walker": ("RGB aux-fix (pixels-only)",
                           ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"]),
                "cheetah": ("RGB pixels-only",
                            ["nesy_seed0", "nesy_seed1", "nesy_seed2"])}

MATCHED_BUDGET = "2.05M env steps (250 updates x 128 envs x 64 steps)"


# --------------------------------------------------------------------------- io
def load(root: Path, env: str, tag: str):
    p = root / env / tag / "pixel_ablation.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["_dir"] = p.parent
    return d


def metric_info(d):
    """(key, human label, value, per-episode std) for a run's intact condition."""
    i = d["results"]["intact"]
    if "upright_fraction_mean" in i:
        return ("upright_fraction_mean", "upright fraction",
                i["upright_fraction_mean"], i["upright_fraction_std"])
    return ("reward_per_step_mean", "reward / step",
            i["reward_per_step_mean"], i["reward_per_step_std"])


def aggregate(runs):
    """Mean + an error bar whose MEANING is returned alongside it.

    One seed -> the spread over that run's 5 eval episodes (measured, but it is
    episode noise, not seed noise). Several seeds -> the spread across seeds.
    """
    import numpy as np

    vals = [metric_info(d)[2] for d in runs]
    if len(runs) == 1:
        return float(vals[0]), float(metric_info(runs[0])[3]), 1, "5 eval episodes"
    return float(np.mean(vals)), float(np.std(vals)), len(runs), f"{len(runs)} seeds"


def collect(root: Path, env: str, tags):
    return [d for d in (load(root, env, t) for t in tags) if d is not None]


def seed_tags(arm, seeds=(0, 1, 2)):
    return [f"{arm}_seed{s}" for s in seeds]


def annotate(ax, x, y, err, n, text=None, fs=8):
    lab = text if text is not None else f"{y:.3f}"
    ax.text(x, y + err + 0.02 * ax.get_ylim()[1], lab, ha="center", va="bottom",
            fontsize=fs, fontweight="bold")


# ---------------------------------------------------------------- fig 1 headline
def fig_headline(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
    any_data = False
    for ax, env in zip(axes, ENVS):
        bars, labels, colors, errs, ns, kinds = [], [], [], [], [], []
        ylab = None
        for arm, (lab, col) in NEW_ARMS.items():
            runs = collect(new_root, env, seed_tags(arm))
            if not runs:
                continue
            ylab = metric_info(runs[0])[1]
            m, e, n, kind = aggregate(runs)
            bars.append(m); errs.append(e); labels.append(lab); colors.append(col)
            ns.append(n); kinds.append(kind)
        if not bars:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs found)", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        any_data = True
        x = np.arange(len(bars))
        b = ax.bar(x, bars, 0.55, yerr=errs, capsize=6, color=colors,
                   edgecolor="white", linewidth=1.2)
        top = max(v + e for v, e in zip(bars, errs))
        ax.set_ylim(0, top * 1.42)
        for i, (v, e, n, kind) in enumerate(zip(bars, errs, ns, kinds)):
            ax.text(i, v + e + top * 0.03, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
            ax.text(i, v + e + top * 0.125, f"n={n} seed{'s' if n > 1 else ''}",
                    ha="center", va="bottom", fontsize=8, color="#555555")
        # Make the winner unmissable.
        if len(bars) == 2:
            w = int(np.argmax(bars))
            d = bars[w] - bars[1 - w]
            rel = 100.0 * d / max(abs(bars[1 - w]), 1e-9)
            wl = "state-only" if w == 0 else "state + RGB"
            ax.annotate("", xy=(w, top * 1.30), xytext=(1 - w, top * 1.30),
                        arrowprops=dict(arrowstyle="-|>", color=colors[w], lw=2.2))
            ax.text(0.5, top * 1.33, f"{wl} wins by {d:.3f}  ({rel:+.0f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=colors[w])
            b[w].set_edgecolor("#333333"); b[w].set_linewidth(2.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(f"{ylab}  (eval, higher is better)", fontsize=10)
        ax.set_title(ENV_TITLE[env], fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
    if not any_data:
        raise SystemExit("headline: no runs found")
    fig.suptitle("Does adding a camera to a skill actor that ALREADY HAS the state help?\n"
                 f"Matched budget, both arms: {MATCHED_BUDGET}, nesy meta, seed 0. "
                 "Each pair differs in ONE config key (RGB_ACTOR).",
                 fontsize=13, y=1.03)
    fig.text(0.5, -0.045,
             "Metric per panel: CartpoleBalance = upright fraction (bounded 0-1); "
             "WalkerWalk / CheetahRun = task reward per step. Different metrics, so the "
             "panels share no axis.\nError bars: +/-1 s.d. across seeds where several "
             "seeds were run, otherwise +/-1 s.d. across the 5 evaluation episodes of the "
             "single seed. Every bar is annotated with its n.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------- fig 2 camera use
def fig_camera(new_root, old_root, out, plt, np):
    rows = []
    for env in ENVS:
        runs = collect(new_root, env, seed_tags("state_plus_rgb"))
        if runs:
            vals = [r["pixel_drop_median"] for r in runs if r.get("pixel_drop_median") is not None]
            inc = any(r.get("inconclusive") for r in runs)
            if vals:
                rows.append((f"{ENV_TITLE[env]}\nstate + RGB  (this campaign)",
                             100 * float(np.mean(vals)),
                             100 * float(np.std(vals)) if len(vals) > 1 else 0.0,
                             len(vals), inc, True))
    rows.append(None)  # separator
    for env in ENVS:
        lab, tags = KNOWN_SEEING[env]
        runs = collect(old_root, env, tags)
        vals = [r["pixel_drop_median"] for r in runs if r.get("pixel_drop_median") is not None]
        inc = any(r.get("inconclusive") for r in runs)
        if vals:
            rows.append((f"{ENV_TITLE[env]}\n{lab}  (reference: known to SEE)",
                         100 * float(np.mean(vals)),
                         100 * float(np.std(vals)) if len(vals) > 1 else 0.0,
                         len(vals), inc, False))
    rows = [r for r in rows if r is not None or True]
    data = [r for r in rows if r is not None]
    if not data:
        raise SystemExit("camera figure: nothing to plot")

    fig, ax = plt.subplots(figsize=(13, 0.78 * len(data) + 3.4))
    ypos = np.arange(len(data))[::-1]
    lo = min(-8, min(r[1] - r[2] for r in data) - 8)
    # Annotations live in a fixed column to the right of every bar, so they can
    # never collide with the threshold line or with each other.
    text_x = 112
    for y, (lab, v, e, n, inc, is_new) in zip(ypos, data):
        col = C_INCONC if inc else (C_SEES if v > 30 else C_BLIND)
        ax.barh(y, v, xerr=e if e else None, capsize=5, color=col, height=0.62,
                edgecolor="#333333" if is_new else "white",
                linewidth=1.8 if is_new else 0.8)
        tag = "INCONCLUSIVE" if inc else ("SEES" if v > 30 else "BLIND")
        ax.text(text_x, y, f"{v:+.1f}%   {tag}   (n={n})", va="center", ha="left",
                fontsize=10.5, fontweight="bold" if is_new else "normal",
                color=col if not inc else "#555555")
    ax.axvline(30, ls="--", lw=2, color="#333333")
    ax.text(30.8, len(data) - 0.52, "verdict threshold 30%", fontsize=9.5,
            color="#333333", ha="left", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB"))
    ax.set_ylim(-0.7, len(data) - 0.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in data], fontsize=9.5)
    ax.set_xlabel("median performance lost when the actor's camera is corrupted\n"
                  "(median over frozen-first / wrong-timestep / blank-image, "
                  "as % of the intact score)", fontsize=10)
    ax.set_xlim(lo, 168)
    ax.set_xticks([t for t in range(0, 101, 20)])
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title("Does the actor use the camera AT ALL when it already has the state?\n"
                 "Bold outline = this campaign's state+RGB arms; thin = committed "
                 "pixels-only arms.\nBoth at " + MATCHED_BUDGET + ".",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.03,
             "Higher = the actor depends on its camera. Near 0% means corrupting or "
             "blanking the image changes performance not at all: the actor is ignoring "
             "the camera and riding the state.\nThe reference arms have no state input, "
             "so they must use pixels or fail; they calibrate what a genuinely seeing "
             "actor looks like under this identical protocol.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------- fig 3 training curves
def fig_curves(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    found = False
    for ax, env in zip(axes, ENVS):
        for arm, (lab, col) in NEW_ARMS.items():
            for si, tag in enumerate(seed_tags(arm)):
                p = new_root / env / tag / "training_curves.json"
                if not p.exists():
                    continue
                c = json.loads(p.read_text())["curves"].get("episode_return")
                if not c:
                    continue
                found = True
                ax.plot(np.arange(len(c)), c, lw=1.9 if si == 0 else 1.0,
                        alpha=1.0 if si == 0 else 0.45, color=col,
                        label=lab.replace("\n", " ") if si == 0 else None)
                # The evaluation uses the FINAL weights, so mark the final point.
                # Without this a late collapse (cartpole state+RGB seed 0 fell from
                # 24.4 to 9.9 over its last three updates) is easy to miss, and the
                # eval numbers look inexplicable next to the curve.
                ax.plot([len(c) - 1], [c[-1]], marker="o", ms=7 if si == 0 else 4,
                        color=col, mec="#222222", mew=1.2, zorder=5)
        ax.set_title(ENV_TITLE[env], fontsize=12, fontweight="bold")
        ax.set_xlabel("training update  (250 = 2.05M env steps)", fontsize=9.5)
        ax.set_ylabel("episode return during TRAINING\n(trainer metric, with exploration noise)",
                      fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    if not found:
        raise SystemExit("curves: nothing found")
    fig.suptitle("Training curves, state-only vs state+RGB at matched budget "
                 "(faint lines = additional seeds)", fontsize=12.5, y=1.04)
    fig.text(0.5, -0.03,
             "CAUTION: this is the TRAINER's episode return, collected with exploration "
             "noise across 128 environments -- NOT the deterministic evaluation metric "
             "used in the other figures. It shows learning progress, not the head-to-head "
             "result.\nThe ringed marker is the FINAL update, which is the checkpoint the "
             "evaluation actually scores. Where a curve collapses at the end, the eval "
             "number reflects the collapse, not the plateau.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FFF4E5", ec="#DDAA66"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------ fig 4 full picture
def fig_full(new_root, old_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    for ax, env in zip(axes, ENVS):
        entries = []   # (label, mean, err, n, color, inconclusive)
        for arm, (lab, col) in NEW_ARMS.items():
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab.replace("\n", " "), m, e, n, col, False))
        for lab, tags in OLD_VARIANTS.get(env, []):
            runs = collect(old_root, env, tags)
            if not runs:
                continue
            m, e, n, _ = aggregate(runs)
            inc = any(r.get("inconclusive") for r in runs)
            entries.append((lab, m, e, n, C_INCONC if inc else C_PIXEL, inc))
        # full-budget arms, if phase 2 has produced them
        for arm, lab, col in (("state_matched_full", "state-only @ 52.4M", C_STATE),
                              ("state_plus_rgb_full", "state+RGB @ 52.4M", C_SPRGB)):
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab, m, e, n, col, False))
        if not entries:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ylab = None
        for tags_root, tags in ((new_root, seed_tags("state_matched")),):
            r = collect(tags_root, env, tags)
            if r:
                ylab = metric_info(r[0])[1]
        if ylab is None:
            r = collect(old_root, env, OLD_VARIANTS[env][0][1])
            ylab = metric_info(r[0])[1] if r else "score"
        entries.sort(key=lambda t: -t[1])
        x = np.arange(len(entries))
        vals = [e[1] for e in entries]
        errs = [e[2] for e in entries]
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
    handles = [mpatches.Patch(color=C_STATE, label="state-only (no camera)"),
               mpatches.Patch(color=C_SPRGB, label="state + RGB"),
               mpatches.Patch(color=C_PIXEL, label="pixels-only variants (committed)"),
               mpatches.Patch(color=C_INCONC, label="inconclusive under the rescore guard")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.005), framealpha=0.95)
    fig.suptitle("Every approach we have, on one axis per environment\n"
                 f"All bars: {MATCHED_BUDGET}, nesy meta, identical evaluation protocol "
                 "(5 deterministic episodes x 250 steps)",
                 fontsize=13, y=1.10)
    fig.text(0.5, -0.02,
             "Units are consistent WITHIN each panel and stated in each panel title; they "
             "differ BETWEEN panels (upright fraction vs reward per step), so bar heights "
             "must not be compared across panels.\nn = number of seeds; error bars are "
             "+/-1 s.d. across seeds when n>1, else across the 5 eval episodes of the "
             "single seed.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# -------------------------------------------------------------- fig 5 budget gap
def fig_budget(new_root, old_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    missing, any_full = [], False
    for ax, env in zip(axes, ENVS):
        entries = []
        runs = collect(new_root, env, seed_tags("state_matched_full"))
        if runs:
            m, e, n, _ = aggregate(runs)
            entries.append(("state-only\n@ 52.4M", m, e, n, C_STATE))
            any_full = True
        else:
            missing.append(ENV_TITLE[env])
        runs = collect(new_root, env, seed_tags("state_plus_rgb_full"))
        if runs:
            m, e, n, _ = aggregate(runs)
            entries.append(("state+RGB\n@ 52.4M", m, e, n, C_SPRGB))
            any_full = True
        for arm, lab, col in (("state_matched", "state-only\n@ 2.05M", C_STATE),
                              ("state_plus_rgb", "state+RGB\n@ 2.05M", C_SPRGB)):
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab, m, e, n, col))
        _, tags = LADDER_PIXEL[env]
        runs = collect(old_root, env, tags)
        if runs:
            m, e, n, _ = aggregate(runs)
            entries.append(("pixels-only\n@ 2.05M", m, e, n, C_PIXEL))
        if not entries:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ylab = metric_info(collect(new_root, env, seed_tags("state_matched"))[0])[1] \
            if collect(new_root, env, seed_tags("state_matched")) else "score"
        x = np.arange(len(entries))
        vals = [e[1] for e in entries]; errs = [e[2] for e in entries]
        hatch = ["//" if "52.4M" in e[0] else "" for e in entries]
        bars = ax.bar(x, vals, 0.6, yerr=errs, capsize=5,
                      color=[e[4] for e in entries], edgecolor="#333333", linewidth=1.0)
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
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
    note = ("Every bar is labelled with its budget. Hatched bars are the 25x-larger "
            "52.4M-step budget; solid bars are 2.05M. Never compare a hatched bar "
            "with a solid one without reading the budget."
            if any_full else
            "Every bar shown is at the 2.05M-step matched budget.")
    if missing:
        note += ("\nNO 52.4M-step state bar is shown for " + ", ".join(missing) +
                 ": the 52.4M state baselines quoted in earlier reports have NO run "
                 "artifact in results/ (it contains only results/rgb), so the number "
                 "cannot be verified. It is OMITTED rather than estimated.")
    fig.suptitle("The budget gap, made measurable\n"
                 "Every state baseline previously cited was trained at 52.4M steps while "
                 "the RGB arms got 2.05M -- a 25x gap.", fontsize=12.5, y=1.05)
    fig.text(0.5, -0.09, note, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FFF4E5", ec="#DDAA66"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


README = """\
# state-only vs state+RGB at matched budget -- figures

Generated by `tools/plot_state_plus_rgb_figures.py` from the committed run JSON.
Regenerate with:

    python tools/plot_state_plus_rgb_figures.py --figure all

## The question these figures answer

The NEXUS paper suggests using RGB inputs for the skill agents. Earlier work here
tested RGB *replacing* the state (`RGB_PROPRIO: none`). These figures test RGB
*added to* the state (`RGB_PROPRIO: full`) against a state-only control at the
same environment-step budget.

## How the comparison is kept honest

Both arms of every pair are generated from ONE base config by
`tools/gen_state_plus_rgb_configs.py`, which asserts that the resolved configs
differ in exactly one key, `RGB_ACTOR`. `RGB_ACTOR` gates only whether the skill
actors have a camera pathway; the ENVIRONMENT keeps `USE_RGB: true` in both arms
and still renders. That matters: `USE_RGB` also switches the task. MuJoCo
Playground's CartpoleBalance keys `ctrl_dt`, `episode_length`, the reward
function and the termination rule on `vision`, and the vec wrapper changes the
actor's state vector, so a `USE_RGB: false` baseline would differ in the
environment, the reward, the horizon and the state representation as well.

Both arms are also scored by the same code: `rgb_pixel_ablation.py` runs the same
`rollout()`, the same metric keys and the same scoring loop for both. The
state-only arm scores the `intact` condition only, because pixel corruptions are
undefined without a pixel input.

## The figures

| file | shows | does NOT claim |
|---|---|---|
| `fig1_headline_state_vs_state_plus_rgb.png` | state-only vs state+RGB, one panel per env, matched 2.05M-step budget | Nothing about other budgets, other metas, or envs outside these three. Panels use different metrics and share no axis. |
| `fig2_does_the_actor_use_the_camera.png` | how much performance each actor loses when its camera is corrupted, against the 30% verdict threshold and against pixels-only arms known to see | It measures the ACTOR's pixel dependence only. It is not a claim about the hierarchy as a whole, and a low value is not evidence the camera *could not* help, only that this actor did not use it. |
| `fig3_training_curves.png` | trainer episode return over training, both arms overlaid | This is the TRAINER's metric with exploration noise, NOT the deterministic eval used elsewhere. The two disagree for pixel arms. Do not read the head-to-head result off this figure. |
| `fig4_all_variants.png` | every committed variant plus the two new arms, per env, on one axis | Units differ between panels; bar heights are not comparable across panels. Variants differ in more than one config key from each other -- only the state-only / state+RGB pair is a controlled contrast. |
| `fig5_budget_gap.png` | the budget question: what we actually measured at 2.05M, plus full-budget arms where they exist | Where no 52.4M artifact exists the bar is OMITTED, not estimated. `results/` contains only `results/rgb`, so the 52.4M state baselines quoted in earlier prose have no verifiable artifact. |

## Reading rules that apply to every figure

* Metrics differ by environment: CartpoleBalance uses the bounded upright
  fraction (0-1), computed geometrically from `qpos` and therefore independent of
  the env's reward function; WalkerWalk and CheetahRun use task reward per step.
* `n` is the number of SEEDS. Error bars are +/-1 s.d. across seeds when n > 1,
  and +/-1 s.d. across the 5 evaluation episodes of the single seed when n = 1.
  Every bar is annotated with its n so the two cannot be confused.
* Runs the rescore guard marks INCONCLUSIVE (intact score not distinguishable
  from zero) are drawn grey and labelled, never given a verdict colour.

## Caveats worth carrying into any write-up

* The evaluation scores the FINAL weights. Where a training curve collapses in
  its last updates the eval number reflects that collapse, which is why fig3
  marks the final point.
* `const_action` is not a pixel-specific control for a `RGB_PROPRIO: full` actor:
  replacing the actor with a per-skill constant removes its state-driven
  variation too. Only the frozen/replay/blank conditions isolate the pixels.
* CartpoleBalance's base config uses `NUM_MINIBATCHES: 8` where walker and
  cheetah use 64, so cartpole takes 8x fewer gradient steps per env step. It is
  identical in both arms of the pair, so it cannot confound state vs state+RGB,
  but cartpole numbers are not comparable across environments.
"""


def write_readme(new_root, old_root, outdir):
    lines = [README, "\n## Measured numbers (auto-generated, do not hand-edit)\n"]
    lines.append("| env | arm | metric | intact | n | train return | "
                 "median pixel drop | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    import numpy as np
    for env in ENVS:
        for arm, (lab, _c) in NEW_ARMS.items():
            for extra in ("", "_full"):
                runs = collect(new_root, env, seed_tags(arm + extra))
                if not runs:
                    continue
                key, ylab, _v, _s = metric_info(runs[0])
                m, e, n, kind = aggregate(runs)
                tr = [r.get("final_train_return") for r in runs
                      if r.get("final_train_return") is not None]
                trs = f"{np.mean(tr):.1f}" if tr else "n/a"
                if runs[0].get("state_only"):
                    drop, verdict = "n/a (no pixels)", "n/a"
                else:
                    ds = [r["pixel_drop_median"] for r in runs
                          if r.get("pixel_drop_median") is not None]
                    drop = f"{100 * np.mean(ds):+.1f}%" if ds else "n/a"
                    if any(r.get("inconclusive") for r in runs):
                        verdict = "INCONCLUSIVE"
                    else:
                        verdict = "SEES" if np.mean(ds) > 0.30 else "**BLIND**"
                budget = "52.4M" if extra else "2.05M"
                lines.append(
                    f"| {ENV_TITLE[env]} | {lab.replace(chr(10), ' ')} @{budget} | {ylab} "
                    f"| {m:.4f} +/- {e:.4f} ({kind}) | {n} | {trs} | {drop} | {verdict} |")
    p = Path(outdir) / "README.md"
    p.write_text("\n".join(lines) + "\n")
    print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", default=NEW_ROOT)
    ap.add_argument("--old-root", default=OLD_ROOT)
    ap.add_argument("--outdir", default="results/rgb/state_plus_rgb/figures")
    ap.add_argument("--figure", default="all",
                    choices=["all", "headline", "camera", "curves", "full", "budget",
                             "readme"])
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new_root, old_root = Path(args.new_root), Path(args.old_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    want = args.figure

    if want in ("all", "headline"):
        fig_headline(new_root, outdir / "fig1_headline_state_vs_state_plus_rgb.png", plt, np)
    if want in ("all", "camera"):
        fig_camera(new_root, old_root, outdir / "fig2_does_the_actor_use_the_camera.png", plt, np)
    if want in ("all", "curves"):
        fig_curves(new_root, outdir / "fig3_training_curves.png", plt, np)
    if want in ("all", "full"):
        fig_full(new_root, old_root, outdir / "fig4_all_variants.png", plt, np)
    if want in ("all", "budget"):
        fig_budget(new_root, old_root, outdir / "fig5_budget_gap.png", plt, np)
    if want in ("all", "readme"):
        write_readme(new_root, old_root, outdir)


if __name__ == "__main__":
    main()
