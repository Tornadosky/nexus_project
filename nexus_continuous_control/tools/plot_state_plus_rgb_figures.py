"""Figures for the state-only vs state+RGB matched-budget campaign.

Reads the committed run JSON only (no GPU, no retraining) and writes the figure
set plus a README describing what each one does and does not claim.

CORRECTION PASS (see `tools/analyze_state_plus_rgb.py` for the full argument).
The first version of these figures took the 5-episode deterministic evaluation
as the headline number while labelling the training curves "PRIMARY", and
binarised camera use at a single 30% median. Both were wrong:

  * the 5-episode eval has an achieved power of 0.16 at this effect size where
    the training return has 0.78, and the two metrics DISAGREE on WalkerWalk --
    the eval shows overlapping ranges, the training return shows complete
    separation (189.5 +/- 1.8 vs 203.2 +/- 6.2, 3/3 seeds, p = 0.054);
  * the 30% median rule reported walker as 2/3 seeds, but `frozen_first` alone
    costs 94.9% / 74.2% / 80.2% on the three seeds and the independent
    in-training probe reads 0.098 / 0.099 / 0.117. Walker is 3/3.

So: the TRAINING RETURN is the primary metric in every figure here and is
labelled as such; the evaluation is shown beside it, never instead of it; and
camera use is drawn condition by condition rather than as one thresholded bar.

Honesty rules are enforced here rather than left to the caller:
  * every panel states its METRIC, its BUDGET, its n and its s.d. convention;
  * s.d. is the SAMPLE s.d. (ddof=1) everywhere. The earlier figures used
    numpy's default population s.d., which at n=3 understates spread by 1.22x;
  * cartpole (upright fraction) and walker/cheetah (reward per step) never
    share an axis;
  * per-SEED points are always drawn, never only a mean;
  * a run the rescore guard marks INCONCLUSIVE is drawn grey and labelled;
  * the 52.4M-step state baseline is plotted only if a run artifact exists.

    python tools/plot_state_plus_rgb_figures.py --figure all
    python tools/plot_state_plus_rgb_figures.py --figure curves --outdir /tmp/f
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

# House palette (seaborn "deep").
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
EVAL30_ROOT = "results/rgb/state_plus_rgb_eval30"
# The pixels-only reference arms (`RGB_PROPRIO: none`, the superseded campaign)
# used to be read out of `results/rgb/ablation/`. That tree was removed; the
# handful of scalars these figures actually quote from it were distilled into
# the JSON below, with provenance, so figs 2/4/6/7/8 stay reproducible.
REF_JSON = "results/rgb/state_plus_rgb/reference_baselines.json"

NEW_ARMS = {
    "state_matched": ("state only (baseline)", C_STATE),
    "state_plus_rgb": ("state + RGB (extension)", C_SPRGB),
}

# The five pixel corruptions, in the order they are drawn. `const_action` is
# kept visually apart: for a `RGB_PROPRIO: full` actor it removes the actor's
# STATE-driven variation too, so it is not a pixel-specific control.
CORRUPTIONS = ["frozen_first", "random_replay", "shuffle_frames", "zeros"]
CORR_LABEL = {
    "frozen_first": "frozen_first\n(image held at t=0)",
    "random_replay": "random_replay\n(real frame, wrong t)",
    "shuffle_frames": "shuffle_frames\n(motion destroyed)",
    "zeros": "zeros\n(blank image)",
    "const_action": "const_action\n(NOT pixel-specific)",
}
CORR_COL = {"frozen_first": "#C44E52", "random_replay": "#DD8452",
            "shuffle_frames": "#8172B3", "zeros": "#64B5CD",
            "const_action": "#B0B0B0"}

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
# Matched-budget pixels-only arms used as the "an actor that MUST see" control.
# `one_key` says whether the arm differs from the state+RGB arm in RGB_PROPRIO
# alone; only cheetah does at 3 seeds (see analyze_state_plus_rgb.py).
KNOWN_SEEING = {
    "cartpole": ("RGB aux-fix (pixels-only)",
                 ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"],
                 False),
    "walker": ("RGB aux-fix (pixels-only)",
               ["nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"],
               False),
    "cheetah": ("RGB pixels-only", ["nesy_seed0", "nesy_seed1", "nesy_seed2"],
                True),
}

MATCHED_BUDGET = "2.05M env steps (250 updates x 128 envs x 64 steps)"
SMOOTH_W = 11
LAST_N = 20
SD_NOTE = ("s.d. is the SAMPLE s.d. (ddof=1) across the 3 seeds, not numpy's "
           "default population s.d.")


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


# --------------------------------------------------- pixels-only reference arms
_REF_CACHE = {}


def _ref(ref_json=None):
    """The distilled pixels-only reference values (see REF_JSON above).

    `REF_JSON` is read at CALL time, not bound as a default, so `--ref-json`
    reaches every reader below.
    """
    ref_json = ref_json or REF_JSON
    if ref_json not in _REF_CACHE:
        p = Path(ref_json)
        _REF_CACHE[ref_json] = (json.loads(p.read_text())["runs"]
                                if p.exists() else {})
    return _REF_CACHE[ref_json]


def load_old(env: str, tag: str, ref_json=None):
    """A reduced stand-in for that run's `pixel_ablation.json`.

    It carries the same `results` / `metric_key` / `inconclusive` fields the
    readers below use, so `metric_info`, `pixel_drops` and `aggregate` behave
    exactly as they did against the removed tree.
    """
    return _ref(ref_json).get(f"{env}/{tag}")


def old_sens(env: str, tag: str, ref_json=None):
    """That run's last-20-update mean pixel sensitivity, precomputed."""
    r = _ref(ref_json).get(f"{env}/{tag}")
    return None if r is None else r.get("pixel_sensitivity_last20")


def collect_old(env, tags, ref_json=None):
    return [d for d in (load_old(env, t, ref_json) for t in tags)
            if d is not None]


def metric_info(d):
    """(key, human label, value, per-episode std) for a run's intact condition."""
    i = d["results"]["intact"]
    if "upright_fraction_mean" in i:
        sd = i.get("upright_fraction_std_sample") or i["upright_fraction_std"]
        return ("upright_fraction_mean", "upright fraction",
                i["upright_fraction_mean"], sd)
    sd = i.get("reward_per_step_std_sample") or i["reward_per_step_std"]
    return ("reward_per_step_mean", "reward / step",
            i["reward_per_step_mean"], sd)


def train_return(root: Path, env: str, tag: str):
    """PRIMARY metric: last-20-update mean of the trainer's episode return."""
    import numpy as np
    c = load_curve(root, env, tag)
    if not c:
        return None
    return float(np.asarray(c, float)[-LAST_N:].mean())


def aggregate(runs):
    """Mean plus an error bar whose MEANING is returned with it (ddof=1)."""
    import numpy as np
    vals = [metric_info(d)[2] for d in runs]
    if len(runs) == 1:
        return float(vals[0]), float(metric_info(runs[0])[3]), 1, \
            "eval episodes of 1 seed"
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(runs), \
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


def stats_pair(a, b):
    """Welch + paired-by-seed + Cohen's d on two arms' per-seed values."""
    import numpy as np
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, p = stats.ttest_ind(b, a, equal_var=False)
        pt, pp = (stats.ttest_rel(b, a) if a.size > 1 and (b - a).std() > 0
                  else (float("nan"), float("nan")))
    n1, n2 = a.size, b.size
    sp2 = ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2)
    d = (b.mean() - a.mean()) / math.sqrt(sp2) if sp2 > 0 else float("inf")
    return {"welch_t": float(t), "welch_p": float(p), "d": d,
            "paired_t": float(pt), "paired_p": float(pp),
            "wins": int((b - a > 0).sum()), "n": int(n1),
            "overlap": bool(a.min() <= b.max() and b.min() <= a.max())}


def pixel_drops(d):
    """{condition: drop fraction} recomputed from the stored per-condition means."""
    if d is None or d.get("state_only"):
        return {}
    k = d.get("metric_key") or ("upright_fraction_mean"
                                if "upright_fraction_mean" in d["results"]["intact"]
                                else "reward_per_step_mean")
    base = d["results"]["intact"][k]
    return {c: float((base - d["results"][c][k]) / max(abs(base), 1e-9))
            for c in d["results"] if c != "intact"}


def uses_camera_corrected(d):
    """Corrected verdict: ANY single pixel corruption costing >30%.

    The stored `actor_uses_pixels` thresholds the MEDIAN of three corruptions
    at 30%, a bar calibrated on `RGB_PROPRIO: none` actors whose only input is
    the camera. A `RGB_PROPRIO: full` actor can ride the state through a
    corrupted frame, so the median under-reads. A corruption changes nothing
    but the actor's image, so one large drop is already proof of dependence.
    """
    dr = pixel_drops(d)
    vals = [dr[c] for c in ("frozen_first", "random_replay", "zeros") if c in dr]
    return bool(vals and max(vals) > 0.30)


# ------------------------------------------------------- fig 1: LEARNING CURVES
def fig_curves(new_root, out, plt, np):
    """THE headline, on the PRIMARY metric. Per-seed curves, sample-s.d. band."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8))
    found = False
    for ax, env in zip(axes, ENVS):
        finals, per_arm = [], {}
        for arm, (lab, col) in NEW_ARMS.items():
            curves = [c for c in (load_curve(new_root, env, t)
                                  for t in seed_tags(arm)) if c]
            if not curves:
                continue
            found = True
            n = min(len(c) for c in curves)
            raw = np.stack([np.asarray(c[:n], float) for c in curves])
            sm = np.stack([smooth(r) for r in raw])
            mu = sm.mean(0)
            sd = sm.std(0, ddof=1) if sm.shape[0] > 1 else np.zeros_like(mu)
            x = np.arange(n)
            # Every seed drawn: the band is a summary, the thin lines are the data.
            for r in sm:
                ax.plot(x, r, color=col, lw=0.9, alpha=0.55, zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.18, lw=0,
                            zorder=2)
            ax.plot(x, mu, color=col, lw=2.6, label=f"{lab}  (n={len(curves)})",
                    zorder=4)
            per_arm[arm] = raw[:, -LAST_N:].mean(1)
            finals.append((x[-1], float(mu[-1]), col))
        if not per_arm:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # Mark the window the primary number is read from. No end-of-curve
        # value is annotated: the smoothed last point is NOT the primary
        # number and printing both invites quoting the wrong one.
        lo = max(0, n - LAST_N)
        ax.axvspan(lo, n - 1, color="#333333", alpha=0.10, zorder=1)
        y0, y1 = ax.get_ylim()
        # Reserve empty space under the data for the statistics box.
        ax.set_ylim(y0 - 0.46 * (y1 - y0), y1)
        ax.text(n + 12, y1, f"last {LAST_N} updates = the PRIMARY number",
                rotation=90, fontsize=8, color="#333333", ha="left", va="top",
                fontweight="bold")
        ax.set_title(f"{ENV_TITLE[env]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("training update      (250 updates = 2.048M env steps)",
                      fontsize=9.5)
        ax.set_ylabel("episode return during training\n(mean over 128 parallel "
                      "envs)", fontsize=9.5)
        ax.set_xlim(0, 288)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
        if len(per_arm) == 2:
            a, b = per_arm["state_matched"], per_arm["state_plus_rgb"]
            st = stats_pair(a, b)
            verdict = ("ranges OVERLAP" if st["overlap"] else "NO OVERLAP")
            col = C_SPRGB if b.mean() > a.mean() else C_STATE
            ax.text(0.5, 0.015,
                    f"last-{LAST_N}-update mean +/- sample s.d. (n=3)\n"
                    f"state only   {a.mean():8.2f} +/- {a.std(ddof=1):.2f}\n"
                    f"state + RGB  {b.mean():8.2f} +/- {b.std(ddof=1):.2f}\n"
                    f"diff {b.mean() - a.mean():+.2f} "
                    f"({100 * (b.mean() - a.mean()) / abs(a.mean()):+.1f}%), "
                    f"{st['wins']}/3 seeds, {verdict}\n"
                    f"Welch p={st['welch_p']:.3f}, d={st['d']:+.2f}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8.6, family="monospace", color=col,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=col, alpha=0.95))
    if not found:
        raise SystemExit("curves: nothing found")
    fig.suptitle("PRIMARY METRIC. Does adding a camera to a skill actor that "
                 "ALREADY HAS the state help?\n"
                 f"nesy meta, matched budget ({MATCHED_BUDGET}); each pair of arms "
                 "differs in exactly ONE config key (RGB_ACTOR); n = 3 seeds per arm",
                 fontsize=13.5, y=1.05)
    fig.text(0.5, -0.06,
             "METRIC: the trainer's episode return, already a mean over 128 "
             "parallel environments, summarised as the mean of its LAST 20 of "
             f"250 updates (shaded). BUDGET: {MATCHED_BUDGET}. n = 3 seeds.\n"
             f"{SD_NOTE} Thin lines are the individual seeds after an "
             f"{SMOOTH_W}-update centred rolling mean; the bold line is their "
             "mean and the band is +/-1 sample s.d. ACROSS SEEDS.\n"
             "This is the PRIMARY metric because it is the one with the "
             "resolution to answer the question: at this effect size its "
             "achieved power at n=3 is 0.78, against 0.16 for the 5-episode "
             "deterministic eval shown in fig5.\nThe 20-update window is not "
             "load-bearing: WalkerWalk is 3/3 seeds with NON-OVERLAPPING "
             "ranges at every window from 5 to 50 updates (+6.2% to +8.1%, p "
             "0.026-0.085) -- the reported window is the middle of that "
             "range, not its best end. See Table 6 of the README.\nIt carries "
             "exploration noise and is not a deterministic score; fig5 gives "
             "that, and where the two disagree the reason is stated there.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------- fig 2: camera use (per seed)
def fig_camera(new_root, out, plt, np):
    """EVERY condition, every seed. No thresholded median anywhere."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.4), sharey=True)
    conds = CORRUPTIONS + ["const_action"]
    w = 0.15
    for ax, env in zip(axes, ENVS):
        seeds = []
        for si, tag in enumerate(seed_tags("state_plus_rgb")):
            d = load(new_root, env, tag)
            if d is None:
                continue
            dr = pixel_drops(d)
            seeds.append((si, tag.rsplit("seed", 1)[1], d, dr))
        if not seeds:
            ax.set_title(f"{ENV_TITLE[env]}\n(no runs)")
            continue
        for j, c in enumerate(conds):
            xs = [si + (j - (len(conds) - 1) / 2) * w for si, _s, _d, _dr in seeds]
            ys = [100 * dr.get(c, 0.0) for _si, _s, _d, dr in seeds]
            ax.bar(xs, ys, w * 0.92, color=CORR_COL[c],
                   hatch="//" if c == "const_action" else None,
                   edgecolor="white", linewidth=0.6,
                   label=CORR_LABEL[c] if ax is axes[0] else None, zorder=3)
            for x, y in zip(xs, ys):
                if abs(y) > 6:
                    ax.text(x, y + (1.6 if y > 0 else -4.6), f"{y:.0f}",
                            ha="center", va="bottom" if y > 0 else "top",
                            fontsize=6.8, rotation=90, zorder=4)
        # The pixels-only control: what an actor that MUST see looks like here.
        lab, tags, one_key = KNOWN_SEEING[env]
        ref = collect_old(env, tags)
        if ref:
            fr = [100 * pixel_drops(r).get("frozen_first", np.nan) for r in ref]
            m = float(np.nanmean(fr))
            ax.axhline(m, ls=":", lw=2, color=C_PIXEL, zorder=2)
            ax.text(-0.48, m + 1.5, f"pixels-only frozen_first {m:.0f}%"
                    f"{'' if one_key else '  (NOT a one-key contrast)'}",
                    fontsize=7.6, color=C_PIXEL, va="bottom", ha="left",
                    fontweight="bold")
        ax.axhline(30, ls="--", lw=1.6, color="#333333", zorder=2)
        ax.axhline(0, lw=0.9, color="#666666", zorder=2)
        ax.text(-0.48, 31, "old 30% verdict bar (median rule)", fontsize=7.6,
                color="#333333", va="bottom")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([
            f"seed {s}\nold verdict: "
            f"{'SEES' if d.get('actor_uses_pixels') else 'ignores'}\n"
            f"corrected: {'USES CAMERA' if uses_camera_corrected(d) else 'no use detected'}\n"
            f"pixel sens {sens_last(new_root, env, 'state_plus_rgb_seed' + s):.4f}"
            for _si, s, d, _dr in seeds], fontsize=8)
        ax.set_title(ENV_TITLE[env], fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of that run's own intact score lost when ONLY the\n"
                       "actor's image is corrupted (higher = more camera use)",
                       fontsize=9.5)
    axes[0].legend(fontsize=7.8, ncol=2, loc="upper left", framealpha=0.95)
    fig.suptitle("Camera use, condition by condition -- the 30% median rule "
                 "does not survive contact with the data\n"
                 f"state+RGB arms only, one group per seed, {MATCHED_BUDGET}, "
                 "nesy meta", fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, -0.10,
             "METRIC: fraction of that run's OWN intact score lost under one "
             "corruption, in %. BUDGET: 2.05M env steps. n = 3 seeds per env, "
             "5 evaluation episodes per bar; bars are per-seed, never averaged.\n"
             "THE OLD RULE: `median over {frozen_first, random_replay, zeros} > "
             "30%`. That bar was calibrated on `RGB_PROPRIO: none` actors, whose "
             "ONLY input is the camera -- corrupt it and nothing is left, which "
             "is why those arms score 94-99% (dotted line).\n"
             "These actors also hold the privileged state, so they can ride it "
             "through a corrupted frame and score far below 30% while still "
             "using the camera. `rgb_pixel_ablation.py` prints exactly that "
             "warning on every one of these runs; it was dropped from the first "
             "write-up.\nWALKER IS 3/3, NOT 2/3: holding the image at the t=0 "
             "frame costs 94.9% / 74.2% / 80.2% on the three seeds. The median "
             "rule labelled seed 0 'ignores' because its other two corruptions "
             "happened to be cheap.\n`const_action` is hatched because it is NOT "
             "a pixel-specific control for a state-reading actor: it removes the "
             "actor's state-driven variation too.",
             ha="center", va="top", fontsize=8.6,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def sens_last(root, env, tag):
    import numpy as np
    c = load_curve(root, env, tag, "pixel_sensitivity")
    return float(np.asarray(c, float)[-LAST_N:].mean()) if c else float("nan")


# ------------------------------------- fig 3: camera use vs performance
def fig_payoff(new_root, out, plt, np):
    """Association only, and now against the PRIMARY metric."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    any_pt = False
    for ax, env in zip(axes, ENVS):
        base = [train_return(new_root, env, t) for t in seed_tags("state_matched")]
        base = [v for v in base if v is not None]
        if base:
            bm = float(np.mean(base))
            bs = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
            ax.axhline(bm, color=C_STATE, lw=2.2, zorder=2)
            ax.axhspan(bm - bs, bm + bs, color=C_STATE, alpha=0.15, zorder=1)
            for v in base:
                ax.plot(-4, v, marker="_", ms=13, color=C_STATE, mew=2.5,
                        ls="none", zorder=3)
            ax.text(0.985, bm, f" state-only mean {bm:.1f} +/- {bs:.1f} (n={len(base)}) ",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8.5, color=C_STATE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_STATE))
        for tag in seed_tags("state_plus_rgb"):
            d = load(new_root, env, tag)
            y = train_return(new_root, env, tag)
            if d is None or y is None:
                continue
            any_pt = True
            x = 100 * pixel_drops(d).get("frozen_first", 0.0)
            sees = uses_camera_corrected(d) and not d.get("inconclusive")
            ax.scatter(x, y, s=250, color=C_SEES if sees else C_BLIND,
                       edgecolor="#222222", linewidth=1.4, zorder=5)
            ax.annotate(f"s{tag.rsplit('seed', 1)[1]}", (x, y),
                        textcoords="offset points", xytext=(0, -3), ha="center",
                        va="center", fontsize=8.5, color="white",
                        fontweight="bold", zorder=6)
        ax.axvline(30, ls="--", lw=1.8, color="#333333", zorder=1)
        ax.set_xlim(-10, 108)
        ax.set_xlabel("that seed's camera dependence: % lost when the image is\n"
                      "held at the t=0 frame (dashed = the old 30% bar)",
                      fontsize=9)
        ax.set_ylabel("PRIMARY train return (last 20 updates)", fontsize=9.5)
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
                      mec="#222222", label="state+RGB seed with no detected camera use"),
        mlines.Line2D([], [], color=C_STATE, lw=2.5,
                      label="state-only mean (band = +/-1 sample s.d. across its 3 seeds)"),
        mlines.Line2D([], [], marker="_", ls="none", ms=13, mec=C_STATE, mew=2.5,
                      label="individual state-only seeds"),
    ], loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.0),
        framealpha=0.95)
    fig.suptitle("Does USING the camera pay off?   one point per state+RGB seed, "
                 "against the state-only control of the same environment\n"
                 f"PRIMARY metric (train return, last 20 updates); nesy meta, "
                 f"{MATCHED_BUDGET}; n = 3 seeds per arm", fontsize=12.5, y=1.11)
    fig.text(0.5, -0.06,
             "METRIC: y is the primary train return (last-20-update mean over "
             "128 envs); x is the % of intact performance lost when the actor's "
             "image is held at the t=0 stack. BUDGET 2.05M steps, n=3 seeds. "
             f"{SD_NOTE}\nx uses `frozen_first` rather than the old median of "
             "three corruptions, because the median mixes conditions that "
             "disagree: on walker seed 0 the same run scores 94.9% here and "
             "4.8% on a wrong-timestep frame.\nUnits differ between panels, so "
             "heights are comparable only against each panel's own blue line. "
             "This is an ASSOCIATION over three seeds per environment, not a "
             "causal effect of seeing on performance.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------- fig 4: pixel sensitivity during training
def fig_sensitivity(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    found = False
    for ax, env in zip(axes, ENVS):
        for tag in seed_tags("state_plus_rgb"):
            c = load_curve(new_root, env, tag, "pixel_sensitivity")
            d = load(new_root, env, tag)
            if not c:
                continue
            found = True
            seed = tag.rsplit("seed", 1)[1]
            sees = bool(d and uses_camera_corrected(d)
                        and not d.get("inconclusive"))
            col = C_SEES if sees else C_BLIND
            y = np.asarray(c, float)
            lab = f"seed {seed} -- last {LAST_N} upd = {y[-LAST_N:].mean():.4f}"
            ax.plot(np.arange(y.size), y, color=col, lw=0.7, alpha=0.25)
            ax.plot(np.arange(y.size), smooth(y), color=col, lw=2.2, label=lab)
        # Pixels-only reference: an actor that has no choice but to see.
        lab_r, tags_r, one_key = KNOWN_SEEING[env]
        refs = [v for v in (old_sens(env, t) for t in tags_r) if v is not None]
        if refs:
            m = float(np.mean(refs))
            ax.axhline(m, ls=":", lw=2.2, color=C_PIXEL)
            ax.text(4, m * 1.12, f"pixels-only reference {m:.3f}"
                    f"{'' if one_key else '  (not a one-key contrast)'}",
                    fontsize=8, color=C_PIXEL, va="bottom", fontweight="bold")
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
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    if not found:
        raise SystemExit("sensitivity: no curves found")
    fig.suptitle("CO-PRIMARY camera-use measure: the trainer's own online probe, "
                 "taken WITHOUT any corruption\n"
                 "state+RGB arms, one line per seed "
                 f"({SMOOTH_W}-update rolling mean, raw faint behind); "
                 f"{MATCHED_BUDGET}", fontsize=12.5, y=1.05)
    fig.text(0.5, -0.06,
             "METRIC: `train/rgb/pixel_sensitivity`, how far the actor's output "
             "moves when its pixel input changes, logged every update during "
             "training. BUDGET 2.05M steps, n = 3 seeds, legend quotes each "
             f"seed's last-{LAST_N}-update mean.\nThis is an INDEPENDENT measure "
             "of camera use: it is open-loop, measured during training, and does "
             "not come from the end-of-training corruption rollouts at all. It "
             "is shown as a co-primary because the corruption verdict was "
             "contested.\nIT AGREES WITH THE CORRECTED READING AND NOT WITH THE "
             "OLD ONE. Walker's three seeds read 0.098 / 0.099 / 0.117 -- "
             "indistinguishable -- so the 'seed 0 is blind, seeds 1 and 2 see' "
             "split was an artifact of the median rule.\nCheetah reads "
             "0.0007-0.0019 and cartpole 0.003-0.026 on the same scale. Colour "
             "encodes the CORRECTED verdict (any single corruption costing >30%).",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------- fig 5: summary bars
def fig_bars(new_root, out, plt, np, eval30_root=None):
    """Every metric side by side: primary on top, then the two eval samples."""
    # The 30-episode row is the empirical resolution of the metric dispute:
    # the SAME frozen policies, more evaluation, and walker's eval ranges stop
    # overlapping. It is only drawn if the full 18-arm sweep exists.
    have30 = bool(eval30_root) and all(
        (Path(eval30_root) / e / f"{a}_seed{s}" / "pixel_ablation.json").exists()
        for e in ENVS for a in NEW_ARMS for s in (0, 1, 2))
    rows = [("PRIMARY: train return (last 20 of 250 updates, 128 envs)", "train"),
            ("SECONDARY: deterministic eval, 5 episodes", "eval5")]
    if have30:
        rows.append(("SECONDARY: SAME weights, 30 episodes (no retraining)",
                     "eval30"))
    fig, axes = plt.subplots(len(rows), 3, figsize=(15.5, 5.1 * len(rows)))
    axes = np.atleast_2d(axes)
    any_data = False
    for ri, (rowtitle, kind) in enumerate(rows):
        for ci, env in enumerate(ENVS):
            ax = axes[ri][ci]
            bars, labels, colors, errs, seedvals = [], [], [], [], []
            ylab = None
            for arm, (lab, col) in NEW_ARMS.items():
                if kind == "train":
                    vals = [train_return(new_root, env, t) for t in seed_tags(arm)]
                    vals = [v for v in vals if v is not None]
                    ylab = "episode return during training"
                else:
                    root = new_root if kind == "eval5" else Path(eval30_root)
                    n_ep = 5 if kind == "eval5" else 30
                    runs = collect(root, env, seed_tags(arm))
                    vals = [metric_info(r)[2] for r in runs]
                    ylab = (f"{metric_info(runs[0])[1]}  ({n_ep} eval episodes)"
                            if runs else "score")
                if not vals:
                    continue
                bars.append(float(np.mean(vals)))
                errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                labels.append(lab.replace(" (", "\n("))
                colors.append(col)
                seedvals.append(vals)
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
            ax.set_ylim(0, top * 1.62)
            for i, sv in enumerate(seedvals):
                for j, val in enumerate(sv):
                    xs = i + 0.16 + 0.075 * j
                    ax.plot(xs, val, marker="o", ms=7, mfc="white",
                            mec="#222222", mew=1.5, ls="none", zorder=6)
                    ax.annotate(f"s{j}", (xs, val), textcoords="offset points",
                                xytext=(0, 9), ha="center", va="bottom",
                                fontsize=7.5, color="#333333", zorder=6)
            for i, (v, e) in enumerate(zip(bars, errs)):
                ax.text(i, v + e + top * 0.03, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=11.5, fontweight="bold")
                ax.text(i, v + e + top * 0.115, "n=3 seeds", ha="center",
                        va="bottom", fontsize=8, color="#555555")
            if len(bars) == 2 and all(len(s) > 1 for s in seedvals):
                st = stats_pair(seedvals[0], seedvals[1])
                w = int(np.argmax(bars))
                wl = "state only" if w == 0 else "state + RGB"
                msg = (f"{wl} higher by {abs(bars[1] - bars[0]):.3f}"
                       f"   |   {st['wins']}/3 seeds to state+RGB\n"
                       f"Welch p={st['welch_p']:.3f}, d={st['d']:+.2f}; "
                       f"paired-by-seed p={st['paired_p']:.3f}")
                msg += ("\nseed ranges OVERLAP" if st["overlap"]
                        else "\nSEED RANGES DO NOT OVERLAP")
                ax.text(0.5, 0.995, msg, transform=ax.transAxes, ha="center",
                        va="top", fontsize=8.6, fontweight="bold",
                        color=colors[w] if not st["overlap"] else "#555555")
                if not st["overlap"]:
                    b[w].set_edgecolor("#333333"); b[w].set_linewidth(2.2)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
            ax.set_ylabel(ylab, fontsize=9.5)
            ax.set_title(f"{ENV_TITLE[env]}" if ri == 0 else "", fontsize=12.5,
                         fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        axes[ri][0].annotate(rowtitle, xy=(-0.22, 0.5),
                             xycoords="axes fraction", rotation=90,
                             ha="center", va="center", fontsize=10.5,
                             fontweight="bold")
    if not any_data:
        raise SystemExit("bars: no runs found")
    fig.suptitle("The metrics side by side -- where they disagreed, the reason "
                 "was resolution, and more evaluation resolves it\n"
                 f"nesy meta, {MATCHED_BUDGET}; each pair differs in ONE config "
                 "key (RGB_ACTOR); n = 3 seeds per bar", fontsize=13, y=1.0)
    tail = ("\nTHIRD ROW is the SAME frozen policies re-scored at 30 episodes "
            "instead of 5 -- no retraining, only more evaluation. On WalkerWalk "
            "that alone takes the eval metric from overlapping seed ranges to "
            "NON-OVERLAPPING ones (+14.4%, 76 of 90 paired episodes, "
            "p < 0.001), onto the primary metric's side. The 5-episode row was "
            "not measuring something different; it could not see."
            if len(rows) == 3 else "")
    fig.text(0.5, -0.035,
             "TOP ROW (primary): trainer's episode return, mean over 128 envs, "
             "averaged over the last 20 of 250 updates. EVAL ROWS: "
             "deterministic evaluation of the FINAL weights -- upright "
             "fraction for cartpole, reward/step for walker and cheetah.\n"
             f"{SD_NOTE} Ringed dots are the individual seeds. Budget is 2.05M "
             "env steps in every bar; the eval rows differ only in how many "
             "episodes score the same weights.\nWALKER IS WHERE THE 5-EPISODE "
             "ROW DISAGREES, and the primary metric is the one to believe: its "
             "achieved power at this effect size and n=3 is 0.78 against the "
             "5-episode metric's 0.16." + tail + "\nThe evaluation scores the "
             "FINAL weights with no checkpoint selection, so a run that "
             "collapses in its last updates (cartpole seed 0 collapses in the "
             "final three) is scored after the collapse.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------ fig 6: all variants
def fig_full(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    for ax, env in zip(axes, ENVS):
        entries = []
        for arm, (lab, col) in NEW_ARMS.items():
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab, m, e, n, col, False))
        for lab, tags in OLD_VARIANTS.get(env, []):
            runs = collect_old(env, tags)
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
        ax.set_ylabel(f"{ylab}  (5-episode eval, intact)", fontsize=10)
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
                 f"SECONDARY metric (5-episode deterministic eval). All bars: "
                 f"{MATCHED_BUDGET}, nesy meta, identical evaluation protocol",
                 fontsize=13, y=1.10)
    fig.text(0.5, -0.02,
             "METRIC: the 5-episode deterministic eval, i.e. the SECONDARY "
             "metric -- the pixels-only variants have no matched training-return "
             "comparison, so this is the only axis they all share. BUDGET 2.05M "
             "env steps throughout.\n"
             f"{SD_NOTE} n = seeds; where n = 1 the bar is +/-1 sample s.d. "
             "across that seed's 5 evaluation episodes instead, which is a "
             "different quantity and is why n is printed on every bar.\n"
             "Units are consistent WITHIN each panel and stated in each panel "
             "title; they differ BETWEEN panels. Only the state-only / state+RGB "
             "pair is a controlled one-variable contrast -- the pixels-only "
             "variants differ from each other, and from the state+RGB arms, in "
             "more than one config key.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------- fig 7: budget gap
def fig_budget(new_root, out, plt, np):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, env in zip(axes, ENVS):
        entries = []
        for arm, lab, col in (("state_matched", "state only\n@ 2.05M", C_STATE),
                              ("state_plus_rgb", "state+RGB\n@ 2.05M", C_SPRGB)):
            runs = collect(new_root, env, seed_tags(arm))
            if runs:
                m, e, n, _ = aggregate(runs)
                entries.append((lab, m, e, n, col))
        runs = collect_old(env, LADDER_PIXEL[env])
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
        ax.set_ylabel(f"{ylab}  (5-episode eval, intact)", fontsize=10)
        ax.set_title(ENV_TITLE[env], fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle("The budget caveat, closed\n"
                 "Every state baseline previously cited was trained at 52.4M steps "
                 "while the RGB arms got 2.05M -- a 25x gap. These are all 2.05M.",
                 fontsize=12.5, y=1.05)
    fig.text(0.5, -0.09,
             "METRIC: 5-episode deterministic eval (the secondary metric; the "
             "pixels-only arms share no other axis with the pair). BUDGET: every "
             f"bar is 2.05M env steps. n = seeds. {SD_NOTE}\n"
             "No 52.4M bar is drawn: the 52.4M state baselines quoted in earlier "
             "prose have NO run artifact in `results/` (it contains only "
             "`results/rgb`), so that number cannot be verified and is OMITTED "
             "rather than estimated.\nThe matched-budget state arms above replace "
             "it as the baseline to cite.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FFF4E5", ec="#DDAA66"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------- fig 8: the encoder goes blind
def fig_blinding(new_root, out, plt, np):
    """Give the actor the state as well and the CNN stops learning to see."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    for ax, env in zip(axes, ENVS):
        lab_r, tags_r, one_key = KNOWN_SEEING[env]
        po_s, sp_s = [], []
        for t in tags_r:
            v = old_sens(env, t)
            if v is not None:
                po_s.append(float(v))
        for t in seed_tags("state_plus_rgb"):
            c = load_curve(new_root, env, t, "pixel_sensitivity")
            if c:
                sp_s.append(float(np.asarray(c, float)[-LAST_N:].mean()))
        if not (po_s and sp_s):
            ax.set_title(f"{ENV_TITLE[env]}\n(missing arms)")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        groups = [("pixels-only\n(RGB_PROPRIO: none)", po_s, C_PIXEL),
                  ("state + RGB\n(RGB_PROPRIO: full)", sp_s, C_SPRGB)]
        x = np.arange(2)
        means = [float(np.mean(g[1])) for g in groups]
        ax.bar(x, means, 0.5, color=[g[2] for g in groups],
               edgecolor="white", linewidth=1.2, zorder=3)
        for i, (_l, vals, _c) in enumerate(groups):
            for j, v in enumerate(vals):
                ax.plot(i + 0.17 + 0.07 * j, v, marker="o", ms=7, mfc="white",
                        mec="#222222", mew=1.5, ls="none", zorder=6)
            ax.text(i, means[i] * 1.35, f"{means[i]:.4f}", ha="center",
                    va="bottom", fontsize=11, fontweight="bold", zorder=6)
        ax.set_yscale("log")
        ax.set_ylim(min(min(po_s), min(sp_s)) * 0.25,
                    max(max(po_s), max(sp_s)) * 12)
        ratio = means[0] / means[1] if means[1] > 0 else float("inf")
        ax.annotate("", xy=(1, means[1] * 1.9), xytext=(1, means[0] * 0.55),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#C44E52"))
        ax.text(1.06, math.sqrt(means[0] * means[1]),
                f"x{ratio:.0f} less\nsensitive" if ratio >= 10 else
                f"x{ratio:.1f} less\nsensitive",
                fontsize=10.5, fontweight="bold", color="#C44E52", va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
        ax.set_ylabel("pixel sensitivity, last 20 updates  (log scale)",
                      fontsize=9.5)
        tt = ENV_TITLE[env] + ("   [one-key contrast]" if one_key
                               else "   [NOT one-key]")
        ax.set_title(tt, fontsize=12.5, fontweight="bold",
                     color="#222222" if one_key else "#888888")
        ax.grid(axis="y", alpha=0.25, which="both")
        ax.set_axisbelow(True)
        if not one_key:
            ax.text(0.5, 0.015, "pixels-only arm here ALSO has\n"
                    "RGB_AUX_STATE_COEF 1.0 + META_DECISION_INTERVAL 4\n"
                    "-> confounded, do not quote as one-key",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, color="#996600",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF4E5",
                              ec="#DDAA66"))
        # Walker HAS a one-key pixels-only run and it points the other way.
        # Drawn because omitting a counterexample would be the same kind of
        # selective reading this pass is correcting.
        v = old_sens(env, "nesy_blind")
        if env == "walker" and v is not None:
            v = float(v)
            ax.plot(0, v, marker="D", ms=11, mfc="#C44E52", mec="#222222",
                    mew=1.4, ls="none", zorder=7)
            ax.annotate("one-key pixels-only\n(nesy_blind, n=1): "
                        f"{v:.4f}\nLOWER than state+RGB --\nthe blinding does "
                        "NOT hold here",
                        xy=(0, v), xytext=(0.30, 0.62), textcoords="axes fraction",
                        fontsize=8, color="#C44E52", fontweight="bold",
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#C44E52"),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#C44E52"))
    fig.suptitle("Hand the actor the state vector as well, and its camera "
                 "encoder stops learning to see\n"
                 f"in-training pixel-sensitivity probe, {MATCHED_BUDGET}, nesy "
                 "meta, n = 3 seeds per bar (dots)", fontsize=13, y=1.03)
    fig.text(0.5, -0.05,
             "METRIC: `train/rgb/pixel_sensitivity`, the trainer's own probe of "
             "how far the actor's action moves when its image changes, averaged "
             f"over the last {LAST_N} of 250 updates. BUDGET 2.05M env steps on "
             "BOTH sides. n = 3 seeds; dots are the seeds.\n"
             "CHEETAH IS THE CLEAN CASE: both cheetah arms derive from "
             "`configs/cheetah_run_nesy.yaml` and the ablation script overrides "
             "NUM_ENVS and TOTAL_TIMESTEPS identically, so they differ in the "
             "actor's proprio width and nothing else.\nAt the same budget the "
             "pixels-only cheetah actor reaches 97-99% camera dependence; adding "
             "the state vector drops sensitivity by a factor of ~370 and leaves "
             "no measurable camera dependence at all. The camera is not merely "
             "unused -- a cheaper channel stops the encoder learning.\n"
             "Cartpole and walker are greyed because their pixels-only arms also "
             "carry an auxiliary pixel->state loss and a longer meta decision "
             "interval, so their ratios confound 'removed the state' with 'added "
             "a loss that forces sight'.",
             ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCCCCC"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


FIGS = {
    "curves": "fig1_headline_learning_curves.png",
    "camera": "fig2_camera_use_per_seed.png",
    "payoff": "fig3_camera_use_vs_performance.png",
    "sensitivity": "fig4_pixel_sensitivity_curves.png",
    "bars": "fig5_summary_bars.png",
    "full": "fig6_all_variants.png",
    "budget": "fig7_budget_gap.png",
    "blinding": "fig8_encoder_blinding.png",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", default=NEW_ROOT)
    ap.add_argument("--ref-json", default=REF_JSON,
                    help="distilled pixels-only reference values; figs "
                         "2/4/6/7/8 draw their reference lines and bars "
                         "from this instead of the removed ablation tree")
    ap.add_argument("--eval30-root", default=EVAL30_ROOT,
                    help="the 30-episode re-evaluation tree; the "
                         "extra row in fig5 is drawn only if all "
                         "18 arms are present there")
    ap.add_argument("--outdir", default="results/rgb/state_plus_rgb/figures")
    ap.add_argument("--figure", default="all", choices=["all"] + list(FIGS))
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new_root = Path(args.new_root)
    globals()["REF_JSON"] = args.ref_json
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    w = args.figure

    if w in ("all", "curves"):
        fig_curves(new_root, outdir / FIGS["curves"], plt, np)
    if w in ("all", "camera"):
        fig_camera(new_root, outdir / FIGS["camera"], plt, np)
    if w in ("all", "payoff"):
        fig_payoff(new_root, outdir / FIGS["payoff"], plt, np)
    if w in ("all", "sensitivity"):
        fig_sensitivity(new_root, outdir / FIGS["sensitivity"], plt, np)
    if w in ("all", "bars"):
        fig_bars(new_root, outdir / FIGS["bars"], plt, np,
                 eval30_root=args.eval30_root)
    if w in ("all", "full"):
        fig_full(new_root, outdir / FIGS["full"], plt, np)
    if w in ("all", "budget"):
        fig_budget(new_root, outdir / FIGS["budget"], plt, np)
    if w in ("all", "blinding"):
        fig_blinding(new_root, outdir / FIGS["blinding"], plt, np)


if __name__ == "__main__":
    main()
