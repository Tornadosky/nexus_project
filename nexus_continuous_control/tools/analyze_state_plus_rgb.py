"""Corrected statistical analysis of the state-only vs state+RGB campaign.

WHY THIS FILE EXISTS
--------------------
The campaign's first write-up quoted ONE metric -- `pixel_ablation.json ->
results.intact`, a 5-episode deterministic evaluation -- and binarised the
camera question with a single 30% threshold. An expert review found that both
choices changed conclusions:

1. THE HEADLINE METRIC THREW AWAY THE CAMPAIGN'S POWER AND INVERTED WALKER.
   The same runs also carry `training_curves.json -> curves.episode_return`,
   128 parallel envs x 250 updates. On the last-20-update mean of that curve
   WalkerWalk separates completely (189.5 +/- 1.8 vs 203.2 +/- 6.2, 3/3 seeds,
   +7.2%, Welch p = 0.054, d = 2.98) where the 5-episode eval showed heavily
   overlapping ranges. Achieved power at n = 3 is 0.78 on the training metric
   against 0.16 on the 5-episode one. `figures/README.md` already labelled the
   training curves "PRIMARY" while the headline table quoted the eval -- an
   internal inconsistency this file resolves in favour of the training metric.

2. THE VERDICT RULE MANUFACTURED A SEED DISAGREEMENT. `median over
   {frozen_first, random_replay, zeros} > 0.30` reported walker as 2/3 seeds
   using the camera. Per seed, `frozen_first` alone costs 94.9% / 74.2% /
   80.2%, and the independent in-training probe `pixel_sensitivity` reads
   0.098 / 0.099 / 0.117 -- indistinguishable across the three seeds. The
   disagreement was in the rule, not in the runs.

3. THE 30% BAR IS BIASED IN THIS REGIME AND THE TOOL SAYS SO. `rgb_pixel_
   ablation.py` prints a WARN on every `RGB_PROPRIO: full` run: the threshold
   was calibrated on `RGB_PROPRIO: none` actors and a small effect is expected
   even when the pixels matter. That caveat was dropped from the write-up.

4. AN EPISODE WAS EXCLUDED THAT SHOULD NOT HAVE BEEN. Evaluation reset keys are
   `PRNGKey(9000 + 97*episode + seed)` -- a function of (episode, seed) and NOT
   of the arm. Both arms therefore start every episode from the SAME initial
   state. `walker/state_matched_seed0` episode 4 (0.055) and
   `walker/state_plus_rgb_seed0` episode 4 (0.652) are the SAME initial state:
   the baseline fell over and the extension did not. That is a legitimate
   paired outcome, not contamination. Dropping it moves walker from +0.072 to
   +0.025, i.e. the exclusion erases the very effect under test. No episode is
   dropped here; the pairing is used instead.

5. THE s.d. CONVENTION WAS NEVER STATED. `np.std` defaults to ddof=0
   (population). At n = 3 that understates the spread by sqrt(3/2) = 1.22x.
   Everything below uses the SAMPLE s.d. (ddof=1) and says so in every table.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not modify any existing result JSON. Every number is recomputed from
the committed raw files and written to a NEW file, `corrected_analysis.json`,
which records the source path of every input. The original 5-episode JSONs and
their stored (uncorrected) verdicts stay exactly as they were produced, so the
correction is auditable rather than retroactive.

    python tools/analyze_state_plus_rgb.py
    python tools/analyze_state_plus_rgb.py --markdown /tmp/tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

ENVS = ["cartpole", "walker", "cheetah"]
ENV_TITLE = {"cartpole": "CartpoleBalance", "walker": "WalkerWalk",
             "cheetah": "CheetahRun"}
ARMS = ["state_matched", "state_plus_rgb"]
ARM_LABEL = {"state_matched": "state only", "state_plus_rgb": "state + RGB"}
SEEDS = (0, 1, 2)
CONDITIONS = ("intact", "frozen_first", "random_replay", "shuffle_frames",
              "zeros", "const_action")

NEW_ROOT = "results/rgb/state_plus_rgb"
EVAL30_ROOT = "results/rgb/state_plus_rgb_eval30"
OLD_ROOT = "results/rgb/ablation"

# Matched-budget pixels-only (`RGB_PROPRIO: none`) reference arms, used for the
# "does adding the state blind the encoder?" contrast.
#
# `clean` marks the ones whose config differs from the state+RGB arm in
# RGB_PROPRIO ALONE. Only cheetah has that at 3 seeds: both its arms derive
# from `configs/cheetah_run_nesy.yaml` and the ablation script overrides
# NUM_ENVS and TOTAL_TIMESTEPS identically, so the two differ in the proprio
# width and nothing else. cartpole/walker `nesy_fixed_*` additionally carry
# RGB_AUX_STATE_COEF: 1.0 and META_DECISION_INTERVAL: 4, so they are reported
# but must not be read as a one-key contrast. walker `nesy_blind` IS one-key
# (plain `walker_walk_nesy.yaml` with USE_RGB forced on) but is a single seed.
PIXELS_ONLY = {
    "cheetah": [("nesy_seed0", "nesy_seed1", "nesy_seed2"), True,
                "configs/cheetah_run_nesy.yaml (USE_RGB forced on; "
                "RGB_PROPRIO defaults to none)"],
    "walker": [("nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"),
               False, "configs/walker_walk_nesy_rgb_aux.yaml (ALSO has "
                      "RGB_AUX_STATE_COEF 1.0 and META_DECISION_INTERVAL 4)"],
    "cartpole": [("nesy_fixed_seed0", "nesy_fixed_seed1", "nesy_fixed_seed2"),
                 False, "configs/cartpole_balance_nesy_rgb_aux.yaml (ALSO has "
                        "RGB_AUX_STATE_COEF 1.0 and META_DECISION_INTERVAL 4)"],
}
PIXELS_ONLY_EXTRA = {
    "walker": [("nesy_blind",), True,
               "configs/walker_walk_nesy.yaml (USE_RGB forced on; RGB_PROPRIO "
               "defaults to none) -- one-key but a SINGLE seed"],
}

LAST_N = 20          # updates averaged at the end of the training curve
ALPHA = 0.05


# ------------------------------------------------------------------ statistics
def mean_sd(vals):
    """(mean, SAMPLE s.d. ddof=1, n). ddof=1 everywhere, deliberately."""
    import numpy as np
    v = np.asarray(vals, float)
    return (float(v.mean()),
            float(v.std(ddof=1)) if v.size > 1 else 0.0,
            int(v.size))


def welch(a, b):
    """Two-sided Welch t-test of b against a, plus Cohen's d and Hedges' g."""
    import numpy as np
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    with warnings.catch_warnings():
        # Fires when an arm's seeds are all identical (cartpole state-only is
        # exactly 1.0000 three times). That is a property of the data.
        warnings.simplefilter("ignore")
        t, p = stats.ttest_ind(b, a, equal_var=False)
    n1, n2 = a.size, b.size
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    sp = math.sqrt(sp2)
    d = float((b.mean() - a.mean()) / sp) if sp > 0 else float("inf")
    # Hedges' small-sample correction; at n=3 per arm it is a 15% shrink.
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return {"t": float(t), "p": float(p), "cohens_d": d,
            "hedges_g": d * J if math.isfinite(d) else d,
            "pooled_sd": float(sp),
            "degenerate_zero_variance": bool(sp == 0.0)}


def achieved_power(a, b, alpha=ALPHA):
    """Power of the two-sample t-test AT THE OBSERVED effect size and n.

    This is post-hoc power. It is reported for exactly one purpose: to show
    that the 5-episode metric could not have detected the effect the training
    metric shows, so 'p was large' on that metric is uninformative. It is NOT
    evidence about any hypothesis.
    """
    import numpy as np
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return None
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    if sp2 <= 0:
        return 1.0
    d = (b.mean() - a.mean()) / math.sqrt(sp2)
    nc = d * math.sqrt(n1 * n2 / (n1 + n2))
    df = n1 + n2 - 2
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def paired(a, b):
    """Paired analysis of b - a over matched units. No unit is dropped."""
    import numpy as np
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    n = diff.size
    out = {"n_pairs": int(n),
           "mean_diff": float(diff.mean()),
           "sd_diff_sample": float(diff.std(ddof=1)) if n > 1 else 0.0,
           "wins_b": int((diff > 0).sum()),
           "wins_a": int((diff < 0).sum()),
           "ties": int((diff == 0).sum()),
           "min_diff": float(diff.min()), "max_diff": float(diff.max())}
    if n > 1 and diff.std(ddof=1) > 0:
        t, p = stats.ttest_rel(b, a)
        out["paired_t"] = float(t)
        out["paired_p"] = float(p)
        out["cohens_dz"] = float(diff.mean() / diff.std(ddof=1))
        out["ci95_low"], out["ci95_high"] = (
            float(x) for x in stats.t.interval(
                0.95, n - 1, loc=diff.mean(),
                scale=diff.std(ddof=1) / math.sqrt(n)))
    else:
        out["paired_t"] = out["paired_p"] = out["cohens_dz"] = None
        out["ci95_low"] = out["ci95_high"] = None
    # Distribution-free companion: at n=3 (seed level) the t-test leans hard on
    # normality, and the sign test does not.
    if n > 0:
        k = out["wins_b"]
        m = out["wins_b"] + out["wins_a"]
        out["sign_test_p"] = (float(stats.binomtest(k, m, 0.5).pvalue)
                              if m > 0 else None)
    if n >= 6:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(b, a).pvalue)
            except ValueError:
                out["wilcoxon_p"] = None
    return out


# ------------------------------------------------------------------------- io
def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def metric_key_of(d):
    return d.get("metric_key") or (
        "upright_fraction_mean" if "upright_fraction_mean" in
        d["results"]["intact"] else "reward_per_step_mean")


def metric_label(env):
    return "upright fraction" if env == "cartpole" else "reward / step"


def episode_key(d):
    """Per-episode series matching this run's HEADLINE metric, if stored.

    The original 5-episode JSONs only stored `per_episode` (reward/step), so
    cartpole's per-episode upright fractions do not exist in them. The re-eval
    runs store `upright_fraction_per_episode` as well.
    """
    i = d["results"]["intact"]
    if metric_key_of(d) == "upright_fraction_mean":
        return ("upright_fraction_per_episode"
                if "upright_fraction_per_episode" in i else None)
    return "per_episode"


class Run:
    """One (env, arm, seed) arm, on every metric it carries."""

    def __init__(self, repo: Path, env, arm, seed):
        import numpy as np
        self.env, self.arm, self.seed = env, arm, seed
        self.tag = f"{arm}_seed{seed}"
        base = repo / NEW_ROOT / env / self.tag
        self.dir = base
        self.abl = _read(base / "pixel_ablation.json")
        self.curves = (_read(base / "training_curves.json") or {}).get("curves")
        self.abl30 = _read(repo / EVAL30_ROOT / env / self.tag /
                           "pixel_ablation.json")
        # POSIX separators regardless of where this is run, so the committed
        # JSON is identical whether generated on the pool or on Windows.
        self.src = {
            "eval5": (Path(NEW_ROOT) / env / self.tag / "pixel_ablation.json").as_posix(),
            "training_curves": (Path(NEW_ROOT) / env / self.tag /
                                "training_curves.json").as_posix(),
            "eval30": ((Path(EVAL30_ROOT) / env / self.tag /
                        "pixel_ablation.json").as_posix() if self.abl30 else None),
        }
        # PRIMARY metric: last-20-update mean of the trainer's episode return,
        # itself already a mean over 128 parallel envs.
        self.train_return = None
        self.n_updates = None
        if self.curves and self.curves.get("episode_return"):
            c = np.asarray(self.curves["episode_return"], float)
            self.n_updates = int(c.size)
            self.train_return = float(c[-LAST_N:].mean())
        self.pixel_sens = None
        if self.curves and self.curves.get("pixel_sensitivity"):
            self.pixel_sens = float(
                np.asarray(self.curves["pixel_sensitivity"], float)[-LAST_N:].mean())

    # -- eval accessors ----------------------------------------------------
    def eval_mean(self, which="eval5"):
        d = self.abl if which == "eval5" else self.abl30
        if d is None:
            return None
        return float(d["results"]["intact"][metric_key_of(d)])

    def eval_episodes(self, which="eval5"):
        """Per-episode series on the headline metric, or None if not stored."""
        d = self.abl if which == "eval5" else self.abl30
        if d is None:
            return None
        k = episode_key(d)
        return None if k is None else list(d["results"]["intact"][k])

    def eval_episodes_rps(self, which="eval5"):
        """Per-episode reward/step -- stored by every run, every env."""
        d = self.abl if which == "eval5" else self.abl30
        return None if d is None else list(d["results"]["intact"]["per_episode"])

    def n_episodes(self, which="eval5"):
        d = self.abl if which == "eval5" else self.abl30
        return None if d is None else int(d.get("episodes", 0))

    def conditions(self, which="eval5"):
        d = self.abl if which == "eval5" else self.abl30
        if d is None or d.get("state_only"):
            return None
        mk = metric_key_of(d)
        base = d["results"]["intact"][mk]
        out = {}
        for c in CONDITIONS:
            if c not in d["results"]:
                continue
            v = d["results"][c][mk]
            out[c] = {"value": float(v),
                      "drop_fraction": float((base - v) / max(abs(base), 1e-9))}
        return out


def load_all(repo: Path):
    return {(e, a, s): Run(repo, e, a, s)
            for e in ENVS for a in ARMS for s in SEEDS}


# ------------------------------------------------------------------- analysis
def compare(runs, env, getter, label, unit):
    """One env, one metric: both arms, seed-level stats, and the paired view."""
    a = [getter(runs[(env, "state_matched", s)]) for s in SEEDS]
    b = [getter(runs[(env, "state_plus_rgb", s)]) for s in SEEDS]
    if any(v is None for v in a + b):
        return None
    am, asd, an = mean_sd(a)
    bm, bsd, bn = mean_sd(b)
    res = {
        "metric": label, "unit": unit,
        "state_only": {"per_seed": a, "mean": am, "sd_sample": asd, "n": an},
        "state_plus_rgb": {"per_seed": b, "mean": bm, "sd_sample": bsd, "n": bn},
        "difference": bm - am,
        "relative_difference_pct": 100.0 * (bm - am) / abs(am) if am else None,
        "seed_ranges_overlap": bool(min(a) <= max(b) and min(b) <= max(a)),
        "welch": welch(a, b),
        "achieved_power_at_observed_effect": achieved_power(a, b),
        # Seeds are the independent replication unit, and they are matched:
        # seed s of one arm and seed s of the other share the config, the
        # initialisation stream and the environment-step budget.
        "paired_by_seed": paired(a, b),
        "sd_convention": "sample s.d., ddof=1",
    }
    return res


def paired_episodes(runs, env, which):
    """Episode-level pairing. Every episode is kept; none is excluded.

    The reset key `PRNGKey(9000 + 97*episode + seed)` is a function of episode
    and seed only, so episode e at seed s starts from the SAME initial state in
    both arms and the two are a matched pair by construction.
    """
    pairs = [(runs[(env, "state_matched", s)], runs[(env, "state_plus_rgb", s)])
             for s in SEEDS]
    # Decide the metric ONCE, for all seeds. Deciding per seed could silently
    # concatenate upright fractions from one seed with reward/step from
    # another -- different units in the same column.
    headline = all(r.eval_episodes(which) is not None
                   for pair in pairs for r in pair)
    pa, pb, tags = [], [], []
    for s, (ra, rb) in zip(SEEDS, pairs):
        if headline:
            ea, eb = ra.eval_episodes(which), rb.eval_episodes(which)
        else:
            # Fall back to reward/step, which every run stores. For cartpole
            # that is NOT the headline metric and the caller is told so.
            ea, eb = ra.eval_episodes_rps(which), rb.eval_episodes_rps(which)
        if ea is None or eb is None:
            return None
        n = min(len(ea), len(eb))
        pa += list(ea[:n]); pb += list(eb[:n])
        tags += [f"s{s}e{i}" for i in range(n)]
    if not pa:
        return None
    out = paired(pa, pb)
    out["pair_labels"] = tags
    out["state_only_episodes"] = pa
    out["state_plus_rgb_episodes"] = pb
    out["metric"] = (metric_label(env) if headline else "reward / step")
    out["is_headline_metric"] = headline
    out["metric_note"] = (
        "" if headline else
        "the 5-episode JSONs stored per-episode reward/step only, so the "
        "episode-level pairing for this env uses reward/step, not the "
        "headline upright fraction (the re-evaluation stores both)")
    out["pseudoreplication_warning"] = (
        "Episodes within a seed share one trained policy, so these n pairs are "
        "NOT n independent replications. This test answers 'for these "
        "particular policies, does the extension score higher from the same "
        "initial states', which is the right question for the excluded-episode "
        "dispute. For generalisation across training runs use paired_by_seed "
        "(n=3).")
    return out


def camera_use(runs, env, which="eval5"):
    """Every condition for every state+RGB seed, plus the independent probe."""
    rows = []
    for s in SEEDS:
        r = runs[(env, "state_plus_rgb", s)]
        d = r.abl if which == "eval5" else r.abl30
        if d is None:
            continue
        conds = r.conditions(which)
        drops = {c: conds[c]["drop_fraction"] for c in conds if c != "intact"}
        pix = [drops[c] for c in ("frozen_first", "random_replay", "zeros")
               if c in drops]
        rows.append({
            "seed": s,
            "intact": conds["intact"]["value"],
            "conditions": conds,
            "stored_verdict_actor_uses_pixels": d.get("actor_uses_pixels"),
            "stored_pixel_drop_median": d.get("pixel_drop_median"),
            "pixel_drop_max": max(pix) if pix else None,
            "pixel_drop_min": min(pix) if pix else None,
            "pixel_sensitivity_last20": r.pixel_sens,
            "rgb_proprio": d.get("rgb_proprio"),
            # The corrected reading: ANY pixel corruption that costs a large
            # fraction of performance proves the actor was using the camera,
            # because the corruption changes nothing else. Requiring the MEDIAN
            # to clear a bar calibrated on pixels-only actors cannot do that.
            "corrected_uses_camera": bool(pix and max(pix) > 0.30),
        })
    return rows


WINDOWS = (5, 10, 20, 30, 50, 100, 250)


def window_sensitivity(runs, env):
    """Is the primary result an artifact of averaging the LAST 20 updates?

    Choosing an end-of-training window is a researcher degree of freedom, so
    it has to be shown not to matter. Windows above ~50 stop measuring final
    performance and start measuring the whole learning trajectory, which is a
    different question; they are included so the crossover is visible rather
    than hidden.
    """
    import numpy as np
    out = []
    for N in WINDOWS:
        a, b = [], []
        for s in SEEDS:
            ra, rb = runs[(env, "state_matched", s)], runs[(env, "state_plus_rgb", s)]
            if not (ra.curves and rb.curves):
                return None
            a.append(float(np.asarray(ra.curves["episode_return"], float)[-N:].mean()))
            b.append(float(np.asarray(rb.curves["episode_return"], float)[-N:].mean()))
        a, b = np.asarray(a), np.asarray(b)
        w = welch(a, b)
        out.append({
            "last_n_updates": N,
            "state_only_mean": float(a.mean()), "state_only_sd_sample": float(a.std(ddof=1)),
            "state_plus_rgb_mean": float(b.mean()),
            "state_plus_rgb_sd_sample": float(b.std(ddof=1)),
            "difference": float(b.mean() - a.mean()),
            "relative_difference_pct": float(100 * (b.mean() - a.mean()) / abs(a.mean())),
            "wins_state_plus_rgb": int((b > a).sum()),
            "seed_ranges_overlap": bool(a.min() <= b.max() and b.min() <= a.max()),
            "welch_p": w["p"],
        })
    return out


def blind_encoder(repo: Path, runs, env):
    """Pixels-only vs state+RGB: what adding the state vector does to the CNN."""
    import numpy as np
    spec = PIXELS_ONLY.get(env)
    if spec is None:
        return None
    tags, clean, prov = spec
    po = []
    for t in tags:
        d = _read(repo / OLD_ROOT / env / t / "pixel_ablation.json")
        c = (_read(repo / OLD_ROOT / env / t / "training_curves.json") or {}).get("curves")
        if d is None:
            continue
        sens = (float(np.asarray(c["pixel_sensitivity"], float)[-LAST_N:].mean())
                if c and c.get("pixel_sensitivity") else None)
        mk = metric_key_of(d)
        base = d["results"]["intact"][mk]
        drops = {k: float((base - d["results"][k][mk]) / max(abs(base), 1e-9))
                 for k in CONDITIONS if k in d["results"] and k != "intact"}
        pix = [drops[k] for k in ("frozen_first", "random_replay", "zeros")
               if k in drops]
        po.append({"tag": t, "pixel_sensitivity_last20": sens,
                   "pixel_drop_median": d.get("pixel_drop_median"),
                   "pixel_drop_max": max(pix) if pix else None,
                   "intact": float(base),
                   "train_return_last20": (
                       float(np.asarray(c["episode_return"], float)[-LAST_N:].mean())
                       if c and c.get("episode_return") else d.get("final_train_return")),
                   "updates": d.get("updates"), "num_envs": d.get("num_envs")})
    sp = [runs[(env, "state_plus_rgb", s)] for s in SEEDS]
    sens_po = [r["pixel_sensitivity_last20"] for r in po
               if r["pixel_sensitivity_last20"] is not None]
    sens_sp = [r.pixel_sens for r in sp if r.pixel_sens is not None]
    out = {
        "one_key_contrast": bool(clean),
        "pixels_only_provenance": prov,
        "pixels_only": po,
        "state_plus_rgb_pixel_sensitivity": [r.pixel_sens for r in sp],
        "mean_pixel_sensitivity_pixels_only": (float(np.mean(sens_po))
                                               if sens_po else None),
        "mean_pixel_sensitivity_state_plus_rgb": (float(np.mean(sens_sp))
                                                  if sens_sp else None),
    }
    if sens_po and sens_sp and np.mean(sens_sp) > 0:
        out["sensitivity_ratio_pixels_only_over_state_plus_rgb"] = float(
            np.mean(sens_po) / np.mean(sens_sp))
    extra = PIXELS_ONLY_EXTRA.get(env)
    if extra:
        tags2, clean2, prov2 = extra
        rows = []
        for t in tags2:
            d = _read(repo / OLD_ROOT / env / t / "pixel_ablation.json")
            c = (_read(repo / OLD_ROOT / env / t / "training_curves.json") or {}).get("curves")
            if d is None:
                continue
            rows.append({
                "tag": t,
                "pixel_sensitivity_last20": (
                    float(np.asarray(c["pixel_sensitivity"], float)[-LAST_N:].mean())
                    if c and c.get("pixel_sensitivity") else None),
                "pixel_drop_median": d.get("pixel_drop_median"),
                "intact": d["results"]["intact"][metric_key_of(d)],
                "train_return_last20": d.get("final_train_return")})
        out["extra_one_key_single_seed"] = {"provenance": prov2, "runs": rows,
                                            "one_key_contrast": bool(clean2)}
    return out


# --------------------------------------------------------------------- render
def fmt_p(p):
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_t(t):
    return "n/a" if t is None else f"{t:+.2f}"


def markdown(an):
    """The tables. Every one states metric, budget, n and s.d. convention."""
    L = []
    add = L.append
    have30 = an["eval30_available"]

    add("### Table 1 -- both metrics side by side, every arm, every seed\n")
    add("PRIMARY = mean episode return over the LAST 20 of 250 training "
        "updates, itself a mean over 128 parallel environments "
        "(`training_curves.json -> curves.episode_return`).  ")
    add("SECONDARY = deterministic evaluation of the FINAL weights, `intact` "
        "condition (`pixel_ablation.json -> results.intact`); cartpole is "
        "scored by upright fraction, walker/cheetah by reward per step.  ")
    add("Budget: 250 updates x 128 envs x 64 steps = 2,048,000 env steps per "
        "arm. n = 3 seeds per arm.\n")
    head = ("| env | arm | seed | PRIMARY train return (last 20 upd) | "
            "SECONDARY eval, 5 ep | ")
    sep = "|---|---|---|---|---|"
    if have30:
        head += "SECONDARY eval, 30 ep | "
        sep += "---|"
    head += "pixel sensitivity (last 20 upd) |"
    sep += "---|"
    add(head)
    add(sep)
    for env in ENVS:
        for arm in ARMS:
            for s in SEEDS:
                r = an["runs"][f"{env}|{arm}|{s}"]
                row = (f"| {ENV_TITLE[env]} | {ARM_LABEL[arm]} | {s} "
                       f"| {r['train_return_last20']:.2f} "
                       f"| {r['eval5_mean']:.4f} ")
                if have30:
                    row += (f"| {r['eval30_mean']:.4f} "
                            if r.get("eval30_mean") is not None else "| n/a ")
                row += (f"| {r['pixel_sensitivity_last20']:.5f} |"
                        if r.get("pixel_sensitivity_last20") is not None
                        else "| n/a (no camera) |")
                add(row)

    add("\n### Table 2 -- per-environment summary on BOTH metrics\n")
    add("All +/- are SAMPLE s.d. (ddof=1) across the 3 seeds. Welch's t-test is "
        "two-sided on the 3 seed means; d is Cohen's d on the pooled s.d.; "
        "power is post-hoc at the observed effect and n=3, reported only to "
        "show what each metric could have detected.\n")
    add("| env | metric | state only | state + RGB | diff | rel | seeds "
        "won by state+RGB | ranges overlap? | Welch t | p | Cohen d | "
        "achieved power |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for env in ENVS:
        for key, name in an["metric_order"]:
            c = an["per_env"][env].get(key)
            if not c:
                continue
            w, ps = c["welch"], c["paired_by_seed"]
            pw = c["achieved_power_at_observed_effect"]
            dg = " (degenerate: one arm has zero seed variance)" if \
                w["degenerate_zero_variance"] else ""
            rel = ("n/a" if c["relative_difference_pct"] is None
                   else f"{c['relative_difference_pct']:+.1f}%")
            dstr = ("inf" if not math.isfinite(w["cohens_d"])
                    else f"{w['cohens_d']:+.2f}")
            ov = "YES" if c["seed_ranges_overlap"] else "**no overlap**"
            pws = "n/a" if pw is None else f"{pw:.2f}"
            add(f"| {ENV_TITLE[env]} | {name} "
                f"| {c['state_only']['mean']:.4f} +/- {c['state_only']['sd_sample']:.4f} "
                f"| {c['state_plus_rgb']['mean']:.4f} +/- {c['state_plus_rgb']['sd_sample']:.4f} "
                f"| {c['difference']:+.4f} | {rel} "
                f"| {ps['wins_b']}/3 | {ov} | {w['t']:+.2f} | {fmt_p(w['p'])}{dg} "
                f"| {dstr} | {pws} |")
    add("\nThe two metrics are not interchangeable and the table shows why: on "
        "WalkerWalk the primary metric separates the arms completely (no seed "
        "of one arm reaches any seed of the other) at power 0.78, while the "
        "5-episode eval overlaps heavily at power 0.16. A large p-value on the "
        "5-episode metric is a statement about that metric's resolution, not "
        "about the arms.")

    add("\n### Table 3 -- paired analysis (no episode excluded)\n")
    add("Evaluation reset keys are `PRNGKey(9000 + 97*episode + seed)`: they "
        "depend on the episode and the seed, NOT on the arm. Episode e at seed "
        "s therefore starts from the SAME initial state in both arms and the "
        "two are a matched pair. Nothing is dropped -- including "
        "`walker/state_matched_seed0` episode 4, whose partner in the other "
        "arm ran from the identical initial state.\n")
    add("| env | pairing | metric | n pairs | mean paired diff | 95% CI | "
        "state+RGB wins | paired t | paired p | sign-test p |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    footnotes = []
    for env in ENVS:
        for which, lab in (("eval5", "episodes (5/seed)"),
                           ("eval30", "episodes (30/seed)")):
            pe = an["paired_episodes"][env].get(which)
            if not pe:
                continue
            ci = ("n/a" if pe.get("ci95_low") is None
                  else f"[{pe['ci95_low']:+.4f}, {pe['ci95_high']:+.4f}]")
            mark = "" if pe["is_headline_metric"] else " *"
            if not pe["is_headline_metric"] and pe["metric_note"] not in footnotes:
                footnotes.append(pe["metric_note"])
            add(f"| {ENV_TITLE[env]} | {lab} | {pe['metric']}{mark} "
                f"| {pe['n_pairs']} | {pe['mean_diff']:+.4f} | {ci} "
                f"| {pe['wins_b']}/{pe['n_pairs']} | {fmt_t(pe['paired_t'])} "
                f"| {fmt_p(pe['paired_p'])} | {fmt_p(pe.get('sign_test_p'))} |")
        c = an["per_env"][env].get("train_return")
        if c:
            ps = c["paired_by_seed"]
            ci = ("n/a" if ps.get("ci95_low") is None
                  else f"[{ps['ci95_low']:+.3f}, {ps['ci95_high']:+.3f}]")
            add(f"| {ENV_TITLE[env]} | **SEEDS (n=3, the independent unit)** "
                f"| train return | {ps['n_pairs']} | {ps['mean_diff']:+.3f} "
                f"| {ci} | {ps['wins_b']}/3 | {fmt_t(ps['paired_t'])} "
                f"| {fmt_p(ps['paired_p'])} | {fmt_p(ps.get('sign_test_p'))} |")
    for f in footnotes:
        add(f"\n\\* {f}")
    add("\nEpisode-level pairs within a seed share one trained policy, so they "
        "are not independent replications: those rows answer *did this policy "
        "do better from the same start*, which is exactly the question the "
        "excluded episode raised. The SEEDS row is the one that speaks to a "
        "new training run, and it is the row to quote for a general claim.")

    add("\n### Table 4 -- camera use: ALL conditions, not one thresholded median\n")
    add("Each cell is the fraction of that run's own `intact` score lost when "
        "only the ACTOR's image is corrupted (the meta-policy always sees the "
        "real state). `pixel sens` is the trainer's independent in-training "
        "probe, averaged over the last 20 updates -- a different measurement "
        "of the same question, taken without any corruption at all.\n")
    add("The stored verdict thresholds the MEDIAN of {frozen_first, "
        "random_replay, zeros} at 30%. That bar was calibrated on "
        "`RGB_PROPRIO: none` actors, whose only input is the camera, so a "
        "corruption there removes everything and those arms score 94-99%. A "
        "`RGB_PROPRIO: full` actor also holds the privileged state and can "
        "ride it through a corrupted frame, so it is EXPECTED to score under "
        "30% even when it is genuinely using the camera. `rgb_pixel_ablation."
        "py` prints that warning on every one of these runs. The corrected "
        "reading asks instead whether ANY single corruption -- which changes "
        "nothing but the actor's image -- costs more than 30%.\n")
    for which, eps in (("eval5", 5), ("eval30", 30)):
        if not any(an["camera_use"][e].get(which) for e in ENVS):
            continue
        add(f"\n**{eps} evaluation episodes per condition.**\n")
        add("| env | seed | intact | frozen_first | random_replay | "
            "shuffle_frames | zeros | const_action | pixel sens | stored "
            "verdict (30% median) | corrected reading |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for env in ENVS:
            for row in an["camera_use"][env].get(which) or []:
                c = row["conditions"]

                def g(k, _c=c):
                    return (f"{100 * _c[k]['drop_fraction']:+.1f}%"
                            if k in _c else "n/a")
                add(f"| {ENV_TITLE[env]} | {row['seed']} | {row['intact']:.4f} "
                    f"| {g('frozen_first')} | {g('random_replay')} "
                    f"| {g('shuffle_frames')} | {g('zeros')} "
                    f"| {g('const_action')} "
                    f"| {row['pixel_sensitivity_last20']:.5f} "
                    f"| {'SEES' if row['stored_verdict_actor_uses_pixels'] else 'ignores'} "
                    f"({100 * row['stored_pixel_drop_median']:+.1f}% median) "
                    f"| {'USES CAMERA' if row['corrected_uses_camera'] else 'no camera use detected'} |")
    if any(an["camera_use"][e].get("eval30") for e in ENVS):
        add("\nThe 30-episode block is the higher-power version of the same "
            "measurement, on the SAME frozen weights. It does not change any "
            "verdict, corrected or stored: walker is still the only "
            "environment where a single corruption costs more than 30%, the "
            "old median rule still labels walker seed 0 'ignores', and "
            "cartpole and cheetah are still at the noise floor.")
    def _row_vals(which, cond):
        out = []
        for row in an["camera_use"]["walker"].get(which) or []:
            c = row["conditions"]
            out.append(f"{100 * c[cond]['drop_fraction']:.1f}%"
                       if cond in c else "n/a")
        return " / ".join(out)

    ff5, ff30 = _row_vals("eval5", "frozen_first"), _row_vals("eval30", "frozen_first")
    add("\nRead across the walker rows: `frozen_first` -- holding the image at "
        "the t=0 stack, changing nothing else -- costs "
        f"{ff5} on the three seeds at 5 episodes"
        + (f" and {ff30} at 30." if ff30 else ".")
        + " An actor that ignored its camera could not lose two thirds of its "
        "performance to a change in that camera. Walker is 3/3, not 2/3, and "
        "the disagreement the median produced was an artifact of the rule -- "
        "at 30 episodes the same rule still calls seed 0 'ignores'. The "
        "in-training probe agrees and is not derived from these rollouts at "
        "all: 0.098 / 0.099 / 0.117 across the three walker seeds -- "
        "indistinguishable -- against 0.0007-0.0019 on cheetah and "
        "0.003-0.026 on cartpole.")
    add("\nWhat the walker columns do NOT support is a claim that the actor "
        "decodes the pose from the image. At 5 episodes `random_replay` "
        f"(real frames from the wrong timestep) costs {_row_vals('eval5', 'random_replay')} "
        f"and `zeros` costs {_row_vals('eval5', 'zeros')}"
        + (f"; at 30 episodes {_row_vals('eval30', 'random_replay')} and "
           f"{_row_vals('eval30', 'zeros')}" if ff30 else "")
        + ". Both stay erratic across seeds at both episode counts while the "
        "constant t=0 frame is reliably fatal, so this is not a sampling "
        "artifact. The dependence that replicates on every seed is on the "
        "image CHANGING, not on it being correct.")

    add("\n### Table 5 -- does adding the state vector blind the encoder?\n")
    add("Same 2.048M-step budget on both sides. `pixels-only` is "
        "`RGB_PROPRIO: none` (the actor has no state input at all); "
        "`state+RGB` is `RGB_PROPRIO: full`. `pixel sens` is the in-training "
        "probe (mean over seeds of each seed's last-20-update mean).\n")
    add("| env | one-key contrast? | pixels-only pixel sens | state+RGB pixel "
        "sens | ratio | pixels-only median drop | provenance |")
    add("|---|---|---|---|---|---|---|")
    import numpy as np
    for env in ENVS:
        be = an["blind_encoder"].get(env)
        if not be:
            continue
        po = be["mean_pixel_sensitivity_pixels_only"]
        sp = be["mean_pixel_sensitivity_state_plus_rgb"]
        ratio = be.get("sensitivity_ratio_pixels_only_over_state_plus_rgb")
        med = [r["pixel_drop_median"] for r in be["pixels_only"]
               if r["pixel_drop_median"] is not None]
        ok = ("YES" if be["one_key_contrast"] else
              "NO -- also differs in aux loss + META_DECISION_INTERVAL")
        rs = ("n/a" if ratio is None else
              (f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"))
        add(f"| {ENV_TITLE[env]} | {ok} | {po:.4f} | {sp:.5f} | {rs} "
            f"| {100 * float(np.mean(med)):.1f}% | {be['pixels_only_provenance']} |")
    add("\n**CheetahRun is the clean case and it is stark.** Both cheetah arms "
        "derive from `configs/cheetah_run_nesy.yaml` and the ablation script "
        "overrides `NUM_ENVS` and `TOTAL_TIMESTEPS` identically, so the two "
        "differ in the actor's proprio width and nothing else. At the same "
        "2.048M steps the pixels-only actor reaches 97-99% camera dependence "
        "and a sensitivity of 0.414; hand it the state vector as well and "
        "sensitivity falls to 0.0011, a factor of ~370, with no measurable "
        "camera dependence left. The camera is not merely unhelpful on "
        "cheetah -- giving the actor a cheaper channel stops the encoder "
        "learning at all. Cost of that blindness: pixels-only reaches an "
        "eval reward/step of 0.307 / 0.408 / 0.485 against state+RGB's "
        "0.438 / 0.508 / 0.509, so the sighted-but-weaker policy is behind, "
        "which is why the gradient prefers the state channel.")
    add("\nCARTPOLE AND WALKER ARE NOT ONE-KEY CONTRASTS and must not be "
        "quoted as if they were: their pixels-only arms additionally carry "
        "`RGB_AUX_STATE_COEF: 1.0` (an auxiliary pixel->state regression that "
        "exists precisely to force the encoder to see) and "
        "`META_DECISION_INTERVAL: 4`. Their ratios confound 'removed the "
        "state' with 'added an auxiliary sight loss'.")
    be = an["blind_encoder"].get("walker") or {}
    extra = be.get("extra_one_key_single_seed")
    if extra and extra["runs"]:
        r = extra["runs"][0]
        add(f"\nWALKER HAS A ONE-KEY POINT AND IT CUTS THE OTHER WAY. "
            f"`{OLD_ROOT}/walker/{r['tag']}` is plain "
            "`configs/walker_walk_nesy.yaml` with `USE_RGB` forced on, so it "
            "differs from the walker state+RGB arm in `RGB_PROPRIO` alone. "
            f"Its sensitivity is {r['pixel_sensitivity_last20']:.4f} -- LOWER "
            "than the state+RGB arms' 0.098-0.117, not higher -- while its "
            f"training return is only {r['train_return_last20']:.1f} against "
            "their 196-209. It is a single seed and so proves little on its "
            "own, but it is evidence that the blinding effect is "
            "environment-specific rather than a general law, and walker is "
            "the environment where it does not hold. That is consistent with "
            "walker being the one environment where the camera helps.")

    add("\n### Table 6 -- is the primary result an artifact of the averaging "
        "window?\n")
    add("Averaging the LAST 20 of 250 updates is a researcher degree of "
        "freedom, so it has to be shown not to carry the result. Windows of "
        "100 and 250 are included to show where the crossover is: beyond ~50 "
        "updates the average stops describing final performance and starts "
        "describing the whole learning trajectory, which is a different "
        "question.\n")
    add("| env | last N updates | state only | state + RGB | rel | seeds to "
        "state+RGB | ranges overlap? | Welch p |")
    add("|---|---|---|---|---|---|---|---|")
    for env in ENVS:
        for r in an["window_sensitivity"].get(env) or []:
            mark = " **<- reported**" if r["last_n_updates"] == LAST_N else ""
            ov = "YES" if r["seed_ranges_overlap"] else "**no**"
            add(f"| {ENV_TITLE[env]} | {r['last_n_updates']}{mark} "
                f"| {r['state_only_mean']:.2f} +/- {r['state_only_sd_sample']:.2f} "
                f"| {r['state_plus_rgb_mean']:.2f} +/- {r['state_plus_rgb_sd_sample']:.2f} "
                f"| {r['relative_difference_pct']:+.1f}% "
                f"| {r['wins_state_plus_rgb']}/3 | {ov} | {fmt_p(r['welch_p'])} |")
    add("\n**WalkerWalk survives this; the result is not window-dependent.** "
        "At every end-of-training window from 5 to 50 updates it is 3/3 "
        "seeds, NON-OVERLAPPING ranges, +6.2% to +8.1%, Welch p between 0.026 "
        "and 0.085. The reported +7.2% / p = 0.054 sits in the middle of that "
        "range and not at its favourable end -- the 5- and 10-update windows "
        "are stronger. It dissolves only at 100+, which no longer measures "
        "final performance.")
    add("\n**CheetahRun does NOT survive it, and the conclusion above is "
        "qualified accordingly.** The DIRECTION is stable: state-only is "
        "ahead at all seven windows. The magnitude and the seed count are "
        "not -- 0/3 seeds at the reported window of 20, but 1/3 at 5, 10, 30, "
        "50 and 100, with Welch p from 0.086 to 0.53. \"A small consistent "
        "cost of order 1-3%\" is supported. \"0/3 seeds, paired p = 0.029\" "
        "is a property of the last-20 window and must not be quoted alone.")
    add("\n**CartpoleBalance** is negative at every window, shrinking "
        "monotonically from -38.9% (last 5) to -7.0% (all 250) as the window "
        "widens to include the pre-collapse period. That gradient is itself "
        "the evidence that this is a LATE-TRAINING collapse and not a lower "
        "attainable ceiling.")
    return "\n".join(L)


# ----------------------------------------------------------------------- main
def build(repo: Path):
    runs = load_all(repo)
    # ALL-OR-NOTHING on the re-evaluation. A partially finished sweep would put
    # a column of mostly-"n/a" into the tables and silently compare arms with
    # different numbers of episodes, which is exactly the kind of quiet
    # inconsistency this pass exists to remove.
    n30 = sum(r.abl30 is not None for r in runs.values())
    have30 = n30 == len(runs)
    if 0 < n30 < len(runs):
        print(f"[note] {EVAL30_ROOT} has {n30}/{len(runs)} arms -- the "
              "30-episode re-evaluation is INCOMPLETE and is ignored here. "
              "Re-run this tool once tools/run_state_plus_rgb_eval30.sh has "
              "finished all 18 arms.")
        for r in runs.values():
            r.abl30 = None
            r.src["eval30"] = None
    metric_order = [("train_return", "PRIMARY train return (last 20 upd)"),
                    ("eval5", "SECONDARY eval, 5 episodes")]
    if have30:
        metric_order.append(("eval30", "SECONDARY eval, 30 episodes"))

    an = {
        "_what": "Corrected analysis of the state-only vs state+RGB campaign.",
        "_generated_by": "tools/analyze_state_plus_rgb.py",
        "_supersedes": (
            "the headline tables in results/rgb/state_plus_rgb/figures/README.md "
            "as first committed, which quoted the 5-episode eval only and "
            "binarised the camera verdict at a 30% median"),
        "_does_not_modify": (
            "No existing result JSON is written. Every source path is recorded "
            "per run under runs[*].sources."),
        "_primary_metric": (
            "curves.episode_return, mean of the last 20 of 250 updates, each "
            "update already a mean over 128 parallel envs"),
        "_secondary_metric": (
            "pixel_ablation.json results.intact, deterministic eval of the "
            "final weights"),
        "_sd_convention": "SAMPLE s.d. (ddof=1) everywhere in this file",
        "_budget": "250 updates x 128 envs x 64 steps = 2,048,000 env steps",
        "_seeds": list(SEEDS),
        "eval30_available": have30,
        "metric_order": metric_order,
        "runs": {}, "per_env": {}, "paired_episodes": {},
        "camera_use": {}, "blind_encoder": {}, "window_sensitivity": {},
    }
    for (e, a, s), r in runs.items():
        an["runs"][f"{e}|{a}|{s}"] = {
            "env": e, "arm": a, "seed": s,
            "train_return_last20": r.train_return,
            "train_curve_updates": r.n_updates,
            "eval5_mean": r.eval_mean("eval5"),
            "eval5_episodes": r.eval_episodes("eval5"),
            "eval5_episodes_reward_per_step": r.eval_episodes_rps("eval5"),
            "eval5_n_episodes": r.n_episodes("eval5"),
            "eval30_mean": r.eval_mean("eval30"),
            "eval30_episodes": r.eval_episodes("eval30"),
            "eval30_n_episodes": r.n_episodes("eval30"),
            "pixel_sensitivity_last20": r.pixel_sens,
            "sources": r.src,
        }
    for env in ENVS:
        an["per_env"][env] = {
            "train_return": compare(runs, env, lambda r: r.train_return,
                                    "training episode return (last 20 updates)",
                                    "env return"),
            "eval5": compare(runs, env, lambda r: r.eval_mean("eval5"),
                             "deterministic eval, 5 episodes", metric_label(env)),
        }
        if have30:
            an["per_env"][env]["eval30"] = compare(
                runs, env, lambda r: r.eval_mean("eval30"),
                "deterministic eval, 30 episodes", metric_label(env))
        an["paired_episodes"][env] = {
            "eval5": paired_episodes(runs, env, "eval5"),
            "eval30": paired_episodes(runs, env, "eval30") if have30 else None,
        }
        an["window_sensitivity"][env] = window_sensitivity(runs, env)
        an["camera_use"][env] = {"eval5": camera_use(runs, env, "eval5")}
        if have30:
            an["camera_use"][env]["eval30"] = camera_use(runs, env, "eval30")
        an["blind_encoder"][env] = blind_encoder(repo, runs, env)
    return an


README_HEAD = """\
# state-only vs state+RGB at matched budget

Numbers regenerated by `tools/analyze_state_plus_rgb.py`; figures by
`tools/plot_state_plus_rgb_figures.py`:

    python tools/analyze_state_plus_rgb.py --readme
    python tools/plot_state_plus_rgb_figures.py --figure all

## The question

The NEXUS paper suggests using RGB inputs for the skill agents. Earlier work
here tested RGB *replacing* the state (`RGB_PROPRIO: none`). This campaign
tests RGB *added to* the state (`RGB_PROPRIO: full`) against a state-only
control at the same environment-step budget -- the first honest
baseline-vs-extension comparison in this project.

## THIS IS A CORRECTED VERSION. What changed and why

An expert review of the first write-up found five reporting choices that
changed or overstated conclusions. Nothing was retrained; the correction is
entirely in which number is quoted and how it is read. A sixth issue,
sensitivity to the averaging window, was found while checking the rest and
is in Table 6.

1. **The headline metric was the low-power one, and it inverted WalkerWalk.**
   The table quoted `pixel_ablation.json -> results.intact`, a **5-episode**
   deterministic evaluation. The same runs also carry
   `training_curves.json -> curves.episode_return`: **128 parallel envs x 250
   updates**. On the last-20-update mean of that curve walker separates
   completely -- 189.51 +/- 1.80 vs 203.20 +/- 6.25, **3/3 seeds, no overlap,
   +7.2%, Welch p = 0.054, d = 2.98** -- where the 5-episode eval showed
   heavily overlapping ranges and p = 0.31. Achieved power at n = 3 is **0.78**
   on the training metric against **0.16** on the 5-episode one. The old
   `figures/README.md` already called the training curves "PRIMARY" while the
   headline table quoted the eval; that inconsistency is resolved here in
   favour of the training return, and both metrics are now always shown.

2. **The camera verdict was binarised by a rule that manufactured a seed
   disagreement.** `median over {frozen_first, random_replay, zeros} > 0.30`
   reported walker as 2/3 seeds. Per seed, `frozen_first` alone -- holding the
   image at the t=0 stack and changing nothing else -- costs **94.9% / 74.2% /
   80.2%**. The independent in-training probe reads **0.098 / 0.099 / 0.117**,
   indistinguishable across the three seeds (cheetah: 0.0007-0.0019; cartpole:
   0.003-0.026). **Walker is 3/3.** Every condition is now reported and the
   boolean is not quoted alone.

3. **The 30% threshold is biased in this regime and the tool says so.**
   `rgb_pixel_ablation.py` prints a WARN on every `RGB_PROPRIO: full` run: the
   threshold was calibrated on `RGB_PROPRIO: none` actors, whose only input is
   the camera, so a small effect is expected here even when the pixels matter.
   That caveat was dropped from the first write-up. It is now in the figures,
   in the tables, and in the emitted JSON as `verdict_caveat`.

4. **The "contaminated evaluation" exclusion was special pleading and has been
   removed.** Evaluation reset keys are `PRNGKey(9000 + 97*episode + seed)` --
   a function of (episode, seed), NOT of the arm. `walker/state_matched_seed0`
   episode 4 (0.055) and `walker/state_plus_rgb_seed0` episode 4 (0.652) ran
   from the **same initial state**: the baseline fell over and the extension
   did not. That is a legitimate paired outcome. Dropping it moved walker from
   +0.072 to +0.025, i.e. the exclusion erased the effect under test. No
   episode is dropped anywhere in this document; a paired analysis is reported
   instead.

5. **The s.d. convention was never stated.** The old tables used `np.std`,
   which defaults to ddof=0 (population). At n = 3 that understates spread by
   sqrt(3/2) = 1.22x. **Everything here uses the SAMPLE s.d. (ddof=1)** and
   says so in every table and every figure caption.

The original 5-episode JSONs and their stored verdicts are UNCHANGED on disk.
The corrected numbers live in `../corrected_analysis.json`, which records the
source path of every input, so the correction is auditable rather than
retroactive.

## How the comparison is kept honest

Both arms of every pair are generated from ONE base config by
`tools/gen_state_plus_rgb_configs.py`, which asserts the resolved configs
differ in exactly one key, `RGB_ACTOR`. That key gates only whether the skill
actors have a camera pathway; the ENVIRONMENT keeps `USE_RGB: true` in both
arms and still renders. This matters, because `USE_RGB` also switches the
task: MuJoCo Playground's CartpoleBalance keys `ctrl_dt`, `episode_length`,
the reward function and the termination rule on `vision`, and the vec wrapper
changes the actor's state vector. A `USE_RGB: false` baseline would differ in
the environment, the reward, the horizon and the state representation too.

Both arms are also scored by the same code: `rgb_pixel_ablation.py` runs the
same `rollout()`, the same metric keys and the same scoring loop for both. The
state-only arm scores the `intact` condition only, because pixel corruptions
are undefined without a pixel input.

## The two metrics

| | PRIMARY | SECONDARY |
|---|---|---|
| what | trainer's episode return, mean over the **last 20 of 250 updates**, each update already a mean over **128 parallel envs** | deterministic evaluation of the **final weights**, `intact` condition |
| source | `training_curves.json -> curves.episode_return` | `pixel_ablation.json -> results.intact` |
| units | environment return (same scale in both arms of a pair) | cartpole: upright fraction (0-1); walker/cheetah: reward per step |
| n | 3 seeds | 3 seeds x 5 episodes, and 3 seeds x 30 episodes from the re-evaluation |
| achieved power at n=3 on walker's observed effect | **0.78** | **0.16** at 5 episodes, **0.35** at 30 |
| carries | exploration noise | no exploration noise, but no checkpoint selection either |

Neither is perfect. The primary metric is the one quoted first because it is
the one with the resolution to answer the question; the secondary metric is
always shown beside it, and where they disagree the disagreement is the
finding, not something to average away.

**The re-evaluation shows the disagreement was resolution, not substance.**
The 18 pickled policies in `~/runs_spr/` were re-scored at 30 episodes
instead of 5 -- no retraining, only more evaluation
(`tools/run_state_plus_rgb_eval30.sh`, results in
`results/rgb/state_plus_rgb_eval30/`, the 5-episode JSONs untouched beside
them). On WalkerWalk that alone moves the eval metric from overlapping seed
ranges (+0.0717, p = 0.31) to **NON-OVERLAPPING** ones (+0.1027, +14.4%,
3/3 seeds, p = 0.17, and 76 of 90 paired episodes at p < 0.001) -- onto the
primary metric's side. The 5-episode eval was not measuring something
different; it could not see.

## The figures

| file | shows | does NOT claim |
|---|---|---|
| `fig1_headline_learning_curves.png` | **PRIMARY.** Episode return over training, per seed, with the last-20-update scoring window shaded and the full statistics in-panel. | It carries exploration noise and is not a deterministic score. fig5 gives that. |
| `fig2_camera_use_per_seed.png` | **Camera use, every condition, every seed** -- no thresholded median anywhere -- against the pixels-only reference. | It measures the ACTOR's pixel dependence only, not the hierarchy's. `const_action` is hatched: for a state-reading actor it is not a pixel-specific control. |
| `fig3_camera_use_vs_performance.png` | Each state+RGB seed's `frozen_first` dependence against its PRIMARY score, with the state-only control per env. | Association across three seeds per env, NOT a causal effect of seeing on performance. |
| `fig4_pixel_sensitivity_curves.png` | **CO-PRIMARY camera-use measure**: the trainer's own in-training probe, which uses no corruption at all, with the pixels-only reference level. | Open-loop sensitivity is not the same quantity as the closed-loop drop; it is shown as an independent second opinion because the corruption verdict was contested. |
| `fig5_summary_bars.png` | **Both metrics side by side**, primary on top and the 5-episode eval underneath, per-seed dots, sample s.d., Welch and paired-by-seed p. | Where seed ranges overlap it says so. |
| `fig6_all_variants.png` | Every committed variant plus the two new arms, on the secondary metric (the only axis they all share). | Only the state-only / state+RGB pair is a controlled contrast; the other variants differ in several keys. Units differ between panels. |
| `fig7_budget_gap.png` | That every bar is at the matched 2.05M budget. | The 52.4M state baseline is OMITTED, not estimated: it has no verifiable artifact. |
| `fig8_encoder_blinding.png` | **NEW.** Pixels-only vs state+RGB in-training pixel sensitivity: adding the state vector stops the encoder learning to see. | Only CheetahRun is a one-key contrast; cartpole and walker are greyed because their pixels-only arms also carry an aux pixel->state loss and a longer meta decision interval. |
"""

README_TAIL = """\

## Reading rules

* **s.d. is the SAMPLE s.d. (ddof=1)** everywhere in this document and in every
  figure. Older tables in git history used the population form and are 1.22x
  narrower at n = 3.
* Metrics differ by environment on the SECONDARY axis: CartpoleBalance uses the
  bounded upright fraction (0-1), computed geometrically from `qpos` and so
  independent of the env's reward function; WalkerWalk and CheetahRun use task
  reward per step. The PRIMARY axis is the environment return and is comparable
  within a pair but not across environments.
* `n` is the number of SEEDS unless a table says otherwise.
* The evaluation scores the FINAL weights with **no checkpoint selection**, so a
  training curve that collapses in its last updates produces an eval number
  that reflects the collapse. CartpoleBalance seed 0 is exactly this case.
* `const_action` is not a pixel-specific control for a `RGB_PROPRIO: full`
  actor: it removes the actor's state-driven variation too. Only the
  frozen/replay/blank conditions isolate the pixels.
* CartpoleBalance's base config uses `NUM_MINIBATCHES: 8` where walker and
  cheetah use 64, so cartpole takes **8x fewer gradient steps per env step**.
  It is identical in both arms, so it cannot confound state vs state+RGB, but
  cartpole is undertrained relative to the other two and its numbers are not
  comparable across environments.
* `RGB_AUGMENT: true` is set in both arms' config files, but DrQ random-shift
  augmentation is applied only in the RGB actor loss, so in practice **it runs
  in the state+RGB arm only**. It is a difference between the arms that
  `RGB_ACTOR` drags along with it.
* The input is a **64x64 GRAYSCALE 3-frame stack**, not colour. "RGB"
  throughout this project is a misnomer inherited from the flag name.

## Provenance

* Raw, unmodified: `../<env>/<arm>_seed<n>/pixel_ablation.json` (5 episodes)
  and `../<env>/<arm>_seed<n>/training_curves.json`.
* Higher-power re-evaluation of the SAME frozen policies from `~/runs_spr/`,
  30 episodes, no retraining: `results/rgb/state_plus_rgb_eval30/` (present
  only if that sweep has been run; `tools/run_state_plus_rgb_eval30.sh`).
* Recomputed statistics with every source path recorded:
  `../corrected_analysis.json`.
* Pixels-only reference arms: `results/rgb/ablation/<env>/`.
"""


def conclusions(an):
    """Per-environment conclusions, at the strength the numbers support."""
    L = ["\n## Per-environment conclusions\n"]
    L.append("The blanket claim that adding a camera \"does not improve "
             "performance\" is WITHDRAWN. It is not true of all three "
             "environments and it was reached on the low-power metric. The "
             "three environments split, and the split is the result.\n")

    def g(env, key):
        return an["per_env"][env][key]

    w = g("walker", "train_return")
    w5 = g("walker", "eval5")
    wp = w["paired_by_seed"]
    L.append("### WalkerWalk -- the camera helps, in 3/3 seeds, and the actor "
             "demonstrably uses it\n")
    L.append(
        f"On the primary metric state+RGB scores "
        f"{w['state_plus_rgb']['mean']:.2f} +/- {w['state_plus_rgb']['sd_sample']:.2f} "
        f"against the state-only control's {w['state_only']['mean']:.2f} +/- "
        f"{w['state_only']['sd_sample']:.2f} (sample s.d., n = 3). Every "
        f"state+RGB seed beats every state-only seed -- the ranges do not "
        f"overlap -- for {w['difference']:+.2f} "
        f"({w['relative_difference_pct']:+.1f}%), Welch "
        f"p = {w['welch']['p']:.3f}, d = {w['welch']['cohens_d']:+.2f}, "
        f"paired-by-seed p = {fmt_p(wp['paired_p'])}. "
        f"The 5-episode eval agrees in direction "
        f"({w5['difference']:+.4f}, {w5['paired_by_seed']['wins_b']}/3 seeds) "
        f"but cannot resolve it (p = {w5['welch']['p']:.2f}, achieved power "
        f"{w5['achieved_power_at_observed_effect']:.2f}).")
    w30 = an["per_env"]["walker"].get("eval30")
    if w30:
        pe30 = an["paired_episodes"]["walker"].get("eval30") or {}
        L.append(
            "\n**The re-evaluation settles the metric dispute empirically.** "
            "Re-scoring the SAME frozen policies at 30 episodes instead of 5 "
            "-- no retraining, only more evaluation -- moves the eval metric "
            f"onto the primary metric's side: {w30['state_only']['mean']:.4f} "
            f"+/- {w30['state_only']['sd_sample']:.4f} vs "
            f"{w30['state_plus_rgb']['mean']:.4f} +/- "
            f"{w30['state_plus_rgb']['sd_sample']:.4f}, "
            f"{w30['difference']:+.4f} ({w30['relative_difference_pct']:+.1f}%), "
            f"{w30['paired_by_seed']['wins_b']}/3 seeds and now "
            f"{'NON-OVERLAPPING' if not w30['seed_ranges_overlap'] else 'still overlapping'} "
            f"where 5 episodes overlapped. Paired over all "
            f"{pe30.get('n_pairs', 0)} episodes, state+RGB wins "
            f"{pe30.get('wins_b', 0)}/{pe30.get('n_pairs', 0)} at "
            f"p {fmt_p(pe30.get('paired_p'))}. The two metrics did not "
            "disagree about the world; the 5-episode one could not see.")
    L.append(
        "\nThe actor is not riding the state: holding its image at the t=0 "
        "frame costs 94.9% / 74.2% / 80.2% of performance on the three seeds, "
        "and the independent in-training probe reads 0.098 / 0.099 / 0.117 -- "
        "an order of magnitude above cartpole and two above cheetah.")
    L.append(
        "\nIt is not an artifact of the averaging window either (Table 6): at "
        "every end-of-training window from 5 to 50 updates it is 3/3 seeds "
        "with non-overlapping ranges, +6.2% to +8.1%, p between 0.026 and "
        "0.085. The reported window is in the middle of that range, not at "
        "its favourable end.")
    L.append(
        "\n**HOW STRONG IS THIS?** Underpowered. n = 3 seeds, p = 0.054 on the "
        "primary metric, and post-hoc power 0.78 means roughly a one-in-five "
        "chance of having missed it had it been real. It is the strongest "
        "result in the campaign and it still needs replication at more seeds "
        "before it should be called established. What it does rule out is the "
        "previous conclusion: this is not 'no effect'.")

    c = g("cheetah", "train_return")
    cp = c["paired_by_seed"]
    be = an["blind_encoder"]["cheetah"]
    L.append("\n### CheetahRun -- no benefit, a small consistent cost, and the "
             "encoder goes blind\n")
    L.append(
        f"State+RGB scores {c['state_plus_rgb']['mean']:.2f} +/- "
        f"{c['state_plus_rgb']['sd_sample']:.2f} against {c['state_only']['mean']:.2f} "
        f"+/- {c['state_only']['sd_sample']:.2f} on the primary metric: "
        f"{c['difference']:+.2f} ({c['relative_difference_pct']:+.1f}%), "
        f"**0/3 seeds** to the extension, paired-by-seed "
        f"p = {fmt_p(cp['paired_p'])} (Welch p = {c['welch']['p']:.3f}; the "
        "ranges do overlap, but every seed moves the same way). The "
        "5-episode eval shows nothing at all "
        f"({g('cheetah', 'eval5')['difference']:+.4f}, p = "
        f"{g('cheetah', 'eval5')['welch']['p']:.2f}).")
    c30 = an["per_env"]["cheetah"].get("eval30")
    if c30:
        pe30 = an["paired_episodes"]["cheetah"].get("eval30") or {}
        L.append(
            f"\nRe-scoring the same frozen policies at 30 episodes confirms "
            f"the eval metric's reading rather than overturning it: "
            f"{c30['difference']:+.4f} ({c30['relative_difference_pct']:+.1f}%), "
            f"Welch p = {c30['welch']['p']:.2f}, and paired over "
            f"{pe30.get('n_pairs', 0)} episodes "
            f"{pe30.get('wins_b', 0)}/{pe30.get('n_pairs', 0)} at "
            f"p {fmt_p(pe30.get('paired_p'))} -- as close to exactly nothing "
            "as this measurement gets. Cheetah is the one environment where "
            "both metrics, at both episode counts, agree.")
    L.append(
        "\n**Do not quote that 0/3 or that p = 0.029 on their own: they are "
        "specific to the last-20-update window.** Across the seven windows in "
        "Table 6 the DIRECTION is stable -- state-only is ahead at every one "
        "of them -- but the seed count is 1/3 at windows of 5, 10, 30, 50 and "
        "100 and Welch p runs from 0.086 to 0.53. What the data supports is "
        "**a small consistent cost of order 1-3%**; at this n it could as "
        "easily be the cost of the extra parameters as of the camera.")
    ratio = be.get("sensitivity_ratio_pixels_only_over_state_plus_rgb")
    L.append(
        f"\n**The finding that was missing from the write-up.** At the SAME "
        f"2.048M-step budget, the cheetah pixels-only arms "
        f"(`results/rgb/ablation/cheetah/nesy_seed{{0,1,2}}`, `RGB_PROPRIO: "
        f"none`) reach **97.3% / 99.3% / 98.1%** camera dependence and an "
        f"in-training sensitivity of "
        f"{be['mean_pixel_sensitivity_pixels_only']:.4f}. Hand the same actor "
        f"the state vector as well and sensitivity collapses to "
        f"{be['mean_pixel_sensitivity_state_plus_rgb']:.5f} -- a factor of "
        f"**{ratio:.0f}** -- with no measurable camera dependence left. This "
        "is a one-key contrast: both arms derive from "
        "`configs/cheetah_run_nesy.yaml` and the ablation script overrides "
        "`NUM_ENVS` and `TOTAL_TIMESTEPS` identically, so they differ in the "
        "actor's proprio width and nothing else. **Adding the state vector "
        "does not merely leave the camera unused; it stops the encoder "
        "learning to see.** Given a cheaper channel that already contains the "
        "answer, the policy gradient has no reason to pay for perception. "
        "See `fig8_encoder_blinding.png`.")
    L.append(
        "\nThe blindness is not free of cost to the pixels-only arm either: it "
        "scores 0.307 / 0.408 / 0.485 reward/step against state+RGB's "
        "0.438 / 0.508 / 0.509, which is exactly why the gradient prefers the "
        "state channel. And the effect is NOT a general law -- see the walker "
        "counterexample in Table 5.")

    cb = g("cartpole", "train_return")
    cb5 = g("cartpole", "eval5")
    L.append("\n### CartpoleBalance -- parity when training was stable; a "
             "stability failure, not a lower ceiling\n")
    b30 = an["per_env"]["cartpole"].get("eval30")
    s1_30 = (an["runs"]["cartpole|state_plus_rgb|1"].get("eval30_mean")
             if b30 else None)
    s1_txt = ("**seed 1 is exactly 1.0000, matching the baseline to four "
              "decimals**" if s1_30 is None else
              f"**seed 1 scores 1.0000 over 5 episodes and {s1_30:.4f} over "
              "30 -- parity with the baseline to within a single non-upright "
              "step in 7,500**")
    L.append(
        "State-only returns an upright fraction of exactly 1.0000 on all three "
        "seeds with zero spread, at 5 and at 30 episodes. State+RGB returns "
        "1.0000 / 0.6824 / 0.2864 over 5 episodes and "
        + (f"{an['runs']['cartpole|state_plus_rgb|0']['eval30_mean']:.4f} / "
           f"{s1_30:.4f} / "
           f"{an['runs']['cartpole|state_plus_rgb|2']['eval30_mean']:.4f} over "
           "30. " if b30 else "")
        + "Read the training curves before reading those numbers:")
    L.append(
        f"\n* {s1_txt}. When training was stable, the camera cost nothing.\n"
        "* **seed 0 sat at ~24.6 return for 240 updates and collapsed in the "
        "FINAL THREE** (24.42 -> 22.23 -> 15.19 -> 9.95). The evaluation "
        "scores the final weights with no checkpoint selection, so it scores "
        "the collapse: 0.6824.\n"
        "* **seed 2 collapsed at update ~150**, peaking at 23.34 and never "
        "recovering, and its eval reflects a genuinely broken policy: 0.2864.")
    L.append(
        f"\nSo the aggregate ({cb5['state_plus_rgb']['mean']:.4f} +/- "
        f"{cb5['state_plus_rgb']['sd_sample']:.4f} vs 1.0000 +/- 0.0000; "
        f"primary metric {cb['difference']:+.2f}, Welch "
        f"p = {cb['welch']['p']:.3f}) describes **late-training instability in "
        "2 of 3 seeds**, not a lower attainable performance. An unused CNN "
        "does not only waste compute; on a problem the baseline solved "
        "deterministically, it destabilised it.")
    L.append(
        "\nTwo caveats that must travel with any cartpole number. "
        "**`NUM_MINIBATCHES: 8` where walker and cheetah use 64**, so cartpole "
        "takes 8x fewer gradient steps per environment step -- it is the least "
        "trained of the three, which is a plausible contributor to the "
        "instability. And **DrQ random-shift augmentation runs only in the RGB "
        "arm**: `RGB_AUGMENT: true` is in both config files, but the "
        "augmentation is applied inside the RGB actor loss, so `RGB_ACTOR` "
        "drags it along. Neither breaks the one-key property of the config "
        "diff, but both mean 'the camera destabilised it' is one hypothesis "
        "among several.")
    L.append(
        "\nThe actor's camera dependence on cartpole is at the noise floor on "
        "all three seeds (largest single corruption 3.6%, in-training "
        "sensitivity 0.003-0.026), so whatever destabilised these runs, it was "
        "not the actor acting on what it saw.")

    L.append("\n### One-line summary\n")
    L.append("| env | camera used? | effect on the PRIMARY metric | strength |")
    L.append("|---|---|---|---|")
    L.append(f"| WalkerWalk | **YES, 3/3 seeds** | "
             f"**{w['relative_difference_pct']:+.1f}%, 3/3 seeds, no overlap** "
             f"| p = {w['welch']['p']:.3f} at n = 3; underpowered, but stable "
             "across every averaging window and the strongest result here |")
    L.append(f"| CheetahRun | no, 0/3 | small cost, order 1-3%, direction "
             "stable across all windows | 0/3 and paired p = "
             f"{fmt_p(cp['paired_p'])} hold at the last-20 window only; plus "
             f"the encoder is blinded ~{ratio:.0f}x |")
    L.append("| CartpoleBalance | no, 0/3 | parity on the 1 seed that stayed "
             "stable; 2/3 collapsed late | a stability finding, confounded by "
             "8x fewer gradient steps and RGB-only augmentation |")
    return "\n".join(L)


def write_readme(repo: Path, an, path: Path):
    body = "\n".join([README_HEAD, conclusions(an),
                      "\n## The numbers\n", markdown(an), README_TAIL])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/rgb/state_plus_rgb/corrected_analysis.json")
    ap.add_argument("--markdown", default=None,
                    help="also write the tables as markdown to this path")
    ap.add_argument("--readme", action="store_true",
                    help="rewrite results/rgb/state_plus_rgb/figures/README.md")
    ap.add_argument("--readme-path",
                    default="results/rgb/state_plus_rgb/figures/README.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo)
    an = build(repo)
    p = repo / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(an, indent=2))
    print("wrote", p)
    md = markdown(an)
    if args.markdown:
        Path(args.markdown).write_text(md)
        print("wrote", args.markdown)
    if args.readme:
        write_readme(repo, an, repo / args.readme_path)
    if not args.quiet:
        print()
        print(md)


if __name__ == "__main__":
    main()
