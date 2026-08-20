"""Summarise the shared-encoder campaign arms next to the committed baselines.

Reads, per arm tag:
  pixel_ablation.json    -> intact metric, per-condition drops, median drop,
                            verdict, inconclusive, final_train_return
  pixel_sensitivity.json -> per-skill relative_to_action_range (responsiveness)
  training_curves.json   -> max(aux_state_loss)  [INTEGRITY: must be exactly 0
                            in every no-aux arm] and the final pixel_sensitivity

Usage:  python _report_campaign.py [--root results/rgb/ablation] [tag ...]
"""
import argparse
import json
import pathlib
import sys

# The verdict rule the harness records: median drop over these three > 0.30.
VERDICT_CONDS = ("frozen_first", "random_replay", "zeros")

BASELINES = {
    "cartpole/nesy_blind": "committed BLIND anchor",
    "cartpole/nesy_blind_seed1": "restored BLIND seed1",
    "cartpole/nesy_blind_seed2": "restored BLIND seed2",
    "cartpole/nesy_fixed_seed0": "committed +FIX (aux+MDI4+LR)",
    "cartpole/nesy_fixed_seed1": "committed +FIX",
    "cartpole/nesy_fixed_seed2": "committed +FIX",
    "walker/nesy_blind": "committed BLIND anchor",
    "walker/nesy_fixed_seed0": "committed +FIX (aux+MDI4)",
    "walker/nesy_fixed_seed1": "committed +FIX",
    "walker/nesy_fixed_seed2": "committed +FIX",
}

CAMPAIGN = [
    ("walker/nesy_noaux_mdi4_seed0", "T0  aux OFF, MDI=4            (control)"),
    ("cartpole/nesy_noaux_mdi4_seed0", "T0  aux OFF, MDI=4            (control)"),
    ("walker/nesy_shared_noaux_mdi4_seed0", "T1a SHARED, aux OFF, MDI=4    *LOAD-BEARING*"),
    ("cartpole/nesy_shared_noaux_mdi4_seed0", "T1a SHARED, aux OFF, MDI=4    *LOAD-BEARING*"),
    ("walker/nesy_shared_seed0", "T1b SHARED only, MDI=1        (shortcut intact)"),
    ("cartpole/nesy_shared_seed0", "T1b SHARED only, MDI=1        (shortcut intact)"),
    ("walker/nesy_shared_metaz_noaux_seed0", "T2  SHARED + meta sees pixels"),
    ("cartpole/nesy_shared_metaz_noaux_seed0", "T2  SHARED + meta sees pixels"),
]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def arm(root, tag):
    d = root / tag
    ab = load(d / "pixel_ablation.json")
    if ab is None:
        return None
    se = load(d / "pixel_sensitivity.json") or {}
    tc = load(d / "training_curves.json") or {}
    curves = tc.get("curves", {})
    aux = curves.get("aux_state_loss") or []
    psens = curves.get("pixel_sensitivity") or []
    intact = (ab.get("results") or {}).get("intact", {})
    drops = ab.get("performance_drop_fraction") or {}
    skills = (se.get("actor_sensitivity") or {})
    return dict(
        tag=tag,
        env=ab.get("env"),
        seed=ab.get("seed"),
        updates=ab.get("updates"),
        config=tc.get("config"),
        upright=intact.get("upright_fraction_mean"),
        rew=intact.get("reward_per_step_mean"),
        drops=drops,
        median=ab.get("pixel_drop_median"),
        sees=ab.get("actor_uses_pixels"),
        strict=ab.get("actor_uses_pixels_strict"),
        incon=ab.get("inconclusive"),
        rule=ab.get("verdict_rule"),
        train_ret=ab.get("final_train_return"),
        aux_max=(max(aux) if aux else None),
        aux_nonzero=(sum(1 for x in aux if x != 0) if aux else None),
        psens_last=(psens[-1] if psens else None),
        psens_max=(max(psens) if psens else None),
        skills={k: (v or {}).get("relative_to_action_range") for k, v in skills.items()},
        responsive=se.get("actor_responsive"),
        render_alive=((se.get("render") or {}).get("alive")),
        hist=se.get("skill_histogram"),
    )


def verdict(a):
    if a["incon"]:
        return "INCONCLUSIVE"
    return "SEES" if a["sees"] else "BLIND"


def fmt(x, spec="%.4f"):
    return "n/a" if x is None else spec % x


def aux_expected_on(a):
    """Did this arm's own config ask for the aux loss? Read the config it recorded."""
    cfg = a.get("config")
    if not cfg:
        return None
    try:
        import yaml

        with open(cfg) as f:
            return float((yaml.safe_load(f) or {}).get("RGB_AUX_STATE_COEF", 0.0)) > 0.0
    except Exception:
        return None


def show(a, note=""):
    metric = "upright=%s" % fmt(a["upright"]) if a["upright"] is not None else ""
    print("  %-40s %s" % (a["tag"], note))
    print(
        "      intact: rew/step=%s %s | train_ret=%.2f | median_drop=%s -> %s"
        % (fmt(a["rew"]), metric, a["train_ret"] or 0, fmt(a["median"]), verdict(a))
    )
    print(
        "      drops:  "
        + "  ".join(
            "%s=%s%s" % (k, fmt(v, "%.3f"), "*" if k in VERDICT_CONDS else "")
            for k, v in a["drops"].items()
        )
    )
    if a["skills"]:
        print(
            "      per-skill responsiveness (pct of action range): "
            + "  ".join(
                "%s=%s" % (k, fmt(v * 100 if v is not None else None, "%.2f"))
                for k, v in a["skills"].items()
            )
        )
    # INTEGRITY: aux_state_loss must be EXACTLY 0 in an arm that turned aux off.
    # A nonzero curve in an aux-ON arm is correct and not a problem.
    on = aux_expected_on(a)
    if on is True:
        integ = "aux ON by config (expected nonzero)"
    elif a["aux_max"] == 0:
        integ = "INTEGRITY OK (exactly 0, aux off)"
    else:
        integ = "*** INTEGRITY FAIL: aux off in config but loss NONZERO -> ARM VOID ***"
    print(
        "      aux_state_loss max=%s nonzero_updates=%s  %s | pixel_sens last=%s max=%s"
        % (fmt(a["aux_max"], "%.6g"), a["aux_nonzero"], integ,
           fmt(a["psens_last"], "%.4f"), fmt(a["psens_max"], "%.4f"))
    )
    if a["render_alive"] is not None and not a["render_alive"]:
        print("      *** render NOT alive -- the camera produced dead frames ***")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/rgb/ablation")
    ap.add_argument("tags", nargs="*")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    if args.tags:
        groups = [("REQUESTED TAGS", [(t, "") for t in args.tags])]
    else:
        groups = [
            ("BASELINES (committed / restored)", list(BASELINES.items())),
            ("CAMPAIGN ARMS", CAMPAIGN),
        ]

    missing = []
    for title, pairs in groups:
        print("=" * 100)
        print(title)
        print("=" * 100)
        for tag, note in pairs:
            a = arm(root, tag)
            if a is None:
                missing.append(tag)
                continue
            show(a, note)
    if missing:
        print("NOT PRESENT YET (%d):" % len(missing))
        for t in missing:
            print("   ", t)

    # ---- the load-bearing comparisons, computed rather than eyeballed ----
    print("=" * 100)
    print("LOAD-BEARING COMPARISONS  (shared vs its one-flag aux-off control)")
    print("=" * 100)
    for env, ctrl, shared in (
        ("walker", "walker/nesy_noaux_mdi4_seed0", "walker/nesy_shared_noaux_mdi4_seed0"),
        ("cartpole", "cartpole/nesy_noaux_mdi4_seed0", "cartpole/nesy_shared_noaux_mdi4_seed0"),
    ):
        c, s = arm(root, ctrl), arm(root, shared)
        if c is None or s is None:
            print("  %-9s incomplete (control=%s shared=%s)"
                  % (env, "yes" if c else "MISSING", "yes" if s else "MISSING"))
            continue
        key = "upright" if c["upright"] is not None else "rew"
        print("  %-9s %-14s control=%s  shared=%s   delta=%s"
              % (env, key, fmt(c[key]), fmt(s[key]),
                 fmt((s[key] or 0) - (c[key] or 0), "%+.4f")))
        print("  %-9s %-14s control=%s  shared=%s   delta=%s"
              % ("", "median_drop", fmt(c["median"]), fmt(s["median"]),
                 fmt((s["median"] or 0) - (c["median"] or 0), "%+.4f")))
        print("  %-9s %-14s control=%s  shared=%s"
              % ("", "verdict", verdict(c), verdict(s)))
        print("  %-9s %-14s control=%s  shared=%s"
              % ("", "train_return", fmt(c["train_ret"], "%.2f"), fmt(s["train_ret"], "%.2f")))
        ck, sk = c["skills"], s["skills"]
        for k in sorted(set(ck) | set(sk)):
            print("  %-9s   skill %-18s control=%6s%%  shared=%6s%%"
                  % ("", k,
                     fmt((ck.get(k) or 0) * 100, "%.2f"),
                     fmt((sk.get(k) or 0) * 100, "%.2f")))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
