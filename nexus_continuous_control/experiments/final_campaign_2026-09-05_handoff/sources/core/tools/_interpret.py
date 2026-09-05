#!/usr/bin/env python3
"""Exact-statistics interpretation of the v2 matrix.

Written as a file, not an inline loop: the Bash tool's outer shell expands $VAR before the
string reaches WSL, so inline `for s in ...` runs with an empty variable.

Every comparison here is budget-matched unless explicitly labelled MISMATCHED. Reports, for
each contrast: n on both sides, the full seed lists, whether the ranges separate
(min(hier) > max(flat)), and an EXACT permutation p -- not a t-test, the samples are small
and non-normal. Where the same seed index exists on both sides, the seed-matched sign test
is reported too and led with, because the pairing is the strength of the design.
"""
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "runs/v2/v2_matrix.json").read_text(encoding="utf-8"))["matrix"]


def cell(env, arm):
    c = M.get(env, {}).get(arm)
    return c if c else None


def exact_perm_p(a, b):
    """One-sided exact permutation p on the difference of means (a > b).

    Enumerates every way of splitting the pooled sample into groups of size |a| and |b| and
    counts how many reach a mean gap at least as large as observed. Exact whenever the number
    of splits is tractable; falls back to reporting the count of splits when it is not.
    """
    n, m = len(a), len(b)
    pooled = list(a) + list(b)
    total = comb(n + m, n)
    if total > 2_000_000:
        return None, total
    obs = sum(a) / n - sum(b) / m
    idx = range(n + m)
    hit = 0
    s_all = sum(pooled)
    for pick in combinations(idx, n):
        sa = sum(pooled[i] for i in pick)
        d = sa / n - (s_all - sa) / m
        if d >= obs - 1e-12:
            hit += 1
    return hit / total, total


def sign_test_p(wins, losses):
    """Two-sided exact binomial sign test at p=0.5, reported one-sided for a directional claim."""
    n = wins + losses
    if n == 0:
        return None
    return sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def contrast(env, hier_arm, flat_arm, label="", mismatch_note=""):
    h, f = cell(env, hier_arm), cell(env, flat_arm)
    print("=" * 92)
    print(f"{env}   {hier_arm}  vs  {flat_arm}   {label}")
    print("=" * 92)
    if not h or not f:
        print(f"  MISSING: {hier_arm if not h else flat_arm} has not landed yet.\n")
        return
    hb, fb = h.get("budgets"), f.get("budgets")
    hsteps = hb[0][1] if hb else None
    fsteps = fb[0][1] if fb else None
    matched = hsteps == fsteps
    print(f"  budget      : hier {hsteps:,}  flat {fsteps:,}   "
          f"{'MATCHED' if matched else '*** MISMATCHED — NOT a clean result ***'}")
    if mismatch_note:
        print(f"                {mismatch_note}")
    print(f"  n           : hier {h['n']}   flat {f['n']}"
          f"{'   [hier n<=3 — PROVISIONAL]' if h['n'] <= 3 else ''}")
    print(f"  hier seeds  : {h['seeds']}")
    print(f"  hier success: {[round(x, 4) for x in h['success']]}")
    print(f"  flat seeds  : {f['seeds']}")
    print(f"  flat success: {[round(x, 4) for x in f['success']]}")
    hmin, hmax = min(h["success"]), max(h["success"])
    fmin, fmax = min(f["success"]), max(f["success"])
    print(f"  hier range  : [{hmin:.4f}, {hmax:.4f}]   mean {h['success_mean']:.4f}")
    print(f"  flat range  : [{fmin:.4f}, {fmax:.4f}]   mean {f['success_mean']:.4f}")
    sep = hmin > fmax
    print(f"  SEPARATED   : {'YES  min(hier) %.4f > max(flat) %.4f' % (hmin, fmax) if sep else 'NO  ranges overlap — a mean win inside overlap is not a win'}")
    p, total = exact_perm_p(h["success"], f["success"])
    if p is None:
        print(f"  exact perm p: not enumerated ({total:,} splits)")
    else:
        print(f"  exact perm p: {p:.3g}   (one-sided, {total:,} splits enumerated)")

    # Seed-matched: same seed index present on both sides.
    shared = sorted(set(h["seeds"]) & set(f["seeds"]))
    if shared:
        hs = dict(zip(h["seeds"], h["success"]))
        fs = dict(zip(f["seeds"], f["success"]))
        wins = sum(1 for s in shared if hs[s] > fs[s])
        losses = len(shared) - wins
        sp = sign_test_p(wins, losses)
        print(f"  SEED-MATCHED: {len(shared)} shared seed indices {shared}")
        print(f"                hier wins {wins}/{len(shared)}   exact sign test p={sp:.3g}")
        print("                (pairing is the strength here — stronger than two independent samples)")
        for s in shared:
            print(f"                  s{s:<3} hier {hs[s]:.4f}  flat {fs[s]:.4f}  "
                  f"{'+' if hs[s] > fs[s] else '-'}{abs(hs[s]-fs[s]):.4f}")
    print()


def zero_profile(env, arms, thresh=0.01):
    """All-or-nothing spread. A mean over a bimodal distribution describes no episode that occurred."""
    print("=" * 92)
    print(f"{env} — per-seed zero/collapse profile (bimodal; the mean describes no run that happened)")
    print("=" * 92)
    for arm in arms:
        c = cell(env, arm)
        if not c:
            print(f"  {arm:<22} MISSING")
            continue
        z = [s for s, v in zip(c["seeds"], c["success"]) if v < thresh]
        nz = [v for v in c["success"] if v >= thresh]
        steps = c["budgets"][0][1] if c.get("budgets") else None
        print(f"  {arm:<22} n={c['n']:<3} budget={steps:>12,}  zeros {len(z)}/{c['n']}"
              f"  zero-seeds {z}")
        if nz:
            print(f"  {'':22} non-zero mean {sum(nz)/len(nz):.4f}  range [{min(nz):.4f}, {max(nz):.4f}]"
                  f"  overall mean {c['success_mean']:.4f}")
    print()


def ceiling(env, arms):
    """A saturation claim rests on the MAXIMUM, not the mean."""
    print("=" * 92)
    print(f"{env} — flat budget ladder (a ceiling claim rests on the maximum)")
    print("=" * 92)
    for arm in arms:
        c = cell(env, arm)
        if not c:
            print(f"  {arm:<22} MISSING")
            continue
        steps = c["budgets"][0][1] if c.get("budgets") else None
        print(f"  {arm:<22} n={c['n']:<3} budget={steps:>12,}  mean {c['success_mean']:.4f}"
              f"  max {max(c['success']):.4f}  min {min(c['success']):.4f}")
    print()


def fisher_exact_greater(a, b, c, d):
    """One-sided Fisher exact: is the collapse rate a/(a+b) higher than c/(c+d)?

    Table is [[a, b], [c, d]] = [[flat collapses, flat ok], [hier collapses, hier ok]].
    """
    n = a + b + c + d
    r1, c1 = a + b, a + c
    p = 0.0
    lo = max(0, c1 - b)
    hi = min(r1, c1)
    for k in range(a, hi + 1):
        p += comb(r1, k) * comb(n - r1, c1 - k)
    return p / comb(n, c1)


def collapse(env, flat_arm, hier_arms, thresh=0.01):
    """Does the hierarchy eliminate the all-or-nothing collapse mode?

    A distinct claim from 'higher mean': the flat baseline has a bimodal failure where a
    fraction of seeds never learn at all. Reported as a rate with an exact Fisher test,
    because a mean over a bimodal distribution describes no episode that ever occurred.
    """
    f = cell(env, flat_arm)
    if not f:
        return
    fz = sum(1 for v in f["success"] if v < thresh)
    fo = f["n"] - fz
    print("=" * 92)
    print(f"{env} — COLLAPSE-MODE ELIMINATION (success < {thresh}), budget-matched arms only")
    print("=" * 92)
    fsteps = f["budgets"][0][1] if f.get("budgets") else None
    print(f"  {flat_arm:<18} collapses {fz}/{f['n']}  ({100*fz/f['n']:.1f}%)  budget {fsteps:,}")
    for arm in hier_arms:
        h = cell(env, arm)
        if not h:
            print(f"  {arm:<18} MISSING")
            continue
        hsteps = h["budgets"][0][1] if h.get("budgets") else None
        hz = sum(1 for v in h["success"] if v < thresh)
        ho = h["n"] - hz
        tag = "MATCHED" if hsteps == fsteps else f"*** MISMATCHED {hsteps:,} vs {fsteps:,} ***"
        p = fisher_exact_greater(fz, fo, hz, ho)
        print(f"  {arm:<18} collapses {hz}/{h['n']}  ({100*hz/h['n']:.1f}%)  budget {hsteps:,}  [{tag}]")
        print(f"  {'':18} Fisher exact one-sided p={p:.3g}"
              f"{'   [n<=3 — PROVISIONAL]' if h['n'] <= 3 else ''}")
    print()


def ladder(env, budgets, families, seeds=(0, 1, 2)):
    """Seed-matched budget ladders — 'whose plateau is it?' in one table.

    budgets:  [label, ...] in increasing order.
    families: {family: [arm_name_per_budget, ...]} aligned to `budgets`; None to skip a cell.

    The arm names differ per family at the same budget (hopper 1x nesy is `nesy·matched`, while
    `nesy·v2` is a 2x arm), so the mapping is explicit rather than templated — getting this wrong
    is exactly the budget asymmetry the campaign has already been bitten by twice.
    """
    print("=" * 92)
    print(f"{env} — seed-matched budget ladders (same seed indices across every budget)")
    print("=" * 92)
    hdr = "  " + f"{'seed/arm':<12}" + "".join(f"{lab:>13}" for lab in budgets)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in seeds:
        for family, arms in families.items():
            row = []
            for arm in arms:
                c = cell(env, arm) if arm else None
                if not c:
                    row.append("--")
                    continue
                d = dict(zip(c["seeds"], c["success"]))
                row.append(f"{d[s]:.4f}" if s in d else "--")
            print(f"  s{s} {family:<9}" + "".join(f"{v:>13}" for v in row))
        print()
    print("  arms: " + " | ".join(
        f"{f}=[{', '.join(a or '--' for a in arms)}]" for f, arms in families.items()))
    print()


def main():
    print("\n" + "#" * 92)
    print("# HOPPER — is the plateau the HIERARCHY's, or the neural meta-controller's?")
    print("#" * 92 + "\n")
    contrast("HopperHop", "neural·budget4x", "flat·budget4x", "[budget-matched 4x]")
    contrast("HopperHop", "nesy·budget4x", "flat·budget4x", "[budget-matched 4x — THE open question]")
    # The decisive within-hierarchy contrast: same budget, same exploration recipe, same seeds,
    # differing ONLY in meta-policy type. If the plateau were the hierarchy's, these would agree.
    contrast("HopperHop", "neural·budget4x", "nesy·budget4x",
             "[budget-matched 4x — WHOSE plateau is it? both arms hierarchical]")
    contrast("HopperHop", "neural·budget8x", "flat·budget8x", "[budget-matched 8x]")
    contrast("HopperHop", "nesy·budget8x", "flat·budget8x", "[budget-matched 8x]")
    contrast("HopperHop", "nesy·matched", "flat", "[budget-matched 1x — the honest comparison]")
    contrast("HopperHop", "neural·v2", "flat", "[budget-matched 1x]")
    ceiling("HopperHop", ["flat", "flat·budget2x", "flat·budget4x", "flat·budget8x", "flat·budget16x"])
    # `nesy·v2` trains at 52,428,800 = 2x, so it belongs in the 2x column, NOT the 1x one; the
    # 1x nesy arm is `nesy·matched`. Placing nesy·v2 at 1x is the exact asymmetry that once had
    # the gate reporting "nesy·v2 BEATS flat" at twice the baseline's steps.
    ladder("HopperHop",
           ["1x", "2x", "4x", "8x", "16x"],
           {
               "flat":   ["flat", "flat·budget2x", "flat·budget4x", "flat·budget8x", "flat·budget16x"],
               "neural": ["neural·v2", "neural·budget2x", "neural·budget4x", "neural·budget8x", None],
               "nesy":   ["nesy·matched", "nesy·v2", "nesy·budget4x", "nesy·budget8x", None],
           })
    zero_profile("HopperHop", ["flat", "nesy·matched", "neural·v2", "nesy·v2"])
    collapse("HopperHop", "flat", ["neural·v2", "nesy·matched"])

    print("\n" + "#" * 92)
    print("# GO1 ROUGH TERRAIN — the campaign's strongest result")
    print("#" * 92 + "\n")
    contrast("Go1JoystickRoughTerrain", "nesy·v2", "flat·v2", "[budget-matched]")
    contrast("Go1JoystickRoughTerrain", "neural·v2", "flat·v2", "[budget-matched]")
    collapse("Go1JoystickRoughTerrain", "flat·v2", ["nesy·v2", "neural·v2"])

    print("\n" + "#" * 92)
    print("# PANDA — the zero-seed reversal and the commitment mechanism")
    print("#" * 92 + "\n")
    contrast("PandaPickCube", "nesy·commit10", "nesy·v2", "[mechanism: intervention vs base arm]")
    zero_profile("PandaPickCube",
                 ["flat·v2", "flat·budget2x", "flat·budget4x", "flat·budget8x",
                  "nesy·v2", "nesy·commit10"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
