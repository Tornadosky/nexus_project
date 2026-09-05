"""Audit the env-step budget of every cell on disk.

Budget is part of an experiment's identity: a cell trained at 2x the steps of the cell it is
compared against is a different experiment wearing the same name, and `tools/analyze_v2.py`
excludes explicit budget tags (`quarter`, `budget2x`, `budget4x`) from the V2 gate for exactly
that reason. What the tag system cannot catch is an UNTAGGED asymmetry — two arms of the same
env, both inside the gate, trained at different budgets — or a tag whose name no longer matches
its number.

This reports, it does not edit. Run it before quoting any cross-arm comparison.

    JAX_PLATFORMS=cpu python tools/audit_budgets.py
"""

from __future__ import annotations

import collections
import glob
import pickle

DIRS = ("runs/verify", "runs/viper")


def main() -> int:
    budgets: dict[tuple, set] = collections.defaultdict(set)
    counts: collections.Counter = collections.Counter()

    for d in DIRS:
        for path in sorted(glob.glob(d + "/*.pkl")):
            try:
                with open(path, "rb") as fh:
                    cfg = pickle.load(fh).get("config", {}) or {}
            except Exception:
                continue
            env = cfg.get("ENV_NAME")
            total = cfg.get("TOTAL_TIMESTEPS")
            variant = str(cfg.get("META_POLICY_TYPE") or "?")
            if not env or total is None:
                continue
            stem = path.split("/")[-1].rsplit("_s", 1)[0]
            key = (env, variant, stem)
            budgets[key].add(int(total))
            counts[key] += 1

    current_env = None
    for key in sorted(budgets):
        env, variant, stem = key
        if env != current_env:
            print("\n=== " + str(env))
            current_env = env
        values = sorted(budgets[key])
        shown = ", ".join(format(v, ",") for v in values)
        flag = "   <-- MIXED BUDGETS WITHIN ONE CELL" if len(values) > 1 else ""
        print("  {:9s} {:44s} n={:<3d} {}{}".format(variant, stem, counts[key], shown, flag))

    # The check that matters: arms of the same env that sit INSIDE the V2 gate (no budget tag in
    # the name) but do not share a budget. Those are compared directly by analyze_v2.
    print("\n" + "=" * 88)
    print("UNTAGGED BUDGET ASYMMETRY (same env, gate-eligible arms, different budgets)")
    print("=" * 88)
    tagged = ("quarter", "budget2x", "budget4x")
    per_env: dict[str, dict[str, set]] = collections.defaultdict(dict)
    for (env, variant, stem), values in budgets.items():
        if any(t in stem for t in tagged):
            continue
        per_env[env][stem] = values

    found = False
    for env in sorted(per_env):
        all_values = set()
        for values in per_env[env].values():
            all_values |= values
        if len(all_values) > 1:
            found = True
            print("\n  " + env)
            for stem in sorted(per_env[env]):
                vals = ", ".join(format(v, ",") for v in sorted(per_env[env][stem]))
                print("    {:46s} {}".format(stem, vals))
    if not found:
        print("\n  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
