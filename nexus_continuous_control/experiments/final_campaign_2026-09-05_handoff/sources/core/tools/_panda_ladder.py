#!/usr/bin/env python3
"""Panda flat, seed-matched across budgets, at the n=12 denominators."""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "runs/v2/v2_matrix.json").read_text(encoding="utf-8"))
P = M["matrix"]["PandaPickCube"]
arms = [("1x", "flat·v2"), ("2x", "flat·budget2x"), ("4x", "flat·budget4x")]
maps = {lab: dict(zip(P[a]["seeds"], P[a]["success"])) for lab, a in arms}
shared = sorted(set.intersection(*(set(m) for m in maps.values())))

print("seed " + "".join(f"{lab:>10}" for lab, _ in arms) + "   pattern")
print("-" * 52)
for s in shared:
    row = [maps[lab][s] for lab, _ in arms]
    pat = "".join("0" if v < 0.01 else "+" for v in row)
    print(f"s{s:<4}" + "".join(f"{v:>10.4f}" for v in row) + f"   {pat}")

print()
for lab, _ in arms:
    v = [maps[lab][s] for s in shared]
    z = sum(1 for x in v if x < 0.01)
    print(f"{lab}: mean {sum(v)/len(v):.4f}   zeros {z}/{len(shared)}")

# Seed-matched 4x vs 1x on the shared set.
a = [maps["4x"][s] for s in shared]
b = [maps["1x"][s] for s in shared]
w = sum(1 for x, y in zip(a, b) if x > y)
n = len(shared)
p = sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n
print(f"\n4x vs 1x seed-matched: {w}/{n} improve   exact sign p={p:.4g}")
a2 = [maps["2x"][s] for s in shared]
w2 = sum(1 for x, y in zip(a2, b) if x > y)
p2 = sum(comb(n, k) for k in range(w2, n + 1)) / 2 ** n
print(f"2x vs 1x seed-matched: {w2}/{n} improve   exact sign p={p2:.4g}")

# Which seeds are zero where.
print("\nzero-seed membership on the shared set:")
for lab, _ in arms:
    print(f"  {lab}: {[s for s in shared if maps[lab][s] < 0.01]}")
