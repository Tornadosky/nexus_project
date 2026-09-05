#!/usr/bin/env python3
"""How the Go1Rough seed separation broke at n=30, quantified."""
import json
from math import comb
from pathlib import Path
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "runs/v2/v2_matrix.json").read_text(encoding="utf-8"))
M = M["matrix"]["Go1JoystickRoughTerrain"]
h, f = M["nesy·v2"], M["flat·v2"]
H, F = h["success"], f["success"]
fmax = max(F)

print(f"nesy n={len(H)} mean={sum(H)/len(H):.4f} min={min(H):.4f}")
print(f"flat n={len(F)} mean={sum(F)/len(F):.4f} max={fmax:.4f}")
print(f"gap  min(nesy) - max(flat) = {min(H) - fmax:+.4f}")
print()
below = [(s, round(v, 4)) for s, v in zip(h["seeds"], H) if v < fmax]
print(f"nesy seeds BELOW flat's max : {below}")
print(f"nesy seeds ABOVE flat's max : {sum(1 for v in H if v > fmax)}/{len(H)}")
print(f"flat's max is seed s{f['seeds'][F.index(fmax)]} = {fmax:.4f}")
print()
u, p = mannwhitneyu(H, F, alternative="greater")
print(f"Mann-Whitney U={u:.0f} of {len(H)*len(F)}   p={p:.4g}")
w = sum(1 for a, b in zip(H, F) if a > b)
sign_p = sum(comb(len(H), k) for k in range(w, len(H) + 1)) / 2 ** len(H)
print(f"seed-matched {w}/{len(H)}   exact sign p={sign_p:.4g}")
print()
print("n=18 -> n=30, what the twelve new seeds did:")
new = [(s, round(v, 4)) for s, v in zip(h["seeds"], H) if s >= 18]
print(f"  new seeds: {new}")
print(f"  new min {min(v for _, v in new):.4f}   old min (s0-s17) {min(H[:18]):.4f}")
