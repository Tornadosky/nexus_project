#!/usr/bin/env python3
"""Integrity audit of the generated dashboard.

Written because a review found several defects that the page did not visibly complain about:
env cards missing whole variants, the primary-success curve silently absent (wrong metric key),
duplicated seeds inflating a band, and videos rendered for only some arms. Every check below
corresponds to one of those, so the same class of defect fails loudly next time.

Checks
------
1.  media on disk   — every .mp4 decodes, has frames, has a companion skill strip + manifest
2.  media on page   — every <video> carries a non-empty data URI
3.  env cards       — each card has a video and curves for every variant present in the data
4.  curves          — return AND primary-success series are actually plottable per run
5.  duplicates      — no (env, variant, seed) appears twice across the run trees
6.  template        — no unrendered f-string placeholders left in the HTML
7.  size            — page fits the 16 MB Artifact limit
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("flat", "neural", "symbolic", "nesy")
RETURN_KEY = "rollout/episode_return"
SUCCESS_KEY = "policy_diag/primary_success_rate"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", default="runs/dashboard.html")
    ap.add_argument("--videos", default="runs/videos")
    ap.add_argument("--runs", nargs="+", default=["runs/verify", "runs/viper"])
    args = ap.parse_args()
    problems: list[str] = []

    # ---- 1. media on disk ------------------------------------------------
    print("== 1. videos on disk ==")
    import imageio.v2 as iio

    vids = sorted(Path(args.videos).glob("*.mp4"))
    for mp4 in vids:
        try:
            rd = iio.get_reader(str(mp4))
            n = sum(1 for _ in rd)
            rd.close()
        except Exception as e:  # noqa: BLE001
            problems.append(f"video unreadable: {mp4.name} ({e})")
            print(f"  BAD  {mp4.name}: {e}")
            continue
        strip = mp4.with_name(mp4.stem + "_skills.png")
        manifest = mp4.with_suffix(".json")
        miss = [p.name for p in (strip, manifest) if not p.exists()]
        if n == 0:
            problems.append(f"video has 0 frames: {mp4.name}")
        if miss:
            problems.append(f"{mp4.name} missing companion(s): {miss}")
        flag = "" if (n > 0 and not miss) else "   <-- PROBLEM"
        print(f"  {mp4.name:<34} {n:>4} frames  {mp4.stat().st_size/1e6:5.2f} MB{flag}")
    print(f"  {len(vids)} videos on disk")

    # ---- 2/6/7. page-level -----------------------------------------------
    print("\n== 2. media embedded on the page ==")
    h = Path(args.html).read_text(encoding="utf-8")
    srcs = re.findall(r'<video[^>]*src="([^"]{0,60})', h)
    empty = [s for s in srcs if not s.startswith("data:video")]
    print(f"  <video> tags: {len(srcs)}   non-data-URI: {len(empty)}")
    if empty:
        problems.append(f"{len(empty)} <video> tags without an inline data URI")
    imgs = h.count("data:image/png;base64,")
    print(f"  inline PNGs: {imgs}")
    ph = re.findall(r"\{[a-z_0-9]+\}", h)
    if ph:
        problems.append(f"unrendered template placeholders: {set(ph)}")
    mb = len(h.encode()) / 1e6
    print(f"  page size: {mb:.2f} MB")
    if mb > 16:
        problems.append(f"page {mb:.2f} MB exceeds the 16 MB artifact limit")

    # ---- 3/4/5. data ------------------------------------------------------
    print("\n== 3-5. runs, curves, duplicates ==")
    per_env: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for d in (Path(x) for x in args.runs):
        if not d.exists():
            continue
        for pkl in sorted(d.rglob("*.pkl")):
            if any(t in pkl.stem for t in ("clip", "quarter", "scaleclip", "hpqn", "diag", "commit")):
                continue
            try:
                with open(pkl, "rb") as fh:
                    ck = pickle.load(fh)
            except Exception:
                problems.append(f"unreadable checkpoint: {pkl}")
                continue
            cfg, m = ck.get("config") or {}, ck.get("metrics") or {}
            env = str(cfg.get("ENV_NAME")); var = str(cfg.get("META_POLICY_TYPE")).lower()
            seed = str(cfg.get("SEED"))
            per_env[env][var].append(pkl.name)
            seen[(env, var, seed)].append(pkl.name)
            for key, label in ((RETURN_KEY, "return"), (SUCCESS_KEY, "success")):
                a = np.asarray(m.get(key, [])).reshape(-1)
                if a.size < 2:
                    problems.append(f"{pkl.name}: {label} series missing/too short")

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    for env in sorted(per_env):
        card = re.search(rf"<h3>{re.escape(env)}</h3>(.*?)(?=<h3>|</section>)", h, re.S)
        body = card.group(1) if card else ""
        nvid = body.count("<video")
        counts = "  ".join(f"{v}:{len(f)}" for v, f in sorted(per_env[env].items()))
        missing = [v for v in per_env[env] if not re.search(rf"\b{v}\b", body)]
        flag = ""
        if not card:
            problems.append(f"{env}: no card on the page"); flag = "   <-- NO CARD"
        elif nvid < len(per_env[env]):
            problems.append(f"{env}: {nvid} videos for {len(per_env[env])} variants"); flag = "   <-- FEW VIDEOS"
        elif missing:
            problems.append(f"{env}: variants not named in card: {missing}"); flag = "   <-- MISSING VARIANT"
        print(f"  {env:<26} {counts:<44} videos={nvid}{flag}")

    # Two different things look alike here and only one is a defect:
    #   * the SAME filename in two trees  -> the same experiment counted twice (a defect);
    #   * DIFFERENT filenames, same seed  -> two different experiments that legitimately share a
    #     seed (cartpole's V1.2 shipped-noise `flat` vs V1.3 `_explore_` corrected-noise cells).
    #     The dashboard dedups these preferring the canonical tag and analyze_v2 keeps them as
    #     separate arms, so they are reported as informational rather than as problems.
    real = {k: v for k, v in dupes.items() if len(set(v)) == 1}
    distinct = {k: v for k, v in dupes.items() if len(set(v)) > 1}
    for k, v in real.items():
        problems.append(f"same experiment counted twice {k}: {v[0]}")
    if real:
        print(f"  duplicate FILES (defect): {len(real)}")
    if distinct:
        print(f"  same seed, different experiments (ok, deduped downstream): {len(distinct)}")
        for k, v in list(distinct.items())[:3]:
            print(f"    {k[0]}/{k[1]}/s{k[2]}: {sorted(set(v))}")
    if not dupes:
        print("  duplicates: none")

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
