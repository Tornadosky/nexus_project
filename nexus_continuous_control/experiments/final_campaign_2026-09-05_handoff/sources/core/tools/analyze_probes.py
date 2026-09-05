#!/usr/bin/env python3
"""Aggregate skill_probe_eval outputs into the dashboard's two probe tables.

* ``skill_probes.json`` — forced-skill probes, averaged over seeds. A skill "matches its name"
  when it ranks in the top-k of the four forced runs on its specialty metric (direction and k
  fixed below, chosen before any result was seen). The full metric values are carried alongside
  the verdict so a reader can re-judge.

* ``skill_ablation.json`` — per-skill eval-time ablation vs the intact eval, per seed, with the
  pre-registered prediction (see tools/run_skill_probes.sh) applied mechanically:
      "large drop"  holds iff ablated <  0.5 * intact
      "drop"        holds iff ablated <  0.8 * intact
      "minor"       holds iff ablated >= 0.8 * intact
      "no collapse" holds iff ablated >= 0.5 * intact
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# specialty metric per skill: (metrics key, direction, top-k among the forced runs)
PROBE_SPECS = {
    "Go1JoystickFlatTerrain": {
        "show": ["go1/no_fall_rate", "go1/velocity_tracking_error_mean",
                 "go1/yaw_tracking_error_mean"],
        "specialty": {
            "stand": ("go1/no_fall_rate", "max", 2),
            "track_velocity": ("go1/velocity_tracking_error_mean", "min", 1),
            "turn": ("go1/yaw_tracking_error_mean", "min", 1),
            "recover": ("go1/no_fall_rate", "max", 2),
        },
    },
    "HopperHop": {
        "show": ["hopper/upright_rate", "hopper/forward_velocity_mean", "hopper/joint_speed"],
        "specialty": {
            "stand_recover": ("hopper/upright_rate", "max", 2),
            "hop_forward": ("hopper/forward_velocity_mean", "max", 1),
            "stabilize_landing": ("hopper/joint_speed", "min", 2),
            "energy_efficient": ("hopper/joint_speed", "min", 2),
        },
    },
}

# (prediction text, criterion) per env/skill — transcribed from run_skill_probes.sh
ABLATION_PREDICTIONS = {
    ("Go1JoystickFlatTerrain", "recover"): ("large drop (<50% of intact)", "large drop"),
    ("Go1JoystickFlatTerrain", "turn"): ("drop (<80% of intact)", "drop"),
    ("Go1JoystickFlatTerrain", "track_velocity"): ("drop (<80% of intact)", "drop"),
    ("Go1JoystickFlatTerrain", "stand"): ("minor (>=80% of intact)", "minor"),
    ("HopperHop", "hop_forward"): ("large drop (<50% of intact)", "large drop"),
    ("HopperHop", "stand_recover"): ("drop (<80% of intact)", "drop"),
    ("HopperHop", "stabilize_landing"): ("minor (>=80% of intact)", "minor"),
    ("HopperHop", "energy_efficient"): ("no collapse (>=50% of intact)", "no collapse"),
}


def _criterion_holds(kind: str, ablated: float, intact: float) -> bool:
    if intact <= 0:
        return False
    r = ablated / intact
    return {
        "large drop": r < 0.5,
        "drop": r < 0.8,
        "minor": r >= 0.8,
        "no collapse": r >= 0.5,
    }[kind]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/probes")
    ap.add_argument("--out", default="runs/probes")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / "*.probe.json")))
    if not files:
        print("no .probe.json files found")
        return 1

    # ---- forced probes: average per (env, skill) over seeds ---------------- #
    forced_acc: dict = defaultdict(lambda: defaultdict(list))
    ablation_rows = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        env, name = d["env"], Path(f).stem.replace(".probe", "")
        for skill, s in d.get("forced", {}).items():
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    forced_acc[(env, skill)][k].append(float(v))
        intact = d.get("intact", {}).get("primary_success_rate")
        for skill, s in d.get("ablated", {}).items():
            pred_text, crit = ABLATION_PREDICTIONS.get((env, skill), ("", None))
            succ = float(s.get("primary_success_rate", float("nan")))
            ablation_rows.append({
                "env": env,
                "checkpoint": name,
                "removed_skill": skill,
                "intact_success": float(intact),
                "success": succ,
                "prediction": pred_text,
                "prediction_held": (_criterion_holds(crit, succ, float(intact))
                                    if crit else None),
            })

    probe_rows = []
    # rank each skill's specialty within its env's forced set
    for env in sorted({e for e, _ in forced_acc}):
        spec = PROBE_SPECS.get(env, {})
        skills = [s for (e, s) in forced_acc if e == env]
        means = {s: {k: float(np.mean(v)) for k, v in forced_acc[(env, s)].items()}
                 for s in skills}
        for s in skills:
            verdict = None
            key = spec.get("specialty", {}).get(s)
            if key:
                metric, direction, topk = key
                vals = {t: means[t].get(metric, float("nan")) for t in skills}
                order = sorted(vals, key=lambda t: vals[t], reverse=(direction == "max"))
                verdict = order.index(s) < topk
            probe_rows.append({
                "env": env,
                "skill": s,
                "episode_return_mean": means[s].get("episode_return_mean", float("nan")),
                "semantics": {k: means[s].get(k, float("nan"))
                              for k in spec.get("show", []) if k in means[s]},
                "specialty": key[0] if key else None,
                "matches_name": verdict,
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "skill_probes.json").write_text(json.dumps(probe_rows, indent=2), encoding="utf-8")
    (out / "skill_ablation.json").write_text(json.dumps(ablation_rows, indent=2), encoding="utf-8")

    print(f"{'env':<24}{'forced skill':<20}{'specialty':<34}verdict")
    for r in probe_rows:
        print(f"{r['env']:<24}{r['skill']:<20}{str(r['specialty']):<34}"
              f"{'PASS' if r['matches_name'] else ('—' if r['matches_name'] is None else 'FAIL')}")
    print()
    print(f"{'env':<24}{'removed':<20}{'intact':>8}{'ablated':>9}  prediction")
    for r in ablation_rows:
        held = ('HELD' if r['prediction_held']
                else ('—' if r['prediction_held'] is None else 'BROKEN'))
        print(f"{r['env']:<24}{r['removed_skill']:<20}{r['intact_success']:>8.3f}"
              f"{r['success']:>9.3f}  {r['prediction']} [{held}]")
    print(f"\nwrote {out / 'skill_probes.json'} and {out / 'skill_ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
