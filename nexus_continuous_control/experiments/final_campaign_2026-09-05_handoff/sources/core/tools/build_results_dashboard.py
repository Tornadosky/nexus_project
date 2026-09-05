#!/usr/bin/env python3
"""Build the per-environment results dashboard from what is on disk.

Companion to ``build_dashboard.py`` (which is the campaign *verification* board). This one is
organised the way the paper is: one section per environment, and inside it one column per
meta-policy variant -- flat / neural / nesy / symbolic -- carrying that arm's training curves,
its skill-level curves, its greedy-eval numbers and its rollout video.

Three rules it enforces, because each has already gone wrong in this repo:

* **Every asset is bound to a checkpoint, never to a filename.** ``runs/videos/*.json`` carries
  the checkpoint path it was rendered from; the env/variant/tag/seed shown next to a clip come
  from that checkpoint's own config, not from the clip's name. ``go1_flat_nesy_s0.mp4``,
  ``go1_nesy_s0.mp4`` and ``go1_nesy_v2_s0.mp4`` are three names for renders of ONE checkpoint,
  and the ``seed`` field inside those sidecars is the *render* seed, not the training seed.
* **The gate is imported, not reimplemented.** ``analyze_v2.collect`` / ``.gate`` are the single
  source of truth for arm keys, tail-means and the pass rule.
* **Training-time and greedy-eval numbers are never mixed in one column.** The training metric
  is logged with epsilon-greedy exploration still on; the deterministic eval is not, and on go1
  they disagree by 0.17 absolute.

Output: a single HTML file in ``runs/`` that references ``runs/videos``, ``runs/frames`` and
``runs/robustness`` by relative path, so it opens straight from the filesystem.

Usage
-----
    python tools/build_results_dashboard.py --out runs/results_dashboard.html
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["flat", "neural", "nesy", "symbolic"]
EXCLUDE_PREFIXES = ("diag",)

# Description of each environment and of what its primary success metric actually counts,
# read off the policy modules in nexus_continuous/policies/ rather than invented here.
ENV_META = {
    "CartpoleBalance": dict(
        success_short="upright fraction",
        short="Cartpole", family="dm_control suite",
        task="Balance an inverted pole on a cart.",
        success="fraction of steps with the pole upright"),
    "CheetahRun": dict(
        success_short="at-speed fraction",
        short="Cheetah", family="dm_control suite",
        task="Planar cheetah runs forward as fast as it can.",
        success="fraction of steps at or above the target forward speed"),
    "WalkerWalk": dict(
        success_short="at-speed fraction",
        short="Walker", family="dm_control suite",
        task="Planar biped walks forward at a target speed.",
        success="fraction of steps upright and at target speed"),
    "HopperHop": dict(
        success_short="hopping fraction",
        short="Hopper", family="dm_control suite",
        task="One-legged hopper hops forward without falling.",
        success="fraction of steps hopping forward at target speed"),
    "PandaPickCube": dict(
        success_short="lift success",
        short="Panda", family="MuJoCo Playground manipulation",
        task="Franka Panda reaches, grasps and lifts a cube.",
        success="lift success -- cube raised above the table by the target height"),
    "Go1JoystickFlatTerrain": dict(
        success_short="tracking success",
        short="Go1 flat", family="MuJoCo Playground locomotion",
        task="Unitree Go1 tracks a commanded linear and yaw velocity on flat ground.",
        success="velocity-tracking success rate"),
    "Go1JoystickRoughTerrain": dict(
        success_short="tracking success",
        short="Go1 rough", family="MuJoCo Playground locomotion",
        task="Unitree Go1 tracks a commanded velocity over a heightfield.",
        success="velocity-tracking success rate"),
}
ENV_ORDER = ["CartpoleBalance", "CheetahRun", "WalkerWalk", "HopperHop",
             "PandaPickCube", "Go1JoystickFlatTerrain", "Go1JoystickRoughTerrain"]

# runs/robustness/<name>.csv -> checkpoint, lifted verbatim from the sweep scripts that produced
# them (tools/run_robustness_sweep.sh, run_robustness_sweep2.sh). The CSV's own env/meta columns
# are checked against the checkpoint's config below; a disagreement is reported, not smoothed.
ROBUSTNESS_MAP_TEMPLATES = {
    "go1_flat_s{s}": "runs/verify/go1_joystick_flat_v2_s{s}.pkl",
    "go1_neural_s{s}": "runs/verify/go1_joystick_neural_v2_s{s}.pkl",
    "go1_nesy_s{s}": "runs/verify/go1_joystick_nesy_v2_s{s}.pkl",
    "go1_symbolic_s{s}": "runs/verify/go1_joystick_symbolic_v2_s{s}.pkl",
    "hopper_flat_s{s}": "runs/viper/hopper_hop_flat_s{s}.pkl",
    "hopper_neural_s{s}": "runs/verify/hopper_hop_neural_v2_s{s}.pkl",
    "hopper_nesy_s{s}": "runs/verify/hopper_hop_nesy_v2_s{s}.pkl",
    "hopper_symbolic_s{s}": "runs/verify/hopper_hop_symbolic_v2_s{s}.pkl",
    "cheetah_flat_s{s}": "runs/viper/cheetah_run_flat_s{s}.pkl",
    "cheetah_neural_s{s}": "runs/verify/cheetah_run_neural_v2_s{s}.pkl",
    "cheetah_nesy_s{s}": "runs/verify/cheetah_run_nesy_v2_s{s}.pkl",
    "cheetah_symbolic_s{s}": "runs/verify/cheetah_run_symbolic_v2_s{s}.pkl",
    "cartpole_flat_s{s}": "runs/verify/cartpole_balance_flat_explore_s{s}.pkl",
    "cartpole_neural_s{s}": "runs/verify/cartpole_balance_neural_explore_s{s}.pkl",
    "cartpole_nesy_s{s}": "runs/verify/cartpole_balance_nesy_explore_s{s}.pkl",
    "cartpole_symbolic_s{s}": "runs/verify/cartpole_balance_symbolic_explore_s{s}.pkl",
    "walker_flat_s{s}": "runs/viper/walker_flat_dm_s{s}.pkl",
    "walker_neural_s{s}": "runs/verify/walker_walk_neural_dm_s{s}.pkl",
    "walker_nesy_s{s}": "runs/verify/walker_walk_nesy_dm_s{s}.pkl",
    "walker_symbolic_s{s}": "runs/verify/walker_walk_symbolic_v2_s{s}.pkl",
    "panda_flat_s{s}": "runs/verify/panda_pick_cube_flat_v2_s{s}.pkl",
    "panda_neural_s{s}": "runs/verify/panda_pick_cube_neural_v2_s{s}.pkl",
    "panda_nesy_s{s}": "runs/verify/panda_pick_cube_nesy_v2_s{s}.pkl",
    "panda_symbolic_s{s}": "runs/verify/panda_pick_cube_symbolic_v2_s{s}.pkl",
    "panda_nesycommit10_s{s}": "runs/verify/panda_pick_cube_nesy_commit10_s{s}.pkl",
}


def robustness_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for tmpl, ck in ROBUSTNESS_MAP_TEMPLATES.items():
        for s in range(6):
            out[tmpl.format(s=s)] = ck.format(s=s)
    return out


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #

# Stored as full curves because the page plots them. Everything else is kept as a tail-mean
# scalar only: at 621 runs a stored-but-unplotted series costs ~0.5 MB of page for nothing.
CURVE_KEYS = [
    "rollout/episode_return",
    "policy_diag/primary_success_rate",
    "rollout/episode_length",
    "mask/violation_rate",
]
FINAL_KEYS = CURVE_KEYS + [
    "policy_diag/primary_goal_metric",
    "returns/env_reward_mean",
    "returns/skill_reward_mean",
]
NPTS = 96


def _ds(a) -> list:
    """Downsample by block-mean, so a 400-point curve stays honest at 96 points."""
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return []
    if a.size <= NPTS:
        vals = a
    else:
        idx = np.linspace(0, a.size, NPTS + 1).astype(int)
        vals = np.array([a[idx[i]:max(idx[i] + 1, idx[i + 1])].mean() for i in range(NPTS)])
    return [None if not np.isfinite(v) else round(float(v), 5) for v in vals]


def _ds_x(a) -> list:
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return []
    if a.size <= NPTS:
        return [float(v) for v in a]
    idx = np.linspace(0, a.size, NPTS + 1).astype(int)
    return [float(a[max(0, idx[i + 1] - 1)]) for i in range(NPTS)]


def _tail(a, frac: int = 10):
    """Mean of the last 10% of a curve. Same definition analyze_v2 scores the gate on."""
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return None
    v = float(a[-max(1, a.size // frac):].mean())
    return round(v, 6) if np.isfinite(v) else None


def load_runs(dirs: list[Path]) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for d in dirs:
        if not d.exists():
            continue
        for pkl in sorted(d.rglob("*.pkl")):
            try:
                with open(pkl, "rb") as fh:
                    ck = pickle.load(fh)
            except Exception as e:  # noqa: BLE001
                print(f"  UNREADABLE {pkl}: {e}")
                continue
            cfg = ck.get("config", {}) or {}
            m = ck.get("metrics", {}) or {}
            variant = str(cfg.get("META_POLICY_TYPE") or "").lower()
            env = str(cfg.get("ENV_NAME") or "?")
            if variant not in VARIANTS or env == "?":
                continue
            stem = pkl.stem
            # Same exclusion analyze_v2 applies. `diag_walker_nesy_e512` is a 512-env ROCm probe
            # at 1/32 of the walker budget; without this it lands in WalkerWalk `nesy` as an
            # n=1 arm and reports its diagnostic number as a result.
            if stem.startswith(EXCLUDE_PREFIXES):
                continue
            tag = analyze_v2._tag_from(stem, variant)
            arm = f"{variant}·{tag}" if tag else variant
            xs = _ds_x(m.get("env_step", np.arange(len(m.get("rollout/episode_return", [])))))
            rec: dict[str, Any] = {
                "key": str(pkl.relative_to(ROOT)).replace("\\", "/"),
                "stem": stem,
                "env": env,
                "variant": variant,
                "tag": tag,
                "arm": arm,
                "seed": int(cfg.get("SEED", -1)),
                "num_envs": int(cfg.get("NUM_ENVS", 0)),
                "total_timesteps": int(cfg.get("TOTAL_TIMESTEPS", 0)),
                "policy": cfg.get("POLICY"),
                "commit": (ck.get("commit_hash") or "")[:10],
                "experimental": analyze_v2._is_experimental(arm),
                "x": xs,
                "curves": {},
                "final": {},
            }
            for k in CURVE_KEYS:
                if k in m:
                    rec["curves"][k] = _ds(m[k])
            for k in FINAL_KEYS:
                if k in m:
                    rec["final"][k] = _tail(m[k])
            skills = [k.split("/", 1)[1] for k in m if k.startswith("skill_return/")]
            skills.sort(key=lambda s: int(s.split("_", 1)[0]) if s.split("_", 1)[0].isdigit() else 0)
            rec["skills"] = skills
            rec["skill_return"] = {s: _ds(m[f"skill_return/{s}"]) for s in skills
                                   if f"skill_return/{s}" in m}
            rec["skill_usage"] = {s: _ds(m[f"skill_usage/{s}"]) for s in skills
                                  if f"skill_usage/{s}" in m}
            rec["skill_return_final"] = {s: _tail(m[f"skill_return/{s}"]) for s in skills
                                         if f"skill_return/{s}" in m}
            rec["skill_usage_final"] = {s: _tail(m[f"skill_usage/{s}"]) for s in skills
                                        if f"skill_usage/{s}" in m}
            rec["mask_available_final"] = {k.split("/", 1)[1]: _tail(m[k]) for k in m
                                           if k.startswith("mask_available/")}
            runs[rec["key"]] = rec
    return runs


# --------------------------------------------------------------------------- #
# videos
# --------------------------------------------------------------------------- #

def load_videos(vdir: Path, fdir: Path, runs: dict[str, dict]) -> list[dict]:
    """Bind every clip to the checkpoint it was rendered from, and audit the binding.

    Everything shown next to a clip -- env, variant, tag, TRAINING seed -- is read from that
    checkpoint's config. The sidecar's own ``env``/``variant``/``seed`` fields are compared
    against it and any disagreement is recorded as a note rather than silently preferred.
    """
    out: list[dict] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    for js in sorted(vdir.glob("*.json")):
        if js.name.startswith("_"):
            continue
        try:
            meta = json.loads(js.read_text())
        except Exception:  # noqa: BLE001
            continue
        mp4 = vdir / (meta.get("video") or f"{js.stem}.mp4")
        if not mp4.exists():
            continue
        digest = hashlib.md5(mp4.read_bytes()).hexdigest()
        # tools/shrink_videos.py keeps every Nth frame so the clip fits an Artifact page. That
        # is honest -- the whole episode is still shown -- but it plays at Nx, and a card that
        # says "1000 frames" beside a 200-frame clip invites the reader to time the behaviour
        # off the video. Measure what is actually in the file and say the speed-up out loud.
        played, fps = _ffprobe_frames(mp4)
        by_hash[digest].append(mp4.name)
        ckpath = (meta.get("checkpoint") or "").replace("\\", "/")
        run = runs.get(ckpath)
        strip = vdir / (meta.get("strip") or f"{js.stem}_skills.png")
        frame = fdir / f"{js.stem}.png"
        notes: list[str] = []
        if run is None:
            notes.append(f"checkpoint {ckpath or '(none)'} not found among loaded runs")
        else:
            if meta.get("env") and meta["env"] != run["env"]:
                notes.append(f"sidecar env {meta['env']} != checkpoint env {run['env']}")
            if meta.get("variant") and meta["variant"] != run["variant"]:
                notes.append(
                    f"sidecar variant {meta['variant']} != checkpoint variant {run['variant']}")
        out.append({
            "name": mp4.stem,
            "video": f"videos/{mp4.name}",
            "strip": f"videos/{strip.name}" if strip.exists() else None,
            "frame": f"frames/{frame.name}" if frame.exists() else None,
            "bytes": mp4.stat().st_size,
            "md5": digest,
            "checkpoint": ckpath,
            "env": run["env"] if run else meta.get("env"),
            "variant": run["variant"] if run else meta.get("variant"),
            "tag": run["tag"] if run else "",
            "arm": run["arm"] if run else "",
            "train_seed": run["seed"] if run else None,
            "render_seed": meta.get("seed"),
            "frames": meta.get("frames"),
            "played_frames": played,
            "fps": fps,
            "speedup": (round(meta["frames"] / played) if played and meta.get("frames")
                        and played > 0 and meta["frames"] / played >= 1.5 else None),
            "episode_return": meta.get("return"),
            "skill_names": meta.get("skill_names") or [],
            "skill_usage": meta.get("skill_usage") or [],
            "notes": notes,
            "resolved": run is not None,
        })
    dupes = {h: names for h, names in by_hash.items() if len(names) > 1}
    for v in out:
        if v["md5"] in dupes:
            others = [n for n in dupes[v["md5"]] if n != Path(v["video"]).name]
            v["notes"].append("byte-identical to " + ", ".join(others))
            v["duplicate_of"] = sorted(dupes[v["md5"]])[0]
    return out


# --------------------------------------------------------------------------- #
# deterministic / robustness evaluation
# --------------------------------------------------------------------------- #

def _read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def load_det(det_root: Path, runs: dict[str, dict]) -> dict[str, dict]:
    """Greedy eval at zero perturbation, keyed by checkpoint path.

    ``run_det_eval.sh`` names each CSV after the checkpoint stem, so the binding is exact; the
    env recorded in the CSV is still checked against the checkpoint's config.
    """
    out: dict[str, dict] = {}
    if not det_root.exists():
        return out
    by_stem = {r["stem"]: k for k, r in runs.items()}
    for csvp in sorted(det_root.rglob("*.csv")):
        rows = _read_csv(csvp)
        if not rows:
            continue
        row = min(rows, key=lambda r: float(r.get("perturbation", 0) or 0))
        key = by_stem.get(csvp.stem)
        if key is None:
            continue
        if row.get("env") and row["env"] != runs[key]["env"]:
            print(f"  DET MISMATCH {csvp.name}: {row['env']} != {runs[key]['env']}")
            continue
        out[key] = {
            "success": float(row["primary_success_rate"]),
            "return": float(row["episode_return_mean"]),
            "return_std": float(row.get("episode_return_std") or 0.0),
            "episode_length": float(row.get("episode_length_mean") or 0.0),
            "goal": float(row.get("primary_goal_metric") or 0.0),
            "source": str(csvp.relative_to(ROOT)).replace("\\", "/"),
        }
    return out


def cross_check_det(det: dict[str, dict], rob: dict[str, dict]) -> list[str]:
    """Where a checkpoint has BOTH a runs/det eval and a sweep, the two must agree.

    Both come from `robustness_eval.py` at 64 episodes and zero perturbation, so any difference
    would mean the evaluation is not deterministic -- and the greedy column, which the page
    presents as the honest number, would be a draw rather than a measurement.
    """
    bad = []
    for key, r in rob.items():
        if key not in det or not r["levels"] or r["levels"][0] != 0.0:
            continue
        d = det[key]["success"]
        if abs(d - r["success"][0]) > 1e-6:
            bad.append(f"{key}: runs/det says {d:.4f}, the sweep's zero-noise point says "
                       f"{r['success'][0]:.4f}")
    return bad


def merge_robustness_into_det(det: dict[str, dict], rob: dict[str, dict]) -> int:
    """A robustness sweep's zero-noise point IS the greedy eval, produced by the same tool.

    `run_det_eval.sh` only ever covered go1 and hopper, so without this the greedy column is
    empty on five of seven environments while the number sits on disk in runs/robustness.
    Existing runs/det entries win; nothing is overwritten.
    """
    added = 0
    for key, r in rob.items():
        if key in det or not r["levels"] or r["levels"][0] != 0.0:
            continue
        det[key] = {"success": r["success"][0], "return": r["return"][0], "return_std": 0.0,
                    "episode_length": 0.0, "goal": 0.0, "source": r["source"] + " (level 0.0)"}
        added += 1
    return added


def load_robustness(rdir: Path, runs: dict[str, dict]) -> dict[str, dict]:
    """Action-noise sweeps, bound to checkpoints through the sweep scripts' own mapping."""
    out: dict[str, dict] = {}
    if not rdir.exists():
        return out
    rmap = robustness_map()
    for csvp in sorted(rdir.glob("*.csv")):
        ck = rmap.get(csvp.stem)
        if ck is None or ck not in runs:
            continue
        rows = sorted(_read_csv(csvp), key=lambda r: float(r["perturbation"]))
        run = runs[ck]
        if rows and rows[0].get("env") and rows[0]["env"] != run["env"]:
            print(f"  ROBUSTNESS MISMATCH {csvp.name}: {rows[0]['env']} != {run['env']}")
            continue
        if rows and rows[0].get("meta") and rows[0]["meta"] != run["variant"]:
            print(f"  ROBUSTNESS MISMATCH {csvp.name}: meta {rows[0]['meta']} != {run['variant']}")
            continue
        out[ck] = {
            "levels": [float(r["perturbation"]) for r in rows],
            "success": [float(r["primary_success_rate"]) for r in rows],
            "return": [float(r["episode_return_mean"]) for r in rows],
            "source": str(csvp.relative_to(ROOT)).replace("\\", "/"),
        }
    return out


# runs/oos/<prefix>_<cond>_<arm>_s<seed>.csv, where the prefix selects which checkpoint family
# the arm name is resolved against. Read off the queue scripts that produced the files:
#   go1_*  tools/_queue_oos_all.sh / _queue_oos_cmd.sh  -> runs/verify/go1_joystick_<arm>_s<n>.pkl
#   rt_*   tools/_queue_oos_roughtrained.sh             -> runs/verify/go1_rough_<arm>_s<n>.pkl
#   r2_*   tools/_queue_rules2_eval.sh                  -> runs/verify/go1_joystick_<arm>_s<n>.pkl
OOS_PREFIX_BASE = {
    "go1": "runs/verify/go1_joystick_{arm}_s{s}.pkl",
    "rt": "runs/verify/go1_rough_{arm}_s{s}.pkl",
    "r2": "runs/verify/go1_joystick_{arm}_s{s}.pkl",
}
OOS_RE = re.compile(r"^(go1|rt|r2)_([a-z0-9]+)_(.+)_s(\d+)$")


def load_oos(odir: Path, runs: dict[str, dict]) -> tuple[dict, list[str]]:
    """Held-out command / terrain evaluations, resolved back to their checkpoints.

    Returns ``{prefix: {condition: [row, ...]}}`` plus any rows that could not be bound to a
    checkpoint, which are reported rather than plotted.
    """
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unresolved: list[str] = []
    if not odir.exists():
        return {}, unresolved
    for csvp in sorted(odir.glob("*.csv")):
        m = OOS_RE.match(csvp.stem)
        if not m:
            unresolved.append(f"{csvp.name}: filename does not parse")
            continue
        prefix, cond, arm, seed = m.groups()
        ck = OOS_PREFIX_BASE[prefix].format(arm=arm, s=seed)
        run = runs.get(ck)
        if run is None:
            unresolved.append(f"{csvp.name}: checkpoint {ck} not loaded")
            continue
        rows = _read_csv(csvp)
        if not rows:
            unresolved.append(f"{csvp.name}: empty")
            continue
        row = rows[0]
        if row.get("meta") and row["meta"] != run["variant"]:
            unresolved.append(
                f"{csvp.name}: meta {row['meta']} != checkpoint variant {run['variant']}")
            continue
        out[prefix][cond].append({
            "variant": run["variant"],
            "arm": run["arm"],
            "seed": run["seed"],
            "eval_env": row.get("env"),
            "train_env": row.get("train_env") or run["env"],
            "success": float(row.get("primary_success_rate") or 0.0),
            "return": float(row.get("episode_return_mean") or 0.0),
            "checkpoint": ck,
            "source": str(csvp.relative_to(ROOT)).replace("\\", "/"),
        })
    return {p: dict(c) for p, c in out.items()}, unresolved


def load_ppo(pdir: Path) -> dict[str, list[dict]]:
    """The Brax PPO baseline, which is the paper's own `PPO (baseline)` arm.

    These CSVs come out of `tools/train_ppo_baseline.py` through the same `robustness_eval`
    harness as every other greedy number on this page, so they are directly comparable to the
    deterministic column -- and to nothing else. They carry no training curve in our metric
    schema, so PPO appears only where greedy evaluations are being compared.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    if not pdir.exists():
        return {}
    for csvp in sorted(pdir.glob("*.csv")):
        rows = _read_csv(csvp)
        if not rows:
            continue
        r = rows[0]
        env = r.get("env")
        if not env:
            continue
        arm = "ppo·shipped" if "shipped" in csvp.stem else "ppo"
        out[env].append({
            "arm": arm,
            "seed": int(r.get("seed") or 0),
            "success": float(r["primary_success_rate"]),
            "return": float(r["episode_return_mean"]),
            "steps": int(float(r.get("num_timesteps") or 0)),
            "source": str(csvp.relative_to(ROOT)).replace("\\", "/"),
        })
    return dict(out)


def ppo_verdicts(ppo: dict[str, list[dict]], cells: dict[str, dict],
                 shipped: dict[str, dict]) -> dict[str, str]:
    """Say plainly, per environment, what the external baseline does to the conclusion.

    A hierarchy that beats our own flat arm has not beaten "flat control" if a standard PPO
    baseline at the same budget beats them both. That is the case on Go1, and it has to be said
    on the page rather than left for the reader to derive from two separate charts.
    """
    out: dict[str, str] = {}
    for env, rows in ppo.items():
        base = [r for r in rows if r["arm"] == "ppo"]
        if not base:
            continue
        pmean = float(np.mean([r["success"] for r in base]))
        psteps = base[0]["steps"]
        ours = []
        for v, arm in (shipped.get(env) or {}).items():
            c = cells.get(f"{env}|{arm}")
            if c and c["det_n"]:
                ours.append((v, arm, c["det_success_mean"], c["steps"], c["det_n"]))
        if not ours:
            continue
        best = max(ours, key=lambda t: t[2])
        matched = "at the same budget" if best[3] == psteps else (
            f"at {best[3]:,} steps against PPO's {psteps:,}")
        if pmean > best[2]:
            out[env] = (
                f"A standard PPO baseline reaches {pmean:.3f} greedy success here (n={len(base)}, "
                f"{psteps:,} env steps), above the best arm on this page &mdash; {best[1]} at "
                f"{best[2]:.3f} (n={best[4]}), {matched}. Whatever the hierarchy does against our "
                "own flat PQN arm, it does not beat flat control in general on this environment. "
                "The comparison the gate scores is internal to the PQN family.")
        else:
            out[env] = (
                f"A standard PPO baseline reaches {pmean:.3f} greedy success here (n={len(base)}, "
                f"{psteps:,} env steps), below the best arm on this page &mdash; {best[1]} at "
                f"{best[2]:.3f} (n={best[4]}), {matched}. The internal comparison is not being "
                "flattered by a weak baseline family on this environment.")
    return out


# --------------------------------------------------------------------------- #
# shipped arms, rule listings, decision panels
# --------------------------------------------------------------------------- #

def pick_shipped(cells: dict[str, dict]) -> dict[str, dict[str, str]]:
    """The one arm per (env, variant) the headline charts use.

    Experimental arms are never eligible. The flat baseline is chosen first, on seed count; every
    hierarchical variant then prefers an arm **at the baseline's own budget**, falling back to the
    largest-n arm only when no matched arm exists. Choosing on seed count alone picks HopperHop
    ``nesy·v2`` (n=15, 52.4M steps) over ``nesy·matched`` (n=12, 26.2M) and quietly puts a 2x arm
    in the headline against a 1x baseline. Every other arm stays visible in the ladder and table.
    """
    by_env: dict[str, list[dict]] = defaultdict(list)
    for c in cells.values():
        if not c["experimental"] and c["success_mean"] is not None:
            by_env[c["env"]].append(c)
    out: dict[str, dict[str, str]] = {}
    for env, arms in by_env.items():
        chosen: dict[str, str] = {}
        flats = [c for c in arms if c["variant"] == "flat"]
        base_steps = None
        if flats:
            best = max(flats, key=lambda c: (c["n"], c["success_mean"]))
            chosen["flat"] = best["arm"]
            base_steps = best["steps"]
        for v in ("neural", "nesy", "symbolic"):
            pool = [c for c in arms if c["variant"] == v]
            if not pool:
                continue
            matched = [c for c in pool if base_steps is not None and c["steps"] == base_steps]
            chosen[v] = max(matched or pool, key=lambda c: (c["n"], c["success_mean"]))["arm"]
        out[env] = chosen
    return out


RULE_FUNCS = [
    ("symbolic_meta_policy", "symbolic meta-policy — the rule program (paper Fig. 5, left)"),
    ("skill_mask", "NeSy admissibility mask — the filtering rules (paper Fig. 5, right)"),
]


def _extract_func(src: str, name: str) -> str | None:
    lines = src.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(f"def {name}("):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j] and not lines[j][0].isspace() and not lines[j].startswith(")"):
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()


def load_rules(runs: dict[str, dict]) -> list[dict]:
    """The executed rule programs, read out of the policy modules the runs name."""
    pdir = ROOT / "nexus_continuous" / "policies"
    seen: dict[tuple[str, str], str] = {}
    for r in runs.values():
        if r["variant"] in ("symbolic", "nesy") and r["policy"]:
            seen[(r["env"], str(r["policy"]))] = r["arm"]
    out: list[dict] = []
    for (env, policy) in sorted(seen):
        f = pdir / f"{policy}.py"
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8")
        for fn, title in RULE_FUNCS:
            code = _extract_func(src, fn)
            if code:
                out.append({"env": env, "policy": policy, "title": title,
                            "source": f"nexus_continuous/policies/{policy}.py::{fn}",
                            "code": code})
    return out


def load_fig6(fdir: Path, runs: dict[str, dict]) -> list[dict]:
    """Decision panels, bound to the checkpoint their filename encodes."""
    out: list[dict] = []
    if not fdir.exists():
        return out
    by_stem = {r["stem"]: r for r in runs.values()}
    for png in sorted(fdir.glob("*.png")):
        stem, _, tpart = png.stem.rpartition("_t")
        run = by_stem.get(stem)
        if run is None:
            print(f"  FIG6 UNRESOLVED {png.name}")
            continue
        out.append({
            "img": f"fig6/{png.name}",
            "env": run["env"], "arm": run["arm"], "seed": run["seed"],
            "step": int(tpart) if tpart.isdigit() else tpart,
            "checkpoint": run["key"],
        })
    return out


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def build_cells(runs: dict[str, dict], det: dict[str, dict]) -> dict[str, dict]:
    """(env, arm) -> per-seed lists. Mirrors analyze_v2.collect, plus curves and eval."""
    cells: dict[tuple[str, str], dict] = {}
    for key, r in runs.items():
        ck = cells.setdefault((r["env"], r["arm"]), {
            "env": r["env"], "arm": r["arm"], "variant": r["variant"], "tag": r["tag"],
            "experimental": r["experimental"], "runs": [], "budgets": set(),
        })
        ck["runs"].append(key)
        ck["budgets"].add((r["num_envs"], r["total_timesteps"]))
    out: dict[str, dict] = {}
    for (env, arm), c in cells.items():
        rs = sorted((runs[k] for k in c["runs"]), key=lambda r: r["seed"])
        succ = [r["final"].get("policy_diag/primary_success_rate") for r in rs]
        rets = [r["final"].get("rollout/episode_return") for r in rs]
        succ_ok = [v for v in succ if v is not None]
        rets_ok = [v for v in rets if v is not None]
        det_s = [det[r["key"]]["success"] for r in rs if r["key"] in det]
        det_r = [det[r["key"]]["return"] for r in rs if r["key"] in det]
        # Compare greedy against training on the SAME seeds. An arm can have 12 trained seeds and
        # 3 evaluated ones; differencing the two means measures the seed sample, not the noise.
        tr_matched = [r["final"].get("policy_diag/primary_success_rate")
                      for r in rs if r["key"] in det]
        tr_matched = [v for v in tr_matched if v is not None]
        budgets = sorted(c["budgets"])
        out[f"{env}|{arm}"] = {
            "env": env, "arm": arm, "variant": c["variant"], "tag": c["tag"],
            "experimental": c["experimental"],
            "runs": [r["key"] for r in rs],
            "seeds": [r["seed"] for r in rs],
            "n": len(rs),
            "success": succ, "return": rets,
            "success_mean": float(np.mean(succ_ok)) if succ_ok else None,
            "success_min": float(np.min(succ_ok)) if succ_ok else None,
            "success_max": float(np.max(succ_ok)) if succ_ok else None,
            "success_std": float(np.std(succ_ok, ddof=1)) if len(succ_ok) > 1 else None,
            "return_mean": float(np.mean(rets_ok)) if rets_ok else None,
            "det_success_mean": float(np.mean(det_s)) if det_s else None,
            "det_return_mean": float(np.mean(det_r)) if det_r else None,
            "det_n": len(det_s),
            "train_success_on_det_seeds": float(np.mean(tr_matched)) if tr_matched else None,
            "budgets": [list(b) for b in budgets],
            "steps": budgets[0][1] if len(budgets) == 1 else None,
            "mixed_budget": len(budgets) > 1,
            "skills": rs[0]["skills"] if rs else [],
        }
    return out


# --------------------------------------------------------------------------- #
# audit -- the checks that make the page trustworthy
# --------------------------------------------------------------------------- #

def _c(s: str) -> str:
    return f"<code>{s}</code>"


def build_audit(runs, cells, videos, det, rob, gate, shipped, oos_unresolved) -> list[dict]:
    """Run every consistency check and return them as renderable blocks.

    A block with an empty ``items`` list is a check that ran and found nothing. That is reported
    as such rather than omitted -- a check that silently disappears when it passes is
    indistinguishable from one that was never written.
    """
    blocks: list[dict] = []

    # 1. asset binding
    items = [f"{_c(v['name'])}: " + "; ".join(v["notes"]) for v in videos if v["notes"]]
    blocks.append({
        "title": "Clip-to-checkpoint binding",
        "severity": "stop" if any(not v["resolved"] for v in videos) else "wait",
        "why": "Every clip is resolved through the checkpoint path in its sidecar, and the "
               "env/variant/seed shown beside it come from that checkpoint's config. Listed here "
               "is every case where the sidecar and the checkpoint disagree, where two clips are "
               "byte-identical, or where the checkpoint could not be found.",
        "items": items})

    # 2. render seed vs training seed
    conf = [v for v in videos if v["train_seed"] is not None
            and v["render_seed"] is not None and v["train_seed"] != v["render_seed"]]
    blocks.append({
        "title": "Render seed is not the training seed",
        "severity": "wait",
        "why": "tools/render_rollout.py writes its own <code>--seed</code> into the sidecar. It "
               "seeds the evaluation episode, not the training run. These clips are the ones "
               "where the two differ, and reading the sidecar's field as a training seed would "
               "mislabel them.",
        "items": [f"{_c(v['name'])}: trained at seed s{v['train_seed']}, rendered with render "
                  f"seed r{v['render_seed']} &mdash; labelled s{v['train_seed']} on this page"
                  for v in conf]})

    # 3. unrepresentative single episodes
    weird = []
    for v in videos:
        if v["episode_return"] is not None and abs(v["episode_return"]) < 1e-9:
            weird.append(f"{_c(v['name'])}: the rendered episode returned exactly 0.0. On a "
                         "bimodal arm that is one draw, not the arm; where several renders of "
                         "the same checkpoint are shown, compare them before reading any one as "
                         "typical")
        elif v["frames"] is not None and v["frames"] < 100:
            weird.append(f"{_c(v['name'])}: episode ended after {v['frames']} frames &mdash; the "
                         "agent terminated early; the clip shows a failure")
    blocks.append({
        "title": "Clips that are not representative of their arm",
        "severity": "wait",
        "why": "A single greedy episode from a bimodal arm can show the failure mode or the "
               "success mode. These clips are still shown &mdash; they are real behaviour &mdash; "
               "but they are marked on their own card so nobody quotes them as the arm's result.",
        "items": weird})

    # 3b. clips that do not play in real time
    fast = [f"{_c(v['name'])}: {v['frames']} env steps shown as {v['played_frames']} frames "
            f"&mdash; plays at about {v['speedup']}x"
            for v in videos if v.get("speedup")]
    blocks.append({
        "title": "Clips that play faster than the episode ran",
        "severity": "wait",
        "why": "<code>tools/shrink_videos.py</code> keeps every Nth frame so the clips fit a "
               "published page. The whole episode is still shown &mdash; nothing is truncated "
               "&mdash; but it runs at Nx, so nothing should be timed off the video. The skill "
               "strip under each clip is untouched and stays at per-step resolution.",
        "items": fast})

    # 3c. derived images that predate their clip
    stale = []
    for v in videos:
        mp4 = ROOT / "runs" / v["video"]
        for kind in ("strip", "frame"):
            rel = v.get(kind)
            if not rel:
                continue
            img = ROOT / "runs" / rel
            if img.exists() and mp4.exists() and img.stat().st_mtime < mp4.stat().st_mtime - 3600:
                if kind == "strip" and v.get("speedup"):
                    continue  # subsampling rewrote the mp4; the strip is from the same render
                stale.append(f"{_c(v['name'])}: {kind} {_c(rel)} is more than an hour older than "
                             "the clip it sits under &mdash; it may describe an earlier render")
    blocks.append({
        "title": "Skill strips or contact sheets older than their clip",
        "severity": "stop" if stale else "ok",
        "why": "A strip or contact sheet regenerated from an earlier render would describe a "
               "different episode than the one playing above it. Re-encodes by "
               "<code>shrink_videos.py</code> are excluded &mdash; those rewrite the mp4 without "
               "re-running the rollout, so the strip still matches.",
        "items": stale})

    # 4. video coverage, at ARM granularity -- a clip of a different arm of the same variant is
    #    not a clip of the headline arm, and on hopper the difference is a factor of two in budget
    missing = []
    have_arm = {(v["env"], v["arm"]) for v in videos}
    have_var = {(v["env"], v["variant"]) for v in videos}
    for env, byv in sorted(shipped.items()):
        for var, arm in sorted(byv.items()):
            if (env, arm) in have_arm:
                continue
            if (env, var) in have_var:
                subs = sorted({v["arm"] for v in videos
                               if v["env"] == env and v["variant"] == var})
                missing.append(
                    f"{env} / {_c(arm)} (the headline {var} arm) has no rendered rollout. The "
                    f"clip shown under {var} is {_c(', '.join(subs))} &mdash; a different arm, "
                    "labelled as such on its own card.")
            else:
                missing.append(f"{env} / {var} ({_c(arm)}) has no rendered rollout at all")
    blocks.append({
        "title": "Headline arms with no video, or with a video of a different arm",
        "severity": "wait",
        "why": "Where a clip is missing the section says so. No clip is ever borrowed from another "
               "environment, another variant or another arm to fill the slot &mdash; but a clip "
               "of a NEIGHBOURING arm of the same variant is easy to misread as the headline "
               "arm's, so those are named here too.",
        "items": missing})

    # 5. budget-scaled arms that are not tagged experimental
    untagged = []
    for c in cells.values():
        if c["experimental"] or c["steps"] is None:
            continue
        env_flat = shipped.get(c["env"], {}).get("flat")
        base = cells.get(f"{c['env']}|{env_flat}", {}).get("steps") if env_flat else None
        if base and c["steps"] != base:
            untagged.append(
                f"{c['env']} / {_c(c['arm'])} runs at {c['steps']:,} env steps against the "
                f"{_c(env_flat)} baseline's {base:,} &mdash; a "
                f"{c['steps']/base:.3g}x mismatch, and the tag is not in "
                "<code>EXPERIMENTAL_TAGS</code>, so it enters the gate as an ordinary arm")
    blocks.append({
        "title": "Budget-scaled arms that the gate treats as ordinary",
        "severity": "stop" if untagged else "ok",
        "why": "<code>analyze_v2.EXPERIMENTAL_TAGS</code> matches the substring "
               "<code>budget</code>, so a scaled arm carrying a different tag passes straight "
               "through. Where the mismatch runs in the baseline's favour the comparison is still "
               "quotable, but only labelled as mismatched &mdash; which is what this block does.",
        "items": untagged})

    # 6. gate-level budget mismatch
    gm = [f"{e}: best flat {_c(v['best_flat'])} at {v['best_flat_budget']:,} steps vs best "
          f"hierarchical {_c(v['best_hier'])} at {v['best_hier_budget']:,}"
          for e, v in gate["per_env"].items() if v.get("budget_matched") is False]
    blocks.append({
        "title": "Gate comparisons drawn across different budgets",
        "severity": "stop" if gm else "ok",
        "why": "The gate reports these rather than subtracting them. A hierarchical win at a "
               "larger budget is not a win; a flat win at a larger budget only raises the bar and "
               "stays quotable, labelled.",
        "items": gm})

    # 7. thin cells
    thin = [f"{c['env']} / {_c(c['arm'])}: n={c['n']} "
            f"(seeds {', '.join('s' + str(s) for s in c['seeds'])})"
            for c in sorted(cells.values(), key=lambda c: (c["env"], c["arm"]))
            if c["n"] < 3 and not c["experimental"]]
    blocks.append({
        "title": "Cells below three seeds",
        "severity": "wait",
        "why": "n &le; 3 is provisional and is said in the same sentence as the number wherever "
               "it appears. Cells below the minimum are excluded from the gate by "
               "<code>analyze_v2</code>.",
        "items": thin})

    # 8. cells whose own seeds disagree on budget
    mixed = [f"{c['env']} / {_c(c['arm'])}: budgets {c['budgets']}"
             for c in cells.values() if c["mixed_budget"]]
    blocks.append({
        "title": "Cells whose seeds were trained at different budgets",
        "severity": "stop" if mixed else "ok",
        "why": "Averaging seeds trained at different budgets fabricates a cell. A reduced-batch "
               "diagnostic once landed inside WalkerWalk <code>nesy</code> and reported 0.224 as "
               "if it were a real result.",
        "items": mixed})

    # 9. bimodal arms
    bimodal = []
    for c in cells.values():
        vals = [v for v in c["success"] if v is not None]
        if len(vals) < 3 or c["success_max"] is None:
            continue
        zeros = sum(1 for v in vals if v < 0.01)
        if zeros and zeros < len(vals) and c["success_max"] > 0.1:
            bimodal.append(
                f"{c['env']} / {_c(c['arm'])}: {zeros} of {len(vals)} seeds at ~0 while the best "
                f"reaches {c['success_max']:.3f} &mdash; the mean ({c['success_mean']:.3f}) "
                "describes no seed that ever ran")
    blocks.append({
        "title": "All-or-nothing arms where the mean describes nothing",
        "severity": "wait",
        "why": "Where seeds are bimodal the mean is not a description of the arm. Every chart on "
               "this page carries per-seed dots for exactly this reason.",
        "items": sorted(bimodal)})

    # 10. training metric vs greedy eval
    gaps = []
    for c in cells.values():
        base = c.get("train_success_on_det_seeds")
        if c["det_n"] and base is not None and c["det_success_mean"] is not None:
            d = c["det_success_mean"] - base
            if abs(d) > 0.1:
                gaps.append(f"{c['env']} / {_c(c['arm'])}: on the {c['det_n']} evaluated seed(s), "
                            f"training tail-mean {base:.3f} vs greedy eval "
                            f"{c['det_success_mean']:.3f} &mdash; a gap of {d:+.3f}")
    blocks.append({
        "title": "Arms where the greedy evaluation disagrees with the training metric",
        "severity": "wait",
        "why": "The training metric is logged with &epsilon;-greedy exploration still on. Where "
               "the two disagree by more than 0.1 the greedy number is the honest one, and both "
               "are shown in the per-arm table.",
        "items": sorted(gaps)})

    # 11. curve alignment inside an arm
    ragged = []
    for c in cells.values():
        lens = {len(runs[k]["curves"].get("policy_diag/primary_success_rate", [])) for k in c["runs"]}
        grids = {(tuple(runs[k]["x"][:2]), tuple(runs[k]["x"][-2:])) for k in c["runs"]}
        if len(lens) > 1 or len(grids) > 1:
            ragged.append(f"{c['env']} / {_c(c['arm'])}: seeds log {sorted(lens)} points on "
                          f"{len(grids)} different step grids &mdash; the mean/±1σ band on this "
                          "arm is drawn from the subset that agrees, and the rest are dropped")
    blocks.append({
        "title": "Arms whose seeds do not share a logging grid",
        "severity": "stop" if ragged else "ok",
        "why": "Every band on this page is a mean over seeds at the same env-step index. Seeds "
               "logged at different cadences cannot be averaged that way, and the chart code "
               "drops the odd ones out rather than stretching them &mdash; which would silently "
               "shrink n below the number printed in the legend.",
        "items": ragged})

    # 12. greedy-eval coverage
    nodet = []
    for env, byv in sorted(shipped.items()):
        for v, arm in sorted(byv.items()):
            c = cells.get(f"{env}|{arm}")
            if not c:
                continue
            if not c["det_n"]:
                nodet.append(f"{env} / {_c(arm)}: no greedy evaluation on disk &mdash; only the "
                             "training metric is available for this arm")
            elif c["det_n"] < 3:
                nodet.append(f"{env} / {_c(arm)}: greedy evaluation covers {c['det_n']} of "
                             f"{c['n']} trained seeds &mdash; provisional")
    blocks.append({
        "title": "Greedy-evaluation coverage of the headline arms",
        "severity": "wait",
        "why": "The deterministic eval is the honest number, but it was queued per checkpoint "
               "rather than per arm and does not cover everything. Where it is missing the page "
               "shows the training metric and says so; it never fills the gap with a neighbour.",
        "items": nodet})

    # 13. eval CSVs that could not be bound
    blocks.append({
        "title": "Evaluation CSVs that could not be bound to a checkpoint",
        "severity": "wait",
        "why": "Held-out and perturbation sweeps are resolved back to the checkpoint their queue "
               "script names, and the env/meta columns inside the CSV are checked against that "
               "checkpoint's config. Anything that does not bind is dropped from the charts and "
               "listed here rather than plotted on a guess.",
        "items": [_c(u) for u in oos_unresolved]})

    return blocks


# --------------------------------------------------------------------------- #
# artifact edition -- the same page with every asset inlined
# --------------------------------------------------------------------------- #

def _ffprobe_frames(path: Path) -> tuple[int | None, float | None]:
    """(frame count, fps) of an mp4, or (None, None) if ffprobe is not available."""
    import subprocess
    fp = _ffprobe()
    if fp is None:
        return None, None
    try:
        out = subprocess.run(
            [fp, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True, timeout=120).stdout.strip()
        rate, n = out.split(",")[0], out.split(",")[1]
        num, _, den = rate.partition("/")
        return int(n), (float(num) / float(den or 1))
    except Exception:  # noqa: BLE001
        return None, None


def _ffprobe() -> str | None:
    import shutil
    for c in FFMPEG_CANDIDATES:
        cand = c.replace("ffmpeg", "ffprobe")
        if shutil.which(cand):
            return shutil.which(cand)
        if Path(cand).exists():
            return cand
    return None


FFMPEG_CANDIDATES = [
    "ffmpeg",
    str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe"),
]


def _ffmpeg() -> str | None:
    import shutil
    for c in FFMPEG_CANDIDATES:
        if shutil.which(c):
            return shutil.which(c)
        if Path(c).exists():
            return c
    return None


def _data_uri(raw: bytes, mime: str) -> str:
    import base64
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def inline_assets(data: dict, cache: Path) -> dict[str, float]:
    """Rewrite every asset path in the payload to a data URI, re-encoded to fit 16 MB.

    A published Artifact cannot reach the filesystem, so the shareable edition has to carry its
    own video. 24 MB of MP4 does not fit, and a page that silently drops the clips is not the
    same page -- so they are re-encoded (440px wide, no audio) rather than omitted, and the
    contact sheets are downscaled to JPEG. Only the clips the page actually shows are carried;
    byte-identical re-renders were already hidden and stay out.
    """
    import subprocess
    from PIL import Image

    ff = _ffmpeg()
    cache.mkdir(parents=True, exist_ok=True)
    sizes = {"video": 0.0, "image": 0.0}

    def img_uri(rel: str, max_w: int, fmt: str, quality: int) -> str | None:
        src = ROOT / "runs" / rel
        if not src.exists():
            return None
        out = cache / (rel.replace("/", "_").rsplit(".", 1)[0] + "." + fmt.lower())
        if not out.exists():
            im = Image.open(src).convert("RGB")
            if im.width > max_w:
                im = im.resize((max_w, max(1, round(im.height * max_w / im.width))),
                               Image.LANCZOS)
            im.save(out, fmt, quality=quality, optimize=True)
        raw = out.read_bytes()
        sizes["image"] += len(raw)
        return _data_uri(raw, "image/jpeg" if fmt == "JPEG" else "image/png")

    def vid_uri(rel: str) -> str | None:
        src = ROOT / "runs" / rel
        if not src.exists():
            return None
        out = cache / rel.replace("/", "_")
        if not out.exists():
            if ff is None:
                print("  no ffmpeg -- cannot build the inline edition")
                return None
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src),
                            "-vf", "scale=400:-2:flags=lanczos,fps=24", "-c:v", "libx264",
                            "-crf", "35", "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", str(out)], check=True)
        raw = out.read_bytes()
        sizes["video"] += len(raw)
        return _data_uri(raw, "video/mp4")

    for v in data["videos"]:
        if v.get("duplicate_hidden"):
            v["video"] = v["strip"] = v["frame"] = None
            continue
        v["video"] = vid_uri(v["video"]) if v["video"] else None
        v["strip"] = img_uri(v["strip"], 760, "PNG", 90) if v["strip"] else None
        v["frame"] = img_uri(v["frame"], 660, "JPEG", 60) if v["frame"] else None
    for p in data["fig6"]:
        p["img"] = img_uri(p["img"], 1000, "JPEG", 78) or p["img"]
    data["inline"] = True
    return {k: v / 1e6 for k, v in sizes.items()}


def _git(*args: str) -> str:
    try:
        import subprocess
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def collect_all(args) -> dict:
    dirs = [ROOT / d for d in args.dirs]
    print("scanning checkpoints ...")
    runs = load_runs(dirs)
    print(f"  {len(runs)} runs")
    det = load_det(ROOT / "runs/det", runs)
    print(f"  {len(det)} deterministic evals")
    rob = load_robustness(ROOT / "runs/robustness", runs)
    print(f"  {len(rob)} robustness sweeps")
    det_disagree = cross_check_det(det, rob)
    n_overlap = sum(1 for k in rob if k in det)
    print(f"  {n_overlap} checkpoints evaluated twice, {len(det_disagree)} disagreements")
    print(f"  +{merge_robustness_into_det(det, rob)} greedy evals from sweep level 0.0"
          f" -> {len(det)} total")
    vids = load_videos(ROOT / "runs/videos", ROOT / "runs/frames", runs)
    print(f"  {len(vids)} videos")
    oos, oos_bad = load_oos(ROOT / "runs/oos", runs)
    for u in oos_bad:
        print(f"  OOS UNRESOLVED {u}")
    cells = build_cells(runs, det)
    print(f"  {len(cells)} cells")
    rules = load_rules(runs)
    fig6 = load_fig6(ROOT / "runs/fig6", runs)
    ppo = load_ppo(ROOT / "runs/ppo")
    print(f"  {sum(len(v) for v in ppo.values())} PPO baseline evals")

    print("gate (imported from analyze_v2) ...")
    matrix = analyze_v2.collect(dirs, min_seeds=args.min_seeds)
    verdict = analyze_v2.gate(matrix, min_seeds=args.min_seeds)

    envs = [e for e in ENV_ORDER if any(r["env"] == e for r in runs.values())]
    envs += sorted({r["env"] for r in runs.values()} - set(envs))
    shipped = pick_shipped(cells)

    # A byte-identical re-render is one clip, shown once. It stays in the provenance table so the
    # duplication itself is visible.
    seen_hash: set[str] = set()
    seen_ck: set[str] = set()
    for v in vids:
        v["warnings"] = [n for n in v["notes"]
                         if "byte-identical" in n or "!=" in n or "not found" in n]
        if v["episode_return"] is not None and abs(v["episode_return"]) < 1e-9:
            v["warnings"].append("this greedy episode returned 0.0 — read it against the arm's "
                                 "seed spread above before treating it as typical")
        if v["frames"] is not None and v["frames"] < 100:
            v["warnings"].append(f"episode terminated after {v['frames']} frames")
        # Three filenames render one Go1 checkpoint at the same render seed: that is one clip
        # and it is shown once. Several renders of one checkpoint at DIFFERENT render seeds are
        # not redundant -- hopper flat returns 0.0 on r0 and hops on r3 and r5, which is the
        # arm's bimodality made visible -- so those are all kept. Everything stays listed in the
        # provenance table either way.
        ident = (v["checkpoint"], v["render_seed"])
        dup_ck = bool(v["checkpoint"]) and ident in seen_ck
        if dup_ck:
            v["notes"].append("another render of the same checkpoint at the same render seed is "
                              "already shown above")
        elif v["checkpoint"] in {c for c, _ in seen_ck}:
            v["notes"].append("same checkpoint as a clip above, rendered at a different seed")
        v["duplicate_hidden"] = v["md5"] in seen_hash or dup_ck
        seen_hash.add(v["md5"])
        if v["checkpoint"]:
            seen_ck.add(ident)

    ppo_note = ppo_verdicts(ppo, cells, shipped)
    # A clip of an experimental arm is real behaviour but not a peer of the shipped arms, and
    # the card has to say so: `hopper_flat_budget8x_hopping` is a flat baseline at EIGHT times
    # the budget, and it hops where the shipped one falls over.
    for v in vids:
        c = cells.get(f"{v['env']}|{v['arm']}") if v["env"] and v["arm"] else None
        v["experimental"] = bool(c and c["experimental"])
        v["arm_steps"] = c["steps"] if c else None
        v["shipped"] = bool(c and shipped.get(v["env"], {}).get(v["variant"]) == v["arm"])
        base = shipped.get(v["env"], {}).get("flat")
        bsteps = cells.get(f"{v['env']}|{base}", {}).get("steps") if base else None
        if v["experimental"] and v["arm_steps"] and bsteps and v["arm_steps"] != bsteps:
            v["warnings"].insert(0, f"experimental arm — {v['arm_steps'] / bsteps:.3g}x the "
                                    f"{base} baseline's budget; not a peer of the shipped arms")
        elif v["experimental"]:
            v["warnings"].insert(0, "experimental arm — changed hyperparameters or rules; not a "
                                    "peer of the shipped arms")

    audit = build_audit(runs, cells, vids, det, rob, verdict, shipped, oos_bad)
    audit.append({
        "title": "Agreement between the two greedy-evaluation sources",
        "severity": "stop" if det_disagree else "ok",
        "why": f"{n_overlap} checkpoints were evaluated twice &mdash; once by "
               "<code>run_det_eval.sh</code> and once as the zero-noise point of an action-noise "
               "sweep. Both call <code>robustness_eval.py</code> at 64 episodes, so they must "
               "agree to the last digit; the page merges them into one greedy column and this is "
               "the check that says the merge is safe.",
        "items": det_disagree})
    audit.insert(0, {
        "title": "External baseline: what Brax PPO does to each conclusion",
        "severity": "stop" if ppo_note else "ok",
        "why": "The gate compares hierarchical PQN against flat PQN. The paper also carries a PPO "
               "baseline, and we have one on two environments. Where PPO beats both sides, the "
               "internal comparison is still valid and still not a claim about flat control in "
               "general.",
        "items": [f"<b>{e}</b> &mdash; {t}" for e, t in sorted(ppo_note.items())]})
    n_find = sum(len(b["items"]) for b in audit)
    print(f"  audit: {len(audit)} checks, {n_find} findings")

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "runs": runs,
        "det": det,
        "robustness": rob,
        "videos": vids,
        "oos": oos,
        "cells": cells,
        "gate": verdict,
        "envs": envs,
        "env_meta": ENV_META,
        "shipped": shipped,
        "rules": rules,
        "fig6": fig6,
        "audit": audit,
        "ppo": ppo,
        "ppo_note": ppo_note,
        "n_checkpoints": len(runs),
        "dirs": [str(d.relative_to(ROOT)).replace("\\", "/") for d in dirs],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["runs/verify", "runs/viper"])
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/results_dashboard.html")
    ap.add_argument("--dump-json", default="")
    ap.add_argument("--inline-assets", action="store_true",
                    help="re-encode and embed every asset as a data URI, for publishing as an "
                         "Artifact (which cannot read the filesystem)")
    args = ap.parse_args(argv)

    data = collect_all(args)
    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(data, indent=1, default=str))
        print(f"wrote {args.dump_json}")

    if args.inline_assets:
        mb = inline_assets(data, ROOT / "runs/_inline_cache")
        print(f"  inlined {mb['video']:.2f} MB video + {mb['image']:.2f} MB images")

    from dashboard_render import render  # noqa: PLC0415

    out = ROOT / args.out
    out.write_text(render(data), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
