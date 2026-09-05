#!/usr/bin/env python3
"""V3.1 — rule coverage for the six hand-written symbolic policies.

Question the plan asks: is every branch of every symbolic rule *reachable*, and is the NeSy
mask ever all-zero (which would make `where(mask, q, -1e9)` degenerate — argmax over a row of
-1e9 returns skill 0 regardless of what the meta-Q learned)?

Method. The policies read their features from the semantic `info` dict and only fall back to
observation indices when a key is missing (`policies/common.py:info_value`). So we can drive
them directly on a synthetic grid of semantic states without instantiating any environment —
this is a property of the *rules*, not of the physics, and it runs on CPU in seconds.

Two passes per policy:
  * random pass — uniform over a box that generously covers every threshold in the rule,
    for branch-reachability and all-zero-mask counting on a large sample;
  * slice pass — a dense 2-D sweep of the two features the rule branches on hardest, with
    the others pinned, rendered as the truth-table grid the plan asks for.

Exit code is non-zero if any branch is unreachable or any all-zero mask row exists.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# Feature boxes. Ranges are deliberately wider than the physical operating range so that a
# branch counts as "unreachable" only when the rule algebra cannot select it, never because
# the sampler was too timid to reach its threshold.
SCALAR = "scalar"

SPECS: dict[str, dict[str, Any]] = {
    "cartpole_balance": {
        "keys": {
            "cart_position": (-1.5, 1.5),
            "pole_angle": (-1.0, 1.0),
            "cart_velocity": (-3.0, 3.0),
            "pole_angular_velocity": (-4.0, 4.0),
        },
        "slice": ("cart_position", "pole_angle"),
        "pin": {"cart_velocity": 0.0, "pole_angular_velocity": 0.0},
    },
    "cheetah_run": {
        "keys": {
            "x_velocity": (-2.0, 12.0),
            "torso_pitch": (-1.5, 1.5),
            "joint_speed": (0.0, 15.0),
        },
        "slice": ("x_velocity", "torso_pitch"),
        "pin": {"joint_speed": 2.0},
    },
    "walker_walk": {
        "keys": {
            "torso_height": (0.3, 1.7),
            "torso_pitch": (-1.7, 1.7),
            "x_velocity": (-1.0, 3.5),
            "joint_speed": (0.0, 15.0),
        },
        "slice": ("x_velocity", "torso_height"),
        "pin": {"torso_pitch": 0.0, "joint_speed": 2.0},
    },
    "hopper_hop": {
        "keys": {
            "torso_height": (0.3, 1.7),
            "torso_pitch": (-1.7, 1.7),
            "x_velocity": (-1.0, 4.0),
            "joint_speed": (0.0, 15.0),
        },
        "slice": ("x_velocity", "torso_height"),
        "pin": {"torso_pitch": 0.0, "joint_speed": 2.0},
    },
    "go1_joystick": {
        "keys": {
            "base_height": (0.10, 0.45),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-1.0, 1.0),
            "yaw_rate": (-2.0, 2.0),
            "command_x": (-1.5, 1.5),
            "command_y": (-1.0, 1.0),
            "command_yaw": (-1.5, 1.5),
        },
        "slice": ("command_x", "command_yaw"),
        "pin": {
            "base_height": 0.32,
            "roll": 0.0,
            "pitch": 0.0,
            "lin_vel_x": 0.0,
            "lin_vel_y": 0.0,
            "yaw_rate": 0.0,
            "command_y": 0.0,
        },
    },
    "panda_pick_cube": {
        # tcp/cube/target are 3-vectors; sampling them independently over the workspace box
        # covers every (dist_tcp_cube, cube_height) combination the rule tests.
        "vec_keys": {
            "tcp_pos": ((-0.2, -0.2, 0.0), (0.2, 0.2, 0.4)),
            "cube_pos": ((-0.2, -0.2, 0.0), (0.2, 0.2, 0.4)),
            "target_pos": ((-0.2, -0.2, 0.05), (0.2, 0.2, 0.4)),
        },
        "keys": {"gripper_open": (0.0, 1.0), "grasped": (0.0, 1.0)},
        "slice": ("dist_tcp_cube", "cube_height"),
        "slice_ranges": {"dist_tcp_cube": (0.0, 0.35), "cube_height": (0.0, 0.30)},
        "pin": {"gripper_open": 0.5, "grasped": 0.0},
    },
}


def _load(name: str):
    return importlib.import_module(f"nexus_continuous.policies.{name}")


def _random_states(spec: dict[str, Any], n: int, rng: np.random.Generator) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key, (lo, hi) in spec.get("keys", {}).items():
        info[key] = rng.uniform(lo, hi, size=n).astype(np.float32)
    for key, (lo, hi) in spec.get("vec_keys", {}).items():
        info[key] = rng.uniform(np.array(lo), np.array(hi), size=(n, 3)).astype(np.float32)
    return info


def _slice_states(policy: str, spec: dict[str, Any], res: int) -> tuple[dict[str, Any], Any, Any]:
    """Dense 2-D sweep of the two branch-dominant features, others pinned."""
    fx, fy = spec["slice"]
    ranges = spec.get("slice_ranges", {})
    rx = ranges.get(fx) or spec["keys"][fx]
    ry = ranges.get(fy) or spec["keys"][fy]
    ax = np.linspace(rx[0], rx[1], res, dtype=np.float32)
    ay = np.linspace(ry[0], ry[1], res, dtype=np.float32)
    gx, gy = np.meshgrid(ax, ay, indexing="xy")
    flat_x, flat_y = gx.ravel(), gy.ravel()
    n = flat_x.size

    info: dict[str, Any] = {}
    for key, val in spec.get("pin", {}).items():
        info[key] = np.full(n, val, dtype=np.float32)

    if policy == "panda_pick_cube":
        # Realize (dist_tcp_cube, cube_height) geometrically: cube at the swept height,
        # tcp displaced from it by the swept distance along +x.
        cube = np.stack([np.zeros(n), np.zeros(n), flat_y], axis=-1).astype(np.float32)
        tcp = cube + np.stack([flat_x, np.zeros(n), np.zeros(n)], axis=-1).astype(np.float32)
        info["cube_pos"] = cube
        info["tcp_pos"] = tcp
        info["target_pos"] = np.tile(np.array([0.1, 0.1, 0.15], np.float32), (n, 1))
    else:
        info[fx] = flat_x
        info[fy] = flat_y
    return info, (ax, ay), (res, res)


def _obs_for(info: dict[str, Any], n: int) -> np.ndarray:
    """Dummy observation. Every feature the rules read is supplied via `info`, so the
    obs-index fallbacks in `common.safe_index` are never consulted; this only satisfies
    `actor_obs` and the batch-shape contract."""
    return np.zeros((n, 32), dtype=np.float32)


def analyze(policy: str, n: int, res: int, seed: int) -> dict[str, Any]:
    mod = _load(policy)
    spec = SPECS[policy]
    names = list(mod.SKILL_NAMES)
    rng = np.random.default_rng(seed)

    info = _random_states(spec, n, rng)
    obs = _obs_for(info, n)
    skills = np.asarray(mod.symbolic_meta_policy(obs, info))
    mask = np.asarray(mod.skill_mask(obs, info)).astype(bool)

    counts = {names[i]: int((skills == i).sum()) for i in range(len(names))}
    unreachable = [names[i] for i in range(len(names)) if counts[names[i]] == 0]
    all_zero = int((~mask.any(axis=-1)).sum())
    avail = {names[i]: float(mask[:, i].mean()) for i in range(len(names))}
    # A selected-but-masked-out state is the pathology Eq. 5 exists to prevent: the symbolic
    # rule names a skill the NeSy mask forbids, so the two disagree about what is legal.
    selected_masked_out = int((~mask[np.arange(n), skills]).sum())

    sinfo, axes, shape = _slice_states(policy, spec, res)
    sobs = _obs_for(sinfo, next(iter(sinfo.values())).shape[0])
    grid = np.asarray(mod.symbolic_meta_policy(sobs, sinfo)).reshape(shape)
    smask = np.asarray(mod.skill_mask(sobs, sinfo)).astype(bool)
    grid_zero = int((~smask.any(axis=-1)).sum())

    return {
        "policy": policy,
        "skill_names": names,
        "n_states": n,
        "branch_counts": counts,
        "branch_fractions": {k: v / n for k, v in counts.items()},
        "unreachable_branches": unreachable,
        "all_zero_mask_rows": all_zero + grid_zero,
        "mask_availability": avail,
        "selected_but_masked_out": selected_masked_out,
        "pass": (not unreachable) and (all_zero + grid_zero == 0),
        "_grid": grid,
        "_axes": axes,
        "_slice": spec["slice"],
    }


def plot(results: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    palette = ["#4C6EF5", "#F59F00", "#0CA678", "#E03131", "#7048E8"]
    ncol = 3
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.3 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, results):
        names = r["skill_names"]
        k = len(names)
        cmap = ListedColormap(palette[:k])
        norm = BoundaryNorm(np.arange(-0.5, k, 1.0), k)
        xs, ys = r["_axes"]
        ax.imshow(
            r["_grid"],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            interpolation="nearest",
        )
        fx, fy = r["_slice"]
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        status = "PASS" if r["pass"] else "FAIL"
        ax.set_title(f"{r['policy']} — {status}", fontsize=11)
        handles = [
            plt.Line2D([], [], marker="s", ls="", color=palette[i], label=f"{i} {names[i]}")
            for i in range(k)
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.85)

    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.suptitle(
        "V3.1 rule coverage — symbolic meta-policy over a synthetic state grid "
        "(other features pinned)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="random states per policy")
    ap.add_argument("--res", type=int, default=240, help="slice grid resolution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/audit"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = [analyze(p, args.n, args.res, args.seed) for p in SPECS]

    print(f"{'policy':<20} {'branches reached':<20} {'all-zero mask':<14} verdict")
    for r in results:
        k = len(r["skill_names"])
        reached = k - len(r["unreachable_branches"])
        print(
            f"{r['policy']:<20} {f'{reached}/{k}':<20} "
            f"{r['all_zero_mask_rows']:<14} {'PASS' if r['pass'] else 'FAIL'}"
        )
        for name, frac in r["branch_fractions"].items():
            print(f"    {name:<24} {frac * 100:6.2f}%  mask avail {r['mask_availability'][name] * 100:6.2f}%")
        if r["unreachable_branches"]:
            print(f"    UNREACHABLE: {r['unreachable_branches']}")
        if r["selected_but_masked_out"]:
            print(
                f"    NOTE: symbolic rule selected a mask-forbidden skill in "
                f"{r['selected_but_masked_out']} / {r['n_states']} states"
            )

    plot(results, args.out / "rule_coverage.png")
    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (args.out / "rule_coverage.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out / 'rule_coverage.json'}")

    failed = [r["policy"] for r in results if not r["pass"]]
    if failed:
        print(f"\nGATE FAILED: {failed}")
        return 1
    print("\nGATE PASSED: every branch reachable, no all-zero mask row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
