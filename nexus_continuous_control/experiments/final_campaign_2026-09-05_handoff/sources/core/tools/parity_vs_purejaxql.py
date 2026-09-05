"""V1.2 — parity check: our flat AC-PQN against upstream purejaxql's.

Why this is the highest-value single check in the campaign
----------------------------------------------------------
Every hierarchical claim in this project is stated *relative to the flat baseline* — "nesy
beats flat by 1.71x", "neural passes the return-ratio gate on 5/5". If our `flat` mode is not
actually a competent PQN, those ratios measure our own baseline's weakness rather than the
hierarchy's strength, and the entire results table quietly inherits the error.

So: run our `META_POLICY_TYPE=flat` arm and upstream `purejaxql/pqn_mujoco_playground.py` on
the same environment with the same environment-step budget, several seeds each, and compare
the learning curves and the final return.

What is and is not matched
--------------------------
**Matched:** environment (CartpoleBalance), total environment steps, seed count, hardware.
**Not matched:** hyperparameters. Each arm runs at its own shipped recipe — upstream's tuned
`pqn_playground_dm_suite` defaults (larger nets, NUM_STEPS=32, GAMMA 0.995, LAMBDA 0.70)
against our `flat_cartpole_balance.yaml`. That is deliberate: the question is whether our
baseline reaches the performance level a competent PQN reaches on this task, not whether two
implementations agree line-for-line under identical settings. Forcing our hyperparameters
onto upstream would test a configuration its authors never endorsed.

Usage
-----
    python tools/parity_vs_purejaxql.py --seeds 0,1,2 --timesteps 9830400 \\
        --ours runs/verify --out runs/parity

Writes `upstream.json` (per-seed curves), `parity.png`, and `parity.json` (the verdict).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PUREJAXQL = Path(__file__).resolve().parents[2] / "vendor" / "purejaxql"

# The gate from docs/VERIFICATION_PLAN.md.
TOLERANCE = 0.15


def _upstream_config(env: str, timesteps: int, seed: int) -> dict[str, Any]:
    """Upstream's own dm-suite recipe, with only the env/budget/seed pinned."""
    import yaml  # noqa: PLC0415

    alg = yaml.safe_load(
        (PUREJAXQL / "purejaxql" / "config" / "alg" / "pqn_playground_dm_suite.yaml").read_text()
    )
    cfg = dict(alg)
    cfg.update(
        {
            "ENV_NAME": env,
            "TOTAL_TIMESTEPS": int(timesteps),
            "NUM_SEEDS": 1,
            "SEED": int(seed),
            "RETURN_METRICS": True,
            "WANDB_MODE": "disabled",
            "ENTITY": "",
            "PROJECT": "",
            "SAVE_PATH": None,
            "HYP_TUNE": False,
        }
    )
    return cfg


def _force_jax_impl() -> None:
    """Make upstream load Playground envs on the JAX backend, not MuJoCo-Warp.

    `PlaygroundVecGymnaxWrapper.__init__` calls `registry.get_default_config(env_name)` and
    loads with it untouched, so it inherits whatever `impl` the installed Playground defaults
    to. With the warp backend in this environment that dies inside `mjx.put_model` with
    `module 'warp.types' has no attribute 'warp_type_to_np_dtype'` — a warp/mujoco version
    skew, not anything about PQN. Our own adapter always pins `impl='jax'`
    (`PLAYGROUND_IMPL`), so pinning it here too is what makes the two arms comparable rather
    than a thumb on the scale. Patched at the registry rather than by editing vendored code.
    """
    from mujoco_playground import registry  # noqa: PLC0415

    if getattr(registry.get_default_config, "_jax_impl_patched", False):
        return
    original = registry.get_default_config

    def patched(env_name, *a, **k):
        cfg = original(env_name, *a, **k)
        try:
            cfg.impl = "jax"
        except Exception:
            pass
        return cfg

    patched._jax_impl_patched = True  # type: ignore[attr-defined]
    registry.get_default_config = patched


def run_upstream(env: str, timesteps: int, seeds: list[int]) -> dict[str, Any]:
    sys.path.insert(0, str(PUREJAXQL))
    import jax  # noqa: PLC0415
    import wandb  # noqa: PLC0415

    wandb.init(mode="disabled")  # the module logs unconditionally inside its callback
    _force_jax_impl()

    from purejaxql.pqn_mujoco_playground import make_train  # noqa: PLC0415

    curves, finals = [], []
    for seed in seeds:
        cfg = _upstream_config(env, timesteps, seed)
        print(f"[upstream] seed {seed} ...", flush=True)
        train = jax.jit(jax.vmap(make_train(cfg)))
        outs = jax.block_until_ready(train(jax.random.split(jax.random.PRNGKey(seed), 1)))
        m = outs["metrics"]
        key = next(
            (k for k in ("test/returned_episode_returns", "returned_episode_returns") if k in m),
            None,
        )
        if key is None:
            raise KeyError(f"no return metric in upstream output; keys={sorted(m)[:20]}")
        series = np.asarray(m[key]).reshape(-1)
        curves.append(series.tolist())
        finals.append(float(np.mean(series[-max(1, len(series) // 10):])))
        print(f"[upstream] seed {seed} final {finals[-1]:.1f}", flush=True)
    return {"curves": curves, "finals": finals, "metric": key, "seeds": seeds}


def load_ours(runs_dir: Path, env: str) -> dict[str, Any]:
    """Read our flat-arm checkpoints and pull the training-return curve from each."""
    curves, finals, seeds = [], [], []
    for pkl in sorted(runs_dir.rglob("*.pkl")):
        try:
            with open(pkl, "rb") as fh:
                ck = pickle.load(fh)
        except Exception:
            continue
        cfg = ck.get("config", {}) or {}
        if str(cfg.get("ENV_NAME")) != env:
            continue
        if str(cfg.get("META_POLICY_TYPE", "")).lower() != "flat":
            continue
        m = ck.get("metrics", {}) or {}
        if "rollout/episode_return" not in m:
            continue
        s = np.asarray(m["rollout/episode_return"]).reshape(-1)
        curves.append(s.tolist())
        finals.append(float(np.mean(s[-max(1, len(s) // 10):])))
        seeds.append(cfg.get("SEED"))
        print(f"[ours] {pkl.name} final {finals[-1]:.1f}")
    return {"curves": curves, "finals": finals, "seeds": seeds}


def _band(ax, curves, label, color, total_steps):
    arrs = [np.asarray(c) for c in curves if len(c) > 1]
    if not arrs:
        return
    n = min(len(a) for a in arrs)
    stack = np.stack([a[:n] for a in arrs])
    # Both arms run the same environment-step budget but a different number of updates
    # (different NUM_STEPS), so the x axis must be env steps, not update index.
    x = np.linspace(0, total_steps, n)
    mean, std = stack.mean(0), stack.std(0)
    ax.plot(x, mean, label=f"{label} (n={len(arrs)})", color=color, lw=1.8)
    if len(arrs) > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, lw=0)


def load_glob(paths: list[Path]) -> dict[str, Any]:
    """Load an explicit list of checkpoints as one arm."""
    curves, finals = [], []
    for pkl in sorted(paths):
        with open(pkl, "rb") as fh:
            ck = pickle.load(fh)
        s = np.asarray(ck["metrics"]["rollout/episode_return"]).reshape(-1)
        curves.append(s.tolist())
        finals.append(float(np.mean(s[-max(1, len(s) // 10):])))
    return {"curves": curves, "finals": finals}


def make_plot(
    ours: dict,
    upstream: dict,
    env: str,
    total_steps: int,
    out: Path,
    extra: dict | None = None,
    extra_label: str = "",
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    _band(ax, upstream["curves"], "purejaxql AC-PQN (upstream)", "#8792a2", total_steps)
    _band(ax, ours["curves"], "ours — shipped config", "#343C96", total_steps)
    if extra and extra.get("curves"):
        _band(ax, extra["curves"], extra_label, "#1B6B45", total_steps)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title(
        f"V1.2 parity — {env}, matched step budget, each at its own recipe",
        fontsize=10,
        loc="left",
    )
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--env", default="CartpoleBalance")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--timesteps", type=int, default=9830400)
    ap.add_argument("--ours", default="runs/verify")
    ap.add_argument("--out", default="runs/parity")
    ap.add_argument("--skip-upstream", action="store_true", help="reuse a saved upstream.json")
    ap.add_argument("--extra", default=None, help="glob of checkpoints to overlay as a third arm")
    ap.add_argument("--extra-label", default="ours — fixed exploration schedule")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    up_path = out_dir / "upstream.json"

    if args.skip_upstream and up_path.exists():
        upstream = json.loads(up_path.read_text())
    else:
        upstream = run_upstream(args.env, args.timesteps, seeds)
        up_path.write_text(json.dumps(upstream), encoding="utf-8")

    ours = load_ours(Path(args.ours), args.env)
    if not ours["curves"]:
        print(f"no flat-arm checkpoints for {args.env} under {args.ours}")
        return 2

    extra = None
    if args.extra:
        from glob import glob as _glob  # noqa: PLC0415

        files = [Path(x) for x in _glob(args.extra)]
        if files:
            extra = load_glob(files)
            print(f"[extra] {len(files)} runs, finals {[round(x, 1) for x in extra['finals']]}")
    make_plot(
        ours, upstream, args.env, args.timesteps, out_dir / "parity.png",
        extra=extra, extra_label=args.extra_label,
    )

    o = float(np.mean(ours["finals"]))
    u = float(np.mean(upstream["finals"]))
    ratio = o / u if u else float("nan")
    passed = bool(abs(ratio - 1.0) <= TOLERANCE)
    verdict = {
        "env": args.env,
        "total_timesteps": args.timesteps,
        "ours_final_mean": o,
        "ours_finals": ours["finals"],
        "upstream_final_mean": u,
        "upstream_finals": upstream["finals"],
        "ratio_ours_over_upstream": ratio,
        "tolerance": TOLERANCE,
        "gate": "PASS" if passed else "FAIL",
    }
    if extra is not None:
        e = float(np.mean(extra["finals"]))
        verdict["fixed_final_mean"] = e
        verdict["fixed_finals"] = extra["finals"]
        verdict["fixed_ratio"] = e / u if u else float("nan")
        verdict["fixed_gate"] = "PASS" if abs(e / u - 1.0) <= TOLERANCE else "FAIL"
        verdict["fixed_label"] = args.extra_label
    (out_dir / "parity.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"V1.2 PARITY — {args.env} @ {args.timesteps:,} env steps")
    print("=" * 72)
    print(f"  ours (flat)        {o:10.1f}   seeds {[round(x, 1) for x in ours['finals']]}")
    print(f"  purejaxql upstream {u:10.1f}   seeds {[round(x, 1) for x in upstream['finals']]}")
    print(f"  ratio              {ratio:10.3f}   gate |ratio-1| <= {TOLERANCE}")
    print(f"  VERDICT            {verdict['gate']}")
    print("=" * 72)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
