"""Action-noise robustness sweep for a saved PPO baseline checkpoint.

Why this exists
---------------
G1 put PPO on Go1JoystickRoughTerrain at the budget-matched 20M steps and it scored 0.9993 /
0.9991 / 0.9904 clean success. If that holds, clean in-distribution success no longer separates
anything on this env, and Pillar A's reliability claim has to be argued under perturbation
instead -- where the campaign has no PPO column at all.

`tools/robustness_eval.py` cannot read a PPO checkpoint (it expects a NEXUS `config` key), and
`tools/train_ppo_baseline.py` only evaluates at noise 0 and only right after training. So this
rebuilds brax's inference function from the saved params and runs the SAME
`robustness_eval.evaluate` every NEXUS cell is scored by, at several noise levels.

Nothing about the metric is re-implemented: same `evaluate`, same `task_metrics`, same episode
count, same eval seed. The selector is `train_ppo_baseline.make_ppo_selector` itself, imported
rather than copied, so the asymmetric state/privileged_state observation split it documents
cannot drift out of sync here.

Self-check
----------
The level-0.0 row must reproduce the number `train_ppo_baseline` already wrote into
`<stem>.csv`. `--verify-against` makes that comparison and refuses to write on a mismatch: a
rebuilt network that normalizes against the wrong statistics would produce a baseline that looks
worse than PPO is, which is exactly the failure that would make the resulting table dishonest.

    JAX_PLATFORMS=cpu python tools/_ppo_robust.py \
        --checkpoint runs/ppo/go1_rough_ppo_s0.pkl \
        --match-checkpoint runs/verify/go1_rough_flat_v2_s0.pkl \
        --levels 0.0,0.05,0.1,0.2,0.3 --out runs/robustness/go1rough_ppo_s0.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp

from nexus_continuous.envs.playground_adapter import build_playground_env
from nexus_continuous.policies.registry import load_policy_module
from robustness_eval import evaluate
from train_ppo_baseline import _shim_device_put_replicated, make_ppo_selector


def rebuild_inference(ck: dict[str, Any], bundle):
    """Reconstruct brax's `make_inference_fn` from the saved params.

    `ppo.train` returns params as (normalizer_params, policy_params, value_params); brax's policy
    factory wants the first two. The network factory kwargs are replayed verbatim out of the
    saved `ppo_config`, so the rebuilt network is structurally the trained one -- including
    `policy_obs_key`/`value_obs_key`, which is what keeps the dict observation wired correctly.
    """
    _shim_device_put_replicated()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from brax.training import types as brax_types

    ppo_cfg = dict(ck["ppo_config"])
    net_kwargs = dict(ppo_cfg.get("network_factory", {}) or {})

    # Observation sizes come from the saved NORMALIZER, not from the env. `params[0]` is brax's
    # running_statistics state and its `mean` has exactly the pytree structure the network was
    # built against -- a dict for the locomotion envs' asymmetric state/privileged_state split.
    # Reading it here means the rebuilt network cannot disagree with the trained one about
    # observation structure, which reading the env back could.
    mean = ck["params"][0].mean
    if hasattr(mean, "items"):
        obs_size = {k: int(v.shape[-1]) for k, v in mean.items()}
    else:
        obs_size = int(mean.shape[-1])

    preprocess = (running_statistics.normalize
                  if ppo_cfg.get("normalize_observations", True)
                  else brax_types.identity_observation_preprocessor)

    network = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)(
        observation_size=obs_size,
        action_size=bundle.action_dim,
        preprocess_observations_fn=preprocess,
    )
    make_inference_fn = ppo_networks.make_inference_fn(network)
    params = tuple(ck["params"])[:2]
    return make_inference_fn, params, ppo_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="runs/ppo/<stem>.pkl")
    ap.add_argument("--match-checkpoint", required=True,
                    help="the NEXUS checkpoint the PPO cell was budget-matched to; supplies the "
                         "eval config and TASK_POLICY so the metric is identical")
    ap.add_argument("--levels", default="0.0,0.05,0.1,0.2,0.3")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=10000)
    ap.add_argument("--verify-against", default=None,
                    help="CSV written by train_ppo_baseline; the 0.0 row must match it")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as fh:
        ck = pickle.load(fh)
    with open(args.match_checkpoint, "rb") as fh:
        ck_cfg = dict(pickle.load(fh)["config"])

    env_name = ck["env_name"]
    if ck_cfg["ENV_NAME"] != env_name:
        raise SystemExit(f"match-checkpoint env {ck_cfg['ENV_NAME']} != {env_name}")

    eval_cfg = dict(ck_cfg)
    eval_cfg["ENV_NAME"] = env_name
    eval_cfg["NORMALIZE_OBS"] = False     # PPO carries its own normalizer inside `params`
    eval_cfg["NORMALIZE_REWARD"] = False
    bundle = build_playground_env(eval_cfg)

    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    task_policy = ck_cfg.get("TASK_POLICY") or ck_cfg.get("POLICY") or env_name
    task_module = load_policy_module(task_policy)
    task_metrics_fn = ((lambda *a: task_module.task_metrics(*a))
                       if hasattr(task_module, "task_metrics") else (lambda *a: {}))

    make_inference_fn, params, ppo_cfg = rebuild_inference(ck, bundle)
    select = make_ppo_selector(make_inference_fn, params, ppo_cfg, action_low, action_high)

    rows = []
    for lvl in [float(x) for x in args.levels.split(",") if x.strip()]:
        summary = evaluate(
            eval_cfg, bundle.env, bundle.env_params, select, lambda o: o, task_metrics_fn,
            action_low, action_high, (action_high - action_low) / 2.0,
            args.num_envs, args.episodes,
            int(eval_cfg.get("EVAL_MAX_STEPS") or bundle.episode_length),
            action_noise=lvl, seed=args.eval_seed, action_dim=bundle.action_dim,
            decision_interval=1, num_skills=0,
        )
        summary.update({
            "env": env_name, "train_env": env_name, "meta": "ppo", "arm": "ppo",
            "seed": ck["seed"], "num_timesteps": ck["num_timesteps"],
            "perturbation": lvl, "mode": "action_noise",
        })
        rows.append(summary)
        print(f"[action_noise {lvl:>5}] success={summary.get('primary_success_rate', float('nan')):.4f} "
              f"return={summary['episode_return_mean']:.2f}", flush=True)

    if args.verify_against:
        want = None
        with open(args.verify_against, newline="") as fh:
            for r in csv.DictReader(fh):
                want = float(r["primary_success_rate"])
                break
        got = float(rows[0]["primary_success_rate"])
        if want is None:
            raise SystemExit(f"no row in {args.verify_against} to verify against")
        if abs(got - want) > args.tol:
            raise SystemExit(
                f"REBUILD MISMATCH: level 0.0 gives {got:.6f}, {args.verify_against} recorded "
                f"{want:.6f} (tol {args.tol}). Refusing to write -- a network rebuilt against "
                f"the wrong normalizer statistics would understate the baseline.")
        print(f"verify OK: level 0.0 {got:.6f} vs recorded {want:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
