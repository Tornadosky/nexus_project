"""PPO baseline arm (overnight runbook L4 / §5b).

Trains MuJoCo Playground's *shipped, tuned* brax PPO on an env and scores the result with the
**same** success metric every NEXUS cell is scored on, by reusing `robustness_eval.evaluate`
with a PPO-backed selector. Nothing about the metric is re-implemented here: the same
`task_metrics` module, the same deterministic rollout, the same `primary_success_rate`. That
reuse is the entire point — a baseline scored by a second, similar-looking implementation is
not a baseline, it is a second experiment.

What is and is not tuned
------------------------
* PPO hyperparameters come verbatim from `mujoco_playground.config.*.brax_ppo_config(env)`.
  None are hand-tuned (runbook §5b).
* `--num-timesteps` IS overridden, to the env-step budget of the corresponding V2 cells, read
  off a NEXUS checkpoint with `--match-checkpoint`. The runbook requires budget parity; the
  shipped PPO budget is far larger (200M for Go1, 60M for HopperHop), so the budget-matched
  number is the quotable one and the shipped-budget caveat is recorded in the CSV
  (`ppo_shipped_timesteps`) rather than left implicit.

`ppo` is in EXPERIMENTAL_TAGS in tools/analyze_v2.py: it is a different algorithm, reported in
its own table, never a cell of the V2 matrix.

Example
-------
    JAX_PLATFORMS=cuda ./.venv-wsl312/bin/python tools/train_ppo_baseline.py \
        --env-name Go1JoystickFlatTerrain --seed 0 \
        --match-checkpoint runs/verify/go1_joystick_flat_v2_s0.pkl \
        --out runs/ppo/go1_joystick_ppo_s0
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

from nexus_continuous.envs.playground_adapter import (
    build_playground_env,
    get_actor_obs,
    get_critic_obs,
)
from nexus_continuous.policies.registry import load_policy_module
from robustness_eval import evaluate


def _shim_device_put_replicated() -> None:
    """Restore `jax.device_put_replicated`, which brax 0.14.2's PPO still calls.

    jax 0.10.1 removed it as part of the pmap migration; brax's PPO trainer is otherwise
    pmap-clean under this version (`jax.pmap` itself still works). Downgrading jax is not an
    option — the NEXUS trainer runs on 0.10.1 and the baseline must run on the same stack it is
    being compared against, so the one missing symbol is reinstated here rather than in the
    environment. Only the single-device case is implemented: replicating across >1 device needs
    a real sharding and a wrong guess would silently train on a fraction of the intended batch.
    """
    if hasattr(jax, "device_put_replicated"):
        return

    def device_put_replicated(x, devices):  # noqa: ANN001 - matches the removed jax signature
        if len(devices) != 1:
            raise NotImplementedError(
                f"device_put_replicated shim supports 1 device, got {len(devices)}")
        return jax.device_put(
            jax.tree_util.tree_map(
                lambda y: jnp.broadcast_to(jnp.asarray(y), (1,) + jnp.shape(y)), x),
            devices[0])

    jax.device_put_replicated = device_put_replicated


def ppo_config_for(env_name: str):
    """The env's shipped brax PPO config, from whichever params module owns the env."""
    from mujoco_playground.config import (
        dm_control_suite_params,
        locomotion_params,
        manipulation_params,
    )

    last: Exception | None = None
    for mod in (locomotion_params, dm_control_suite_params, manipulation_params):
        try:
            return mod.brax_ppo_config(env_name)
        except Exception as exc:  # noqa: BLE001 - each module raises its own KeyError/ValueError
            last = exc
    raise SystemExit(f"no shipped brax PPO config for {env_name!r}: {last}")


# Plumbing-only settings. The shipped PPO configs run 2048-8192 envs, which is a GPU workload:
# validating this script on CPU with them takes longer than the GPU slot it is waiting for. A
# `--smoke` run shrinks the trainer to something a CPU can finish, and is labelled `ppo_smoke`
# in the CSV so its numbers can never be read as a baseline.
SMOKE_OVERRIDES = {
    "num_envs": 32,
    "batch_size": 32,
    "num_minibatches": 1,
    "unroll_length": 5,
    "num_evals": 1,
    "num_resets_per_eval": 0,
}


def train_ppo(env_name: str, seed: int, num_timesteps: int | None, progress_path: Path,
              smoke: bool = False):
    """Run brax PPO on the Playground env. Returns (make_inference_fn, params, ppo_cfg)."""
    _shim_device_put_replicated()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper

    env_cfg = registry.get_default_config(env_name)
    # The registry default impl is Warp, which crashes on this box in `warp_type_to_np_dtype`
    # (runbook pre-flight). Every NEXUS cell is pinned to jax; the baseline must be too, or it
    # is not the same environment implementation.
    if "impl" in env_cfg:
        env_cfg.impl = "jax"
    env = registry.load(env_name, config=env_cfg)
    eval_env = registry.load(env_name, config=env_cfg)

    ppo_cfg = ppo_config_for(env_name)
    shipped_timesteps = int(ppo_cfg["num_timesteps"])
    kwargs = dict(ppo_cfg)
    net_kwargs = dict(kwargs.pop("network_factory", {}) or {})
    if num_timesteps is not None:
        kwargs["num_timesteps"] = int(num_timesteps)
    if smoke:
        kwargs.update(SMOKE_OVERRIDES)

    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

    progress: list[dict[str, Any]] = []
    t0 = time.time()

    def progress_fn(step: int, metrics: dict[str, Any]) -> None:
        row = {"step": int(step), "wallclock_s": round(time.time() - t0, 1)}
        row.update({k: float(v) for k, v in metrics.items() if jnp.ndim(v) == 0})
        progress.append(row)
        progress_path.write_text(json.dumps(progress, indent=1))
        print(f"[ppo {env_name} step={step:>12,}] "
              f"reward={row.get('eval/episode_reward', float('nan')):.3f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        seed=seed,
        progress_fn=progress_fn,
        **kwargs,
    )
    return make_inference_fn, params, {**ppo_cfg, "shipped_num_timesteps": shipped_timesteps}


def make_ppo_selector(make_inference_fn, params, ppo_cfg, action_low, action_high):
    """Wrap brax's deterministic policy in `robustness_eval`'s `(obs, hold) -> (a, skill, hold)`.

    The observation handed to PPO must have the SAME structure it trained on — Playground's
    locomotion envs use a dict with an asymmetric `state`/`privileged_state` split, and brax's
    observation normalizer was fit on that structure. Feeding it a bare array would silently
    normalize against the wrong statistics and produce a baseline that looks worse than PPO is.
    """
    net_kwargs = dict(ppo_cfg.get("network_factory", {}) or {})
    policy_key = net_kwargs.get("policy_obs_key", "")
    value_key = net_kwargs.get("value_obs_key", "")

    # Whether to wrap the observation in a dict is decided by the NORMALIZER, not by the config.
    # `policy_obs_key` is present in the shipped PPO config for envs whose observation is a bare
    # array too -- PandaPickCube ships `policy_obs_key='state'` and a flat observation -- and
    # keying off the config alone wrapped that array into {'state': ...}, after which brax's
    # `normalizer_select` did `processor_params.mean['state']` on an ARRAY mean and died with
    # "JAX does not support string indexing; got idx='state'". It killed panda_pick_cube_ppo_s0
    # in evaluation on 2026-08-21, after training had finished and the checkpoint was written.
    # `params[0]` is brax's running_statistics state; its `mean` is a Mapping exactly when the
    # network was built over a dict observation, which is the real question being asked here.
    norm_mean = getattr(params[0], "mean", None)
    dict_obs = hasattr(norm_mean, "items")
    if policy_key and not dict_obs:
        policy_key = value_key = ""

    inference = make_inference_fn(params, deterministic=True)
    key = jax.random.PRNGKey(0)

    def select(obs, hold=None):
        actor_obs = get_actor_obs(obs)
        if policy_key:
            ppo_obs = {policy_key: actor_obs}
            if value_key and value_key != policy_key:
                ppo_obs[value_key] = get_critic_obs(obs)
        else:
            ppo_obs = actor_obs
        action, _ = inference(ppo_obs, key)
        skill = jnp.zeros((actor_obs.shape[0],), dtype=jnp.int32)
        return jnp.clip(action, action_low, action_high), skill, hold

    return select


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-name", required=True, help="Playground registry env name.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--match-checkpoint", default=None,
                   help="NEXUS checkpoint whose TOTAL_TIMESTEPS (and TASK_POLICY) this run "
                        "matches. Required for a quotable comparison.")
    p.add_argument("--num-timesteps", type=int, default=None,
                   help="Explicit env-step budget; overrides --match-checkpoint's.")
    p.add_argument("--task-policy", default=None,
                   help="Policy module supplying task_metrics (default: from the checkpoint, "
                        "else the env name).")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--eval-seed", type=int, default=10000)
    p.add_argument("--out", required=True, help="Output stem: writes <stem>.pkl and <stem>.csv.")
    p.add_argument("--smoke", action="store_true",
                   help="Plumbing check only: shrink the trainer so it runs on CPU and label "
                        "the result `ppo_smoke`. Never a baseline number.")
    args = p.parse_args(argv)

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if stem.with_suffix(".csv").exists():
        print(f"HAVE {stem.with_suffix('.csv')}")
        return

    budget = args.num_timesteps
    task_policy = args.task_policy
    ck_cfg: dict[str, Any] = {}
    if args.match_checkpoint:
        with open(args.match_checkpoint, "rb") as fh:
            ck_cfg = dict(pickle.load(fh)["config"])
        if budget is None:
            budget = int(ck_cfg["TOTAL_TIMESTEPS"])
        task_policy = task_policy or ck_cfg.get("TASK_POLICY") or ck_cfg.get("POLICY")
        if ck_cfg["ENV_NAME"] != args.env_name:
            raise SystemExit(f"checkpoint env {ck_cfg['ENV_NAME']} != --env-name {args.env_name}; "
                             "a budget must be matched against the same env")
    if budget is None:
        raise SystemExit("pass --match-checkpoint or --num-timesteps: an unmatched budget is "
                         "not comparable to the V2 cells")
    task_policy = task_policy or args.env_name

    print(f"PPO baseline: {args.env_name} seed={args.seed} budget={budget:,} env steps")
    make_inference_fn, params, ppo_cfg = train_ppo(
        args.env_name, args.seed, budget, stem.with_suffix(".progress.json"), smoke=args.smoke)
    with open(stem.with_suffix(".pkl"), "wb") as fh:
        pickle.dump({"params": params, "ppo_config": dict(ppo_cfg), "env_name": args.env_name,
                     "seed": args.seed, "num_timesteps": budget}, fh)
    print("wrote", stem.with_suffix(".pkl"))

    # ---- score it exactly like a NEXUS cell -------------------------------------------- #
    eval_cfg = dict(ck_cfg) if ck_cfg else {"ENV_NAME": args.env_name}
    eval_cfg["ENV_NAME"] = args.env_name
    eval_cfg["NORMALIZE_OBS"] = False      # PPO carries its own normalizer in `params`
    eval_cfg["NORMALIZE_REWARD"] = False
    bundle = build_playground_env(eval_cfg)
    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    task_module = load_policy_module(task_policy)
    task_metrics_fn = ((lambda *a: task_module.task_metrics(*a))
                       if hasattr(task_module, "task_metrics") else (lambda *a: {}))

    select = make_ppo_selector(make_inference_fn, params, ppo_cfg, action_low, action_high)
    summary = evaluate(
        eval_cfg, bundle.env, bundle.env_params, select, lambda o: o, task_metrics_fn,
        action_low, action_high, (action_high - action_low) / 2.0,
        args.num_envs, args.episodes,
        int(eval_cfg.get("EVAL_MAX_STEPS") or bundle.episode_length),
        action_noise=0.0, seed=args.eval_seed, action_dim=bundle.action_dim,
        decision_interval=1, num_skills=0,
    )
    summary.update({
        "env": args.env_name, "train_env": args.env_name,
        "meta": "ppo_smoke" if args.smoke else "ppo",
        "arm": "ppo_smoke" if args.smoke else "ppo",
        "seed": args.seed, "num_timesteps": budget,
        "ppo_shipped_timesteps": ppo_cfg["shipped_num_timesteps"],
        "perturbation": 0.0, "mode": "in_distribution",
    })
    out_csv = stem.with_suffix(".csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(summary))
        w.writeheader()
        w.writerow(summary)
    print(f"success={summary.get('primary_success_rate', float('nan')):.4f} "
          f"return={summary['episode_return_mean']:.2f}")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
