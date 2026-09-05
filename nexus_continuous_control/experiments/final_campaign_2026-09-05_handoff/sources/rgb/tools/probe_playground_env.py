#!/usr/bin/env python3
"""Probe a MuJoCo Playground environment reset/step for raw feature ranges."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from nexus_continuous.envs.playground_adapter import build_playground_env, get_policy_obs
from nexus_continuous.policies.registry import load_policy_module
from nexus_continuous.utils import load_config


def _summary(name: str, value) -> None:
    arr = np.asarray(jax.device_get(value), dtype=float)
    print(
        f"{name}: shape={arr.shape} finite={np.isfinite(arr).all()} "
        f"mean={np.nanmean(arr):.6g} min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config, [f"NUM_ENVS={args.num_envs}", "NORMALIZE_OBS=false"])
    bundle = build_playground_env(cfg)
    policy = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    rng = jax.random.PRNGKey(int(cfg.get("SEED", 0)))
    reset_rng = jax.random.split(rng, args.num_envs)
    obs, state = bundle.env.reset(reset_rng, bundle.env_params)
    action = jnp.zeros((args.num_envs, bundle.action_dim))

    print(f"env={cfg['ENV_NAME']} action_dim={bundle.action_dim}")
    _summary("reset/actor", obs["actor"])
    for step in range(args.steps):
        rng, step_rng = jax.random.split(rng)
        obs_next, state, reward, done, info = bundle.env.step(
            jax.random.split(step_rng, args.num_envs),
            state,
            action,
            bundle.env_params,
        )
        policy_obs = get_policy_obs(obs_next)
        rewards = policy.skill_rewards(get_policy_obs(obs), policy_obs, action, reward, done, info)
        mask = policy.skill_mask(policy_obs)
        diagnostics = policy.diagnostics(get_policy_obs(obs), policy_obs, action, reward, done, info)
        print(f"step={step}")
        _summary("reward", reward)
        _summary("done", done.astype(jnp.float32))
        _summary("actor", obs_next["actor"])
        _summary("skill_rewards", rewards)
        _summary("mask", mask.astype(jnp.float32))
        for key, value in diagnostics.items():
            _summary(f"diag/{key}", value)
        obs = obs_next
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
