"""Flat AC-PQN baseline policy.

This module intentionally exposes one skill only. Its reward is exactly the
environment reward, and its mask is always available, so it shares the NEXUS
continuous-control wrapper/logging stack without using symbolic rules.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

SKILL_NAMES = ("flat_actor",)
NUM_SKILLS = len(SKILL_NAMES)


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, obs, action, done, info
    return env_reward[..., None]


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    del info
    if isinstance(obs, dict):
        obs = obs.get("actor", obs.get("raw_actor"))
    return jnp.zeros(jnp.asarray(obs).shape[:-1], dtype=jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    del info
    if isinstance(obs, dict):
        obs = obs.get("actor", obs.get("raw_actor"))
    return jnp.ones(jnp.asarray(obs).shape[:-1] + (NUM_SKILLS,), dtype=bool)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, obs, action, done, info
    return {"flat/env_reward": env_reward}


def task_metrics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, obs, action, done, info
    return {
        "primary_goal_metric": env_reward,
        "primary_success_rate": (env_reward > 0.0).astype(jnp.float32),
    }


def explain_policy() -> str:
    return "Flat baseline: one actor/critic trained directly on environment reward."
