"""JAX-compatible helpers for hand-written NEXUS policies.

The policy modules work with both dictionary observations from PureJAXQL's
Playground wrapper and raw arrays. They deliberately use forgiving fallbacks:
if a semantic key is unavailable in `info`, the functions fall back to stable
observation indices. That keeps the code runnable across Playground releases
while preserving a clear place to tighten the feature mapping for a specific env.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp


def actor_obs(obs: Any) -> jnp.ndarray:
    if isinstance(obs, Mapping):
        for key in ("raw_actor", "state", "actor", "obs", "observation", "raw_critic", "critic"):
            if key in obs:
                return jnp.asarray(obs[key])
    return jnp.asarray(obs)


def feature_info(obs: Any, info: Any | None = None) -> Any:
    """Return compact semantic policy features carried in obs/info, if present."""

    obs_info = None
    if isinstance(obs, Mapping):
        maybe_info = obs.get("policy_info")
        if isinstance(maybe_info, Mapping):
            obs_info = maybe_info
    if isinstance(info, Mapping):
        if obs_info is None:
            return info
        merged = dict(obs_info)
        merged.update(info)
        return merged
    return obs_info if obs_info is not None else info


def safe_index(x: jnp.ndarray, idx: int, default: float = 0.0) -> jnp.ndarray:
    """Read a feature index if available; otherwise return a broadcast default."""

    if idx < 0:
        idx = x.shape[-1] + idx
    if 0 <= idx < x.shape[-1]:
        return x[..., idx]
    return jnp.zeros(x.shape[:-1], dtype=x.dtype) + default


def safe_slice(x: jnp.ndarray, start: int, stop: int, default: float = 0.0) -> jnp.ndarray:
    width = stop - start
    if start >= 0 and stop <= x.shape[-1]:
        return x[..., start:stop]
    return jnp.zeros(x.shape[:-1] + (width,), dtype=x.dtype) + default


def info_value(info: Any, keys: tuple[str, ...], default: jnp.ndarray) -> jnp.ndarray:
    if isinstance(info, Mapping):
        for key in keys:
            if key in info:
                value = jnp.asarray(info[key])
                # Squeeze only TRAILING singleton dims, never the leading batch axis.
                #
                # This used to be a bare `jnp.squeeze`, which is correct for every batch size
                # except one. At batch=1 it deleted the batch axis too: panda's `tcp_pos`
                # [1, 3] became [3], `_vec_info` then "repaired" it along the wrong axis to
                # [3, 1], and `skill_mask` came out [3, 4] for a single environment — which
                # surfaced far downstream as `vmap got inconsistent sizes ... action
                # float32[3, 8]` when rendering a PandaPickCube rollout.
                #
                # Training never hit it (1024+ envs, so the batch axis is not a singleton);
                # only single-env paths — rendering and per-episode eval — were affected.
                # The original intent, stripping trailing singletons off scalar metrics
                # stored as [B, 1], is preserved exactly.
                while value.ndim > 1 and value.shape[-1] == 1:
                    value = value[..., 0]
                return value
    return default


def l2_norm(x: jnp.ndarray, axis: int = -1, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps)


def action_cost(action: jnp.ndarray, coeff: float = 1e-3) -> jnp.ndarray:
    return coeff * jnp.sum(jnp.square(action), axis=-1)


def improvement(prev_value: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Positive when value increased from previous state."""

    return value - prev_value


def decrease(prev_value: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Positive when value decreased from previous state."""

    return prev_value - value


def ensure_mask_has_skill(mask: jnp.ndarray, default_skill: int = 0) -> jnp.ndarray:
    any_active = jnp.any(mask, axis=-1, keepdims=True)
    default = jnp.zeros_like(mask)
    default = default.at[..., default_skill].set(True)
    return jnp.where(any_active, mask, default)


def categorical_from_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """Logits for uniform sampling over a boolean mask."""

    mask = ensure_mask_has_skill(mask)
    return jnp.where(mask, 0.0, -1.0e9)


def select_by_priority(conditions: list[jnp.ndarray], default: int) -> jnp.ndarray:
    """Return first active condition's index using vectorized jnp.where.

    Args:
      conditions: list of boolean arrays with shape [batch].
      default: default skill index.
    """

    out = jnp.zeros_like(conditions[0], dtype=jnp.int32) + default
    for idx in reversed(range(len(conditions))):
        out = jnp.where(conditions[idx], idx, out)
    return out
