"""Fallback implementation of 'interpreter.py' importing from
'nexus_continuous.policies.common'.

If the import fails (case when 'common.py' hasn't been filled yet), interpreter 
falls the equivalent implementation here instead of crashin so the LLM extension
stays independently runnable and testable. 

When 'policies.common' is available, it is always preferred against this module.
"""
from __future__ import annotations
from typing import Any, Iterable, Mapping
import jax.numpy as jnp 

def actor_obs(obs: Any) -> jnp.ndarray:
    """ Return the flat actor-observable array for 'obs'. """
    
    if isinstance(obs, Mapping):
        for key in ("state", "actor", "obs"):
            if key in obs:
                return jnp.asarray(obs[key])
        raise KeyError(
            "actor_obs: obs mapping has none of the expected keys"
            f"('state', 'actor', 'obs'); got keys = {list(obs.keys())}"
        )
    return jnp.asarray(obs)

def feature_info(obs: Any, info: Any | None) -> Mapping[str, Any]:
    """ Return a name -> value mapping of semantic/named features, if any."""
    del obs 
    if info is None: 
        return {}
    if isinstance(info, Mapping):
        return info
    for key in ("metrics", "state"):
        val = getattr(info, key, None)
        if isinstance(val, Mapping):
            return val 
    return {}

def info_value(semantic: Mapping[str, Any], names: Iterable[str], default: jnp.ndarray) -> jnp.ndarray:
    """ Return 'semantic[name]' for the first matching name, else 'default'. """
    
    for name in names:
        if name in semantic:
            return jnp.asarray(semantic[name])
    return default 

def safe_index(x: jnp.ndarray, idx: int, default: float = 0.0) -> jnp.ndarray:
    """ Index the last axis of 'x' at 'idx'.
        Fall back to 'default' if out of range.
    """
    x = jnp.asarray(x)
    n = x.shape[-1] if x.ndim > 0 else 0
    pos = idx if idx >= 0 else n + idx
    if n == 0 or pos < 0 or pos >= n:
        batch_shape = x.shape[:-1] if x.ndim > 0 else ()
        return jnp.full(batch_shape, default, dtype = jnp.float32)
    return x[..., idx]
        