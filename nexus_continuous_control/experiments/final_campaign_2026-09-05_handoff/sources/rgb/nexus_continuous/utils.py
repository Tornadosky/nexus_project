"""Small utilities shared by scripts and algorithms."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml


def tree_take(tree: Any, indices: jnp.ndarray, axis: int = 0) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.take(x, indices, axis=axis), tree)


def flatten_time_env(tree: Any) -> Any:
    """Flatten [T, E, ...] arrays to [T*E, ...] throughout a pytree."""

    def _reshape(x):
        return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

    return jax.tree_util.tree_map(_reshape, tree)


def make_minibatches(batch: Any, rng: jax.Array, num_minibatches: int) -> Any:
    leaves = jax.tree_util.tree_leaves(batch)
    batch_size = leaves[0].shape[0]
    permutation = jax.random.permutation(rng, batch_size)
    shuffled = tree_take(batch, permutation, axis=0)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_minibatches, batch_size // num_minibatches) + x.shape[1:]),
        shuffled,
    )


def load_config(path: str | os.PathLike[str], overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = yaml.safe_load(value)
    return cfg


def save_pickle_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = jax.device_get(dict(payload))
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    meta_path = path.with_suffix(path.suffix + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(path.name)}, f, indent=2)


def as_float(x: Any) -> float:
    return float(np.asarray(jax.device_get(x)))


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"Cannot convert {type(obj)} to dict")


def global_norm(tree: Any) -> jnp.ndarray:
    leaves = [jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.stack(leaves))) if leaves else jnp.asarray(0.0)
