"""Post-hoc Weights & Biases logging for continuous-control NEXUS.

Why post-hoc: ``run_training`` is ``jax.jit`` + ``vmap``-over-seeds with a
``lax.scan`` update loop, so ``wandb.log`` cannot be called inside the loop. The
training output instead carries a stacked metrics pytree (leading update axis, and
a seed axis when ``NUM_SEEDS > 1``). After training finishes we replay that history
into one W&B run per seed.

The per-update reduction here is a deliberate mirror of
``tools/collect_nexus_results.py`` (``_to_numpy`` / ``_series_from_metric``) so the
live dashboards and the authoritative offline CSVs agree leaf-for-leaf.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # jax is present in real runs; keep import-safe for offline tests.
    import jax
except Exception:  # pragma: no cover - exercised only without jax installed
    jax = None


# Modes that mean "do not track at all". Note: "offline" is NOT here -- offline is
# enabled tracking that writes locally for a later ``wandb sync``.
_DISABLED_VALUES = {"disabled", "off", "false", "0", "none", "no"}


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a JAX/NumPy/Python metric leaf to a float NumPy array."""
    if jax is not None:
        try:
            value = jax.device_get(value)
        except Exception:
            pass
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = arr.astype(float)
    return arr


def _series_from_metric(
    arr: np.ndarray, seed_index: int | None = None, num_seeds: int = 1
) -> np.ndarray:
    """Reduce a metric array to a 1-D per-update series.

    Mirrors ``tools/collect_nexus_results.py``. Handles shapes::

        [updates]
        [updates, epochs, minibatches]
        [seeds, updates]
        [seeds, updates, epochs, minibatches]
    """
    if arr.ndim == 0:
        return np.asarray([float(arr)])
    arr = arr.astype(float, copy=False)
    if seed_index is not None and num_seeds > 1 and arr.ndim >= 2 and arr.shape[0] == num_seeds:
        arr = arr[seed_index]
    if arr.ndim == 1:
        return arr
    return np.nanmean(arr, axis=tuple(range(1, arr.ndim)))


def _summary_scalars(
    metrics: Any, seed_index: int | None = None, num_seeds: int = 1
) -> dict[str, float]:
    """Reduce an eval/summary metric dict to per-seed scalars."""
    out: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        try:
            arr = _to_numpy(value)
        except Exception:
            continue
        if (
            seed_index is not None
            and num_seeds > 1
            and arr.ndim >= 1
            and arr.shape[0] == num_seeds
        ):
            arr = arr[seed_index]
        try:
            scalar = float(np.nanmean(arr))
        except Exception:
            continue
        if np.isfinite(scalar):
            out[key] = scalar
    return out


def resolve_settings(
    config: dict[str, Any], cli_disable: bool = False
) -> tuple[bool, dict[str, Any]]:
    """Decide whether to track and assemble ``wandb.init`` kwargs.

    Precedence for disabling: ``cli_disable`` (``--no-wandb``) > ``WANDB.enabled:
    false`` in config > ``WANDB.mode`` / ``$WANDB_MODE`` in the disabled set. An
    ``offline`` mode stays *enabled* (local logging for later ``wandb sync``).
    """
    wandb_cfg = dict(config.get("WANDB", {}) or {})
    env_mode = os.environ.get("WANDB_MODE")
    raw_mode = str(wandb_cfg.get("mode") or env_mode or "online").lower()

    enabled_flag = str(wandb_cfg.get("enabled", True)).lower() not in _DISABLED_VALUES
    enabled = (not cli_disable) and enabled_flag and raw_mode not in _DISABLED_VALUES

    mode = "offline" if raw_mode == "offline" else "online"
    env_name = str(config.get("ENV_NAME", "unknown"))
    meta_type = str(config.get("META_POLICY_TYPE", "unknown"))
    alg_name = str(config.get("ALG_NAME", "nexus"))

    default_tags = [env_name, meta_type, alg_name]
    settings = {
        "project": wandb_cfg.get("project")
        or os.environ.get("WANDB_PROJECT")
        or "nexus-continuous-control",
        "entity": wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY"),
        "mode": mode,
        "group": wandb_cfg.get("group") or f"{env_name}_{meta_type}",
        "tags": list(wandb_cfg.get("tags") or default_tags),
        "job_type": wandb_cfg.get("job_type", "train"),
    }
    return enabled, settings


# Metric-name prefixes that are pure logging noise (flat/constant debug counters).
# Dropping them at the source keeps the live workspace and reports uncluttered.
_DENYLIST_PREFIXES = ("debug/",)


def _is_denied(key: str) -> bool:
    return any(key.startswith(p) for p in _DENYLIST_PREFIXES)


# Canonical scalars surfaced at the top level of each run summary for easy tables.
_HEADLINE_EVAL_KEYS = (
    "primary_success_rate",
    "primary_goal_metric",
    "episode_return_mean",
    "episode_length_mean",
    "mask/violation_rate",
)


def replay_history_to_run(
    run: Any,
    config: dict[str, Any],
    output: Any,
    *,
    seed_index: int | None = None,
    seed_value: int = 0,
) -> None:
    """Replay one seed's stacked metrics history into an already-open W&B run.

    Shared by ``log_training_run`` (which creates the runs) and the sweep agent
    (which logs into the run the W&B agent already started).
    """
    metrics = getattr(output, "metrics", None) or {}
    eval_metrics = getattr(output, "eval_metrics", None) or {}
    num_seeds = int(config.get("NUM_SEEDS", 1) or 1)
    total_timesteps = config.get("TOTAL_TIMESTEPS")

    series_by_metric: dict[str, np.ndarray] = {}
    max_len = 0
    for key, value in metrics.items():
        if _is_denied(key):
            continue
        try:
            series = _series_from_metric(_to_numpy(value), seed_index, num_seeds)
        except Exception as exc:
            print(f"[wandb] skipping metric {key!r}: {exc!r}")
            continue
        series_by_metric[key] = series
        max_len = max(max_len, len(series))

    env_steps = series_by_metric.get("env_step")
    if env_steps is None or len(env_steps) == 0:
        if total_timesteps and max_len > 0:
            env_steps = np.linspace(
                float(total_timesteps) / max_len, float(total_timesteps), max_len
            )
        else:
            env_steps = np.arange(max_len)

    for i in range(max_len):
        row: dict[str, float] = {}
        for key, series in series_by_metric.items():
            if key == "env_step":
                continue
            if i < len(series) and np.isfinite(series[i]):
                row[key] = float(series[i])
        row["env_step"] = float(env_steps[i]) if i < len(env_steps) else float(i)
        run.log(row, step=i)

    eval_scalars = _summary_scalars(eval_metrics, seed_index, num_seeds)
    for key, value in eval_scalars.items():
        run.summary[f"eval/{key}"] = value
    for key in _HEADLINE_EVAL_KEYS:
        if key in eval_scalars:
            run.summary[key] = eval_scalars[key]


def log_training_run(
    config: dict[str, Any],
    output: Any,
    *,
    commit_hash: str = "unknown",
    cli_disable: bool = False,
    extra_config: dict[str, Any] | None = None,
) -> list[str]:
    """Replay a finished ``run_training`` output into W&B, one run per seed.

    Returns the list of created run ids (empty when tracking is disabled). Any W&B
    failure is swallowed with a warning: live tracking must never break a training
    job, because the offline pipeline is what gates depend on.
    """
    enabled, settings = resolve_settings(config, cli_disable=cli_disable)
    if not enabled:
        return []

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on install
        print(
            f"[wandb] tracking requested but wandb is not importable ({exc!r}); "
            "skipping live logging. Install with `pip install -e .` (wandb is a "
            "declared dependency) or pass --no-wandb."
        )
        return []

    num_seeds = int(config.get("NUM_SEEDS", 1) or 1)
    base_seed = int(config.get("SEED", 0))

    run_ids: list[str] = []
    for seed in range(num_seeds):
        seed_idx = seed if num_seeds > 1 else None
        seed_value = seed if num_seeds > 1 else base_seed

        run_config = dict(config)
        run_config.update({"commit_hash": commit_hash, "seed": seed_value})
        if extra_config:
            run_config.update(extra_config)

        try:
            run = wandb.init(
                project=settings["project"],
                entity=settings["entity"],
                mode=settings["mode"],
                group=settings["group"],
                tags=settings["tags"],
                job_type=settings["job_type"],
                name=f"{settings['group']}_seed{seed_value}",
                config=run_config,
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover - network/auth dependent
            print(f"[wandb] init failed ({exc!r}); skipping live logging for seed {seed_value}.")
            continue

        try:
            replay_history_to_run(
                run, config, output, seed_index=seed_idx, seed_value=seed_value
            )
            run_ids.append(run.id)
        finally:
            run.finish()

    return run_ids
