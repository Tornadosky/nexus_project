#!/usr/bin/env python3
"""Collect NEXUS continuous-control run outputs into a review bundle.

The training script saves pickle payloads with:
  - config
  - runner_state
  - metrics

This collector extracts the metrics into CSVs, creates diagnostic plots, copies logs/config/env-info,
and writes a zip that can be uploaded for review.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import traceback
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - user dependency problem
    raise SystemExit(
        "collect_nexus_results.py requires pandas. Install with: pip install -e .[analysis]"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - user dependency problem
    raise SystemExit(
        "collect_nexus_results.py requires matplotlib. Install with: pip install -e .[analysis]"
    ) from exc


@dataclass
class RunRecord:
    run_id: str
    path: Path
    rel_path: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    eval_metrics: dict[str, Any]
    eval_episode_table: dict[str, Any]
    stage: str
    seed_from_name: int | None


def _safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "unnamed"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload, got {type(payload)}")
    return payload


def _to_numpy(value: Any) -> np.ndarray:
    """Convert JAX/NumPy/Python values to a NumPy array."""
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value)
    # Object arrays often come from unsupported leaves. Best-effort conversion below.
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception:
            raise ValueError(f"Cannot convert object array to float: shape={arr.shape}")
    return arr


def _infer_stage(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for name in (
        "patch_smoke",
        "finalization_one_seed",
        "final_research_matrix",
        "final_go1_matrix",
        "smoke",
        "one_seed",
        "main",
        "extension",
    ):
        if name in parts or any(name in part for part in parts):
            return name
    return "unknown"


def _infer_seed(path: Path, config: dict[str, Any]) -> int | None:
    m = re.search(r"seed(\d+)", path.stem)
    if m:
        return int(m.group(1))
    if "SEED" in config:
        try:
            return int(config["SEED"])
        except Exception:
            return None
    return None


def _run_id_from_path(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_smoke_seed\d+$", "", stem)
    stem = re.sub(r"_seed\d+$", "", stem)
    return stem


def discover_runs(runs_root: Path) -> tuple[list[RunRecord], list[dict[str, str]]]:
    records: list[RunRecord] = []
    failures: list[dict[str, str]] = []
    for path in sorted(runs_root.rglob("*.pkl")):
        try:
            payload = _load_pickle(path)
            config = payload.get("config", {}) or {}
            metrics = payload.get("metrics", {}) or {}
            eval_metrics = payload.get("eval_metrics", {}) or {}
            eval_episode_table = payload.get("eval_episode_table", {}) or {}
            if not isinstance(metrics, dict):
                raise ValueError(f"metrics is not a dict: {type(metrics)}")
            if not isinstance(eval_metrics, dict):
                raise ValueError(f"eval_metrics is not a dict: {type(eval_metrics)}")
            if not isinstance(eval_episode_table, dict):
                raise ValueError(
                    f"eval_episode_table is not a dict: {type(eval_episode_table)}"
                )
            rec = RunRecord(
                run_id=_run_id_from_path(path),
                path=path,
                rel_path=str(path.relative_to(runs_root)),
                config=dict(config),
                metrics=metrics,
                eval_metrics=eval_metrics,
                eval_episode_table=eval_episode_table,
                stage=_infer_stage(path),
                seed_from_name=_infer_seed(path, dict(config)),
            )
            records.append(rec)
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return records, failures


def _series_from_metric(arr: np.ndarray, seed_index: int | None = None, num_seeds: int = 1) -> np.ndarray:
    """Return a 1-D update series from a metric array.

    Expected shapes:
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


def metrics_to_dataframes(records: list[RunRecord]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    long_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for rec in records:
        cfg = rec.config
        num_seeds = int(cfg.get("NUM_SEEDS", 1) or 1)
        seed_values: Iterable[int]
        if num_seeds > 1:
            seed_values = range(num_seeds)
        else:
            seed_values = [int(rec.seed_from_name if rec.seed_from_name is not None else cfg.get("SEED", 0))]

        env_name = cfg.get("ENV_NAME", "unknown")
        policy = cfg.get("POLICY", env_name)
        meta_type = cfg.get("META_POLICY_TYPE", "unknown")
        total_timesteps = cfg.get("TOTAL_TIMESTEPS", None)
        num_envs = cfg.get("NUM_ENVS", None)
        num_steps = cfg.get("NUM_STEPS", None)

        metric_arrays: dict[str, np.ndarray] = {}
        for metric, value in rec.metrics.items():
            try:
                metric_arrays[metric] = _to_numpy(value)
            except Exception as exc:
                errors.append({"run_id": rec.run_id, "metric": metric, "error": repr(exc)})

        for seed in seed_values:
            seed_idx = seed if num_seeds > 1 else None
            series_by_metric: dict[str, np.ndarray] = {}
            max_len = 0
            for metric, arr in metric_arrays.items():
                try:
                    series = _series_from_metric(arr, seed_idx, num_seeds)
                    series_by_metric[metric] = series
                    max_len = max(max_len, len(series))
                except Exception as exc:
                    errors.append({"run_id": rec.run_id, "metric": metric, "error": repr(exc)})

            # Prefer real env_step if present.
            env_steps = series_by_metric.get("env_step")
            if env_steps is None or len(env_steps) == 0:
                if total_timesteps and max_len > 0:
                    env_steps = np.linspace(float(total_timesteps) / max_len, float(total_timesteps), max_len)
                else:
                    env_steps = np.arange(max_len)

            for metric, series in series_by_metric.items():
                if metric == "env_step":
                    continue
                for update_idx, value in enumerate(series):
                    if update_idx >= len(env_steps):
                        env_step = np.nan
                    else:
                        env_step = float(env_steps[update_idx])
                    long_rows.append(
                        {
                            "run_id": rec.run_id,
                            "stage": rec.stage,
                            "seed": int(seed),
                            "env_name": env_name,
                            "policy": policy,
                            "meta_policy_type": meta_type,
                            "total_timesteps": total_timesteps,
                            "num_envs": num_envs,
                            "num_steps": num_steps,
                            "source_pickle": rec.rel_path,
                            "update_idx": int(update_idx),
                            "env_step": env_step,
                            "metric": metric,
                            "value": float(value) if np.isfinite(value) else value,
                        }
                    )

        inventory_rows.append(
            {
                "run_id": rec.run_id,
                "stage": rec.stage,
                "seed_from_name": rec.seed_from_name,
                "env_name": env_name,
                "policy": policy,
                "meta_policy_type": meta_type,
                "total_timesteps": total_timesteps,
                "num_envs": num_envs,
                "num_steps": num_steps,
                "num_epochs": cfg.get("NUM_EPOCHS", None),
                "num_minibatches": cfg.get("NUM_MINIBATCHES", None),
                "save_path": cfg.get("SAVE_PATH", None),
                "source_pickle": rec.rel_path,
                "num_metrics": len(metric_arrays),
            }
        )

    long_df = pd.DataFrame(long_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    if long_df.empty:
        wide_df = pd.DataFrame()
    else:
        index_cols = [
            "run_id",
            "stage",
            "seed",
            "env_name",
            "policy",
            "meta_policy_type",
            "total_timesteps",
            "num_envs",
            "num_steps",
            "source_pickle",
            "update_idx",
            "env_step",
        ]
        wide_df = long_df.pivot_table(
            index=index_cols,
            columns="metric",
            values="value",
            aggfunc="mean",
        ).reset_index()
        wide_df.columns.name = None
    return long_df, wide_df, inventory_df, errors


def deterministic_eval_to_dataframes(
    records: list[RunRecord],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for rec in records:
        if not rec.eval_episode_table:
            continue
        cfg = rec.config
        env_name = cfg.get("ENV_NAME", "unknown")
        policy = cfg.get("POLICY", env_name)
        meta_type = cfg.get("META_POLICY_TYPE", "unknown")
        seed = int(rec.seed_from_name if rec.seed_from_name is not None else cfg.get("SEED", 0))

        arrays: dict[str, np.ndarray] = {}
        for metric, value in rec.eval_episode_table.items():
            try:
                arr = _to_numpy(value).reshape(-1)
                arrays[metric] = arr
            except Exception as exc:
                errors.append({"run_id": rec.run_id, "metric": metric, "error": repr(exc)})
        if not arrays:
            continue

        num_rows = max(len(arr) for arr in arrays.values())
        eval_seed = cfg.get("EVAL_SEED", rec.eval_metrics.get("eval_seed", None))
        try:
            eval_seed = int(np.asarray(_to_numpy(eval_seed)).reshape(-1)[0])
        except Exception:
            eval_seed = None

        for idx in range(num_rows):
            row = {
                "run_id": rec.run_id,
                "stage": rec.stage,
                "env_name": env_name,
                "policy": policy,
                "meta_policy_type": meta_type,
                "seed": seed,
                "eval_seed": eval_seed,
                "source_pickle": rec.rel_path,
                "eval_episode_index": idx,
            }
            for metric, arr in arrays.items():
                if idx < len(arr):
                    value = arr[idx]
                    row[metric] = float(value) if np.isfinite(value) else value
            episode_rows.append(row)

        summary: dict[str, Any] = {
            "run_id": rec.run_id,
            "stage": rec.stage,
            "env_name": env_name,
            "policy": policy,
            "meta_policy_type": meta_type,
            "seed": seed,
            "eval_seed": eval_seed,
            "source_pickle": rec.rel_path,
            "num_eval_episodes": int(num_rows),
        }
        returns = pd.to_numeric(pd.Series(arrays.get("episode_return", [])), errors="coerce")
        lengths = pd.to_numeric(pd.Series(arrays.get("episode_length", [])), errors="coerce")
        summary["episode_return_mean"] = float(returns.mean()) if len(returns) else math.nan
        summary["episode_return_std"] = float(returns.std(ddof=0)) if len(returns) else math.nan
        summary["episode_length_mean"] = float(lengths.mean()) if len(lengths) else math.nan
        for metric, arr in arrays.items():
            if metric in ("eval_episode_index", "episode_return", "episode_length"):
                continue
            vals = pd.to_numeric(pd.Series(arr), errors="coerce")
            summary[metric] = float(vals.mean()) if len(vals) else math.nan

        # Prefer payload summary values where present; they should match the episode table.
        for metric, value in rec.eval_metrics.items():
            if metric in {"eval_seed", "num_eval_episodes"}:
                continue
            try:
                arr = _to_numpy(value).reshape(-1)
                if arr.size:
                    summary[metric] = float(arr[0]) if np.isfinite(arr[0]) else arr[0]
            except Exception:
                pass
        summary_rows.append(summary)

    episodes_df = pd.DataFrame(episode_rows)
    summary_df = pd.DataFrame(summary_rows)
    task_success_df = summary_df.copy()
    return episodes_df, summary_df, task_success_df, errors


def _last_window_mean(df: pd.DataFrame, metric: str, frac: float = 0.1) -> float:
    if metric not in df.columns or df.empty:
        return math.nan
    n = max(1, int(math.ceil(len(df) * frac)))
    vals = pd.to_numeric(df[metric].tail(n), errors="coerce")
    return float(vals.mean()) if len(vals) else math.nan


def make_final_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()

    metric_cols = [c for c in wide_df.columns if c not in {
        "run_id", "stage", "seed", "env_name", "policy", "meta_policy_type", "total_timesteps",
        "num_envs", "num_steps", "source_pickle", "update_idx", "env_step"
    }]
    preferred = [
        "env/returned_episode_returns",
        "returns/env_reward_mean",
        "env/original_reward",
        "env/returned_episode_lengths",
        "episode/done_fraction",
        "loss/critic",
        "loss/actor",
        "loss/meta",
        "train/critic_abs_td",
        "train/meta_abs_td",
    ]
    skill_usage = sorted([c for c in metric_cols if c.startswith("skill_usage/")])
    skill_reward = sorted([c for c in metric_cols if c.startswith("skill_reward/")])
    mask_metrics = sorted(
        [
            c
            for c in metric_cols
            if c.startswith("mask_available/")
            or c.startswith("mask_selected_when_available/")
            or c.startswith("mask_selected_given_available/")
            or c.startswith("mask/violation_rate")
            or c.startswith("mask_violation/")
        ]
    )
    raw_diag_metrics = sorted([c for c in metric_cols if c.startswith("policy_diag/")])
    selected_metrics = (
        [m for m in preferred if m in metric_cols]
        + skill_usage
        + skill_reward
        + mask_metrics
        + raw_diag_metrics
    )

    rows = []
    group_cols = ["run_id", "stage", "seed", "env_name", "policy", "meta_policy_type", "source_pickle"]
    for keys, group in wide_df.sort_values("update_idx").groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["num_updates"] = int(group["update_idx"].nunique())
        base["final_env_step"] = float(pd.to_numeric(group["env_step"], errors="coerce").max())
        for metric in selected_metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            base[f"final/{metric}"] = float(vals.iloc[-1]) if len(vals) else math.nan
            base[f"last10pct_mean/{metric}"] = _last_window_mean(group, metric)
            base[f"first10pct_mean/{metric}"] = float(vals.head(max(1, int(math.ceil(len(vals) * 0.1)))).mean()) if len(vals) else math.nan
            if len(vals):
                base[f"finite/{metric}"] = bool(np.isfinite(vals.to_numpy(dtype=float)).all())
        rows.append(base)
    return pd.DataFrame(rows)


def _preferred_return_metric(columns: Iterable[str]) -> str | None:
    for candidate in (
        "env/returned_episode_returns",
        "returns/env_reward_mean",
        "env/original_reward",
    ):
        if f"last10pct_mean/{candidate}" in columns:
            return candidate
    return None


def make_learning_trends(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    metric = _preferred_return_metric(summary_df.columns)
    if metric is None:
        return pd.DataFrame()
    rows = []
    for _, row in summary_df.iterrows():
        first = pd.to_numeric(pd.Series([row[f"first10pct_mean/{metric}"]]), errors="coerce").iloc[0]
        last = pd.to_numeric(pd.Series([row[f"last10pct_mean/{metric}"]]), errors="coerce").iloc[0]
        delta = last - first if np.isfinite(first) and np.isfinite(last) else math.nan
        rows.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "env_name": row["env_name"],
                "meta_policy_type": row["meta_policy_type"],
                "metric": metric,
                "first10pct_mean": first,
                "last10pct_mean": last,
                "delta": delta,
                "positive_learning_trend": bool(np.isfinite(delta) and delta > 0.0),
            }
        )
    return pd.DataFrame(rows)


def make_baseline_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    metric = _preferred_return_metric(summary_df.columns)
    if metric is None:
        return pd.DataFrame()
    metric_col = f"last10pct_mean/{metric}"
    df = summary_df[["env_name", "meta_policy_type", "seed", metric_col]].copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    agg = df.groupby(["env_name", "meta_policy_type"], dropna=False)[metric_col].agg(
        ["mean", "std", "count"]
    )
    rows = []
    for env, env_df in agg.reset_index().groupby("env_name", dropna=False):
        flat = env_df.loc[env_df["meta_policy_type"] == "flat"]
        flat_mean = float(flat["mean"].iloc[0]) if not flat.empty else math.nan
        for item in env_df.itertuples(index=False):
            ratio = item.mean / flat_mean if np.isfinite(flat_mean) and flat_mean != 0.0 else math.nan
            rows.append(
                {
                    "env_name": env,
                    "meta_policy_type": item.meta_policy_type,
                    "metric": metric,
                    "final_mean": item.mean,
                    "final_std": item.std,
                    "num_seeds": item.count,
                    "flat_final_mean": flat_mean,
                    "ratio_to_flat": ratio,
                    "meets_neural_80pct": bool(
                        item.meta_policy_type == "neural" and np.isfinite(ratio) and ratio >= 0.8
                    ),
                    "meets_nesy_70pct": bool(
                        item.meta_policy_type == "nesy" and np.isfinite(ratio) and ratio >= 0.7
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_deterministic_baseline_comparison(eval_summary_df: pd.DataFrame) -> pd.DataFrame:
    if eval_summary_df.empty or "episode_return_mean" not in eval_summary_df.columns:
        return pd.DataFrame()
    df = eval_summary_df[
        ["env_name", "meta_policy_type", "seed", "episode_return_mean"]
    ].copy()
    df["episode_return_mean"] = pd.to_numeric(df["episode_return_mean"], errors="coerce")
    agg = df.groupby(["env_name", "meta_policy_type"], dropna=False)["episode_return_mean"].agg(
        ["mean", "std", "count"]
    )
    rows = []
    for env, env_df in agg.reset_index().groupby("env_name", dropna=False):
        flat = env_df.loc[env_df["meta_policy_type"] == "flat"]
        flat_mean = float(flat["mean"].iloc[0]) if not flat.empty else math.nan
        for item in env_df.itertuples(index=False):
            ratio = item.mean / flat_mean if np.isfinite(flat_mean) and flat_mean != 0.0 else math.nan
            rows.append(
                {
                    "env_name": env,
                    "meta_policy_type": item.meta_policy_type,
                    "metric": "deterministic_episode_return",
                    "final_mean": item.mean,
                    "final_std": item.std,
                    "num_seeds": item.count,
                    "flat_final_mean": flat_mean,
                    "ratio_to_flat": ratio,
                    "meets_neural_80pct": bool(
                        item.meta_policy_type == "neural" and np.isfinite(ratio) and ratio >= 0.8
                    ),
                    "meets_nesy_70pct": bool(
                        item.meta_policy_type == "nesy" and np.isfinite(ratio) and ratio >= 0.7
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_skill_disentanglement(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    usage_cols = [c for c in summary_df.columns if c.startswith("last10pct_mean/skill_usage/")]
    reward_cols = [c for c in summary_df.columns if c.startswith("last10pct_mean/skill_reward/")]
    rows = []
    for _, row in summary_df.iterrows():
        usage = pd.to_numeric(row[usage_cols], errors="coerce").to_numpy(dtype=float) if usage_cols else np.array([])
        rewards = (
            pd.to_numeric(row[reward_cols], errors="coerce").to_numpy(dtype=float)
            if reward_cols
            else np.array([])
        )
        usage = usage[np.isfinite(usage)]
        rewards = rewards[np.isfinite(rewards)]
        entropy = math.nan
        if usage.size:
            probs = usage / max(float(usage.sum()), 1e-12)
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        rows.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "env_name": row["env_name"],
                "meta_policy_type": row["meta_policy_type"],
                "num_usage_skills": int((usage > 0.01).sum()) if usage.size else 0,
                "usage_entropy": entropy,
                "skill_reward_std": float(np.std(rewards)) if rewards.size else math.nan,
                "skill_rewards_finite": bool(np.isfinite(rewards).all()) if rewards.size else False,
                "skill_rewards_nonconstant": bool(np.std(rewards) > 1e-6) if rewards.size else False,
            }
        )
    return pd.DataFrame(rows)


def make_mask_diagnostics(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    mask_cols = [
        c
        for c in summary_df.columns
        if c.startswith("last10pct_mean/mask_available/")
        or c.startswith("last10pct_mean/mask_selected_when_available/")
        or c.startswith("last10pct_mean/mask_selected_given_available/")
        or c.startswith("last10pct_mean/mask/violation_rate")
        or c.startswith("last10pct_mean/mask_violation/")
    ]
    rows = []
    for _, row in summary_df.iterrows():
        for col in mask_cols:
            metric = col.split("last10pct_mean/", 1)[1]
            if metric.startswith("mask/violation_rate"):
                kind = "mask_violation_rate"
                skill = "all"
            else:
                kind, skill = metric.split("/", 1)
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            rows.append(
                {
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "env_name": row["env_name"],
                    "meta_policy_type": row["meta_policy_type"],
                    "metric": metric,
                    "kind": kind,
                    "skill": skill,
                    "last10pct_mean": value,
                    "mask_violation_value": value if "violation" in metric else math.nan,
                }
            )
    return pd.DataFrame(rows)


def make_raw_feature_diagnostics(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()
    diag_cols = [c for c in wide_df.columns if c.startswith("policy_diag/")]
    rows = []
    group_cols = ["run_id", "seed", "env_name", "meta_policy_type"]
    for keys, group in wide_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for col in diag_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            rows.append(
                {
                    **base,
                    "feature": col.replace("policy_diag/", "", 1),
                    "mean": float(vals.mean()) if vals.notna().any() else math.nan,
                    "std": float(vals.std()) if vals.notna().any() else math.nan,
                    "min": float(vals.min()) if vals.notna().any() else math.nan,
                    "max": float(vals.max()) if vals.notna().any() else math.nan,
                    "finite": bool(np.isfinite(vals.dropna().to_numpy(dtype=float)).all())
                    if vals.notna().any()
                    else False,
                }
            )
    return pd.DataFrame(rows)


def _plot_metric_curves(df: pd.DataFrame, metric: str, out_path: Path, title: str, group_by: list[str]) -> None:
    if df.empty or metric not in df.columns:
        return
    plot_df = df[["env_step", metric] + group_by].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["env_step", metric])
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    for keys, group in plot_df.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = "/".join(str(k) for k in keys)
        group = group.sort_values("env_step")
        plt.plot(group["env_step"], group[metric], label=label, linewidth=1.5, alpha=0.9)
    plt.xlabel("environment steps")
    plt.ylabel(metric)
    plt.title(title)
    if plot_df[group_by].drop_duplicates().shape[0] <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_group_mean(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if df.empty or metric not in df.columns:
        return
    plot_df = df[["env_step", "env_name", "meta_policy_type", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["env_step", metric])
    if plot_df.empty:
        return

    # Bin by exact env_step after grouping; seeds should have matching steps for a run.
    agg = plot_df.groupby(["env_name", "meta_policy_type", "env_step"], dropna=False)[metric].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(10, 6))
    for (env, meta), group in agg.groupby(["env_name", "meta_policy_type"], dropna=False):
        group = group.sort_values("env_step")
        label = f"{env}/{meta}"
        plt.plot(group["env_step"], group["mean"], label=label, linewidth=1.8)
        if group["std"].notna().any():
            x = group["env_step"].to_numpy(dtype=float)
            mean = group["mean"].to_numpy(dtype=float)
            std = group["std"].fillna(0.0).to_numpy(dtype=float)
            plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("environment steps")
    plt.ylabel(metric)
    plt.title(title)
    if agg[["env_name", "meta_policy_type"]].drop_duplicates().shape[0] <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_final_bars(summary_df: pd.DataFrame, metric_col: str, out_path: Path, title: str) -> None:
    if summary_df.empty or metric_col not in summary_df.columns:
        return
    df = summary_df[["env_name", "meta_policy_type", "run_id", metric_col]].copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_col])
    if df.empty:
        return
    agg = df.groupby(["env_name", "meta_policy_type"], dropna=False)[metric_col].agg(["mean", "std"]).reset_index()
    labels = [f"{r.env_name}\n{r.meta_policy_type}" for r in agg.itertuples(index=False)]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 0.7), 6))
    plt.bar(x, agg["mean"].to_numpy(dtype=float), yerr=agg["std"].fillna(0).to_numpy(dtype=float), capsize=3)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_plots(wide_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> None:
    if wide_df.empty:
        return
    metric_cols = [c for c in wide_df.columns if c not in {
        "run_id", "stage", "seed", "env_name", "policy", "meta_policy_type", "total_timesteps",
        "num_envs", "num_steps", "source_pickle", "update_idx", "env_step"
    }]
    plot_specs = [
        "env/returned_episode_returns",
        "returns/env_reward_mean",
        "env/original_reward",
        "env/returned_episode_lengths",
        "episode/done_fraction",
        "noise",
        "meta_epsilon",
        "loss/critic",
        "loss/actor",
        "loss/meta",
        "train/critic_abs_td",
        "train/meta_abs_td",
        "train/actor_q",
        "train/meta_q",
    ]
    plot_specs = [m for m in plot_specs if m in metric_cols]

    # Per-run curves.
    for metric in plot_specs:
        _plot_metric_curves(
            wide_df,
            metric,
            out_dir / "by_run" / f"{_safe_name(metric)}.png",
            f"{metric} by run/seed",
            ["run_id", "seed"],
        )
        _plot_group_mean(
            wide_df,
            metric,
            out_dir / "aggregate" / f"{_safe_name(metric)}_mean_by_env_variant.png",
            f"{metric}: mean ± std by env/variant",
        )

    # Skill plots by run.
    for prefix in ("skill_usage/", "skill_reward/"):
        for env, env_df in wide_df.groupby("env_name", dropna=False):
            skill_metrics = sorted([m for m in metric_cols if m.startswith(prefix)])
            if not skill_metrics:
                continue
            env_skill_metrics = [m for m in skill_metrics if m in env_df.columns]
            for run_id, run_df in env_df.groupby("run_id", dropna=False):
                plt.figure(figsize=(10, 6))
                any_plotted = False
                for metric in env_skill_metrics:
                    vals = pd.to_numeric(run_df[metric], errors="coerce")
                    if vals.notna().any():
                        ordered = run_df.assign(_metric=vals).sort_values("env_step")
                        plt.plot(ordered["env_step"], ordered["_metric"], label=metric.split("/", 1)[1])
                        any_plotted = True
                if any_plotted:
                    plt.xlabel("environment steps")
                    plt.ylabel(prefix.rstrip("/"))
                    plt.title(f"{prefix.rstrip('/')} for {run_id}")
                    plt.legend(fontsize=8)
                    plt.tight_layout()
                    path = out_dir / "by_run" / f"{_safe_name(run_id)}_{_safe_name(prefix)}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(path, dpi=150)
                plt.close()

    # Final bars.
    candidate_final_metrics = [
        "last10pct_mean/env/returned_episode_returns",
        "last10pct_mean/returns/env_reward_mean",
        "last10pct_mean/env/original_reward",
        "last10pct_mean/env/returned_episode_lengths",
        "last10pct_mean/episode/done_fraction",
    ]
    for metric_col in candidate_final_metrics:
        if metric_col in summary_df.columns:
            _plot_final_bars(
                summary_df,
                metric_col,
                out_dir / "aggregate" / f"final_bar_{_safe_name(metric_col)}.png",
                f"Final-window {metric_col}",
            )

    # Per-env final skill usage bars.
    usage_cols = [c for c in summary_df.columns if c.startswith("last10pct_mean/skill_usage/")]
    for env, env_df in summary_df.groupby("env_name", dropna=False):
        if not usage_cols:
            continue
        means = []
        labels = []
        for col in usage_cols:
            val = pd.to_numeric(env_df[col], errors="coerce").mean()
            if np.isfinite(val):
                labels.append(col.split("skill_usage/", 1)[1])
                means.append(val)
        if not means:
            continue
        x = np.arange(len(labels))
        plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
        plt.bar(x, means)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("last10pct_mean skill usage")
        plt.title(f"Final skill usage: {env}")
        plt.tight_layout()
        path = out_dir / "by_env" / f"{_safe_name(env)}_final_skill_usage.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_supporting_files(runs_root: Path, out_dir: Path, records: list[RunRecord], include_checkpoints: bool) -> None:
    # Logs.
    logs_out = out_dir / "logs"
    logs_out.mkdir(parents=True, exist_ok=True)
    for ext in ("*.log", "*.txt"):
        for path in sorted((runs_root / "logs").glob(ext)):
            shutil.copy2(path, logs_out / path.name)

    # Environment info.
    env_info_candidates = [runs_root / "env_info", runs_root / "verification" / "env_info"]
    for src in env_info_candidates:
        if src.exists():
            _copy_tree_if_exists(src, out_dir / "env_info")
            break

    # Config snapshots from payload.
    config_out = out_dir / "configs"
    config_out.mkdir(parents=True, exist_ok=True)
    for rec in records:
        seed = rec.seed_from_name if rec.seed_from_name is not None else rec.config.get("SEED", "unknown")
        fname = f"{_safe_name(rec.stage)}_{_safe_name(rec.run_id)}_seed{seed}_config.json"
        with (config_out / fname).open("w", encoding="utf-8") as f:
            json.dump(rec.config, f, indent=2, default=_json_default)

    if include_checkpoints:
        ckpt_out = out_dir / "checkpoints"
        ckpt_out.mkdir(parents=True, exist_ok=True)
        for rec in records:
            dst = ckpt_out / _safe_name(rec.rel_path).replace("_pkl", ".pkl")
            shutil.copy2(rec.path, dst)
            json_path = rec.path.with_suffix(rec.path.suffix + ".json")
            if json_path.exists():
                shutil.copy2(json_path, ckpt_out / f"{dst.name}.json")


def write_diagnostics(
    out_dir: Path,
    records: list[RunRecord],
    load_failures: list[dict[str, str]],
    metric_errors: list[dict[str, str]],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# NEXUS result diagnostics")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Runs loaded: {len(records)}")
    lines.append(f"Pickle load failures: {len(load_failures)}")
    lines.append(f"Metric extraction errors: {len(metric_errors)}")
    lines.append("")

    if records:
        by_stage = defaultdict(int)
        by_env = defaultdict(int)
        by_variant = defaultdict(int)
        for r in records:
            by_stage[r.stage] += 1
            by_env[str(r.config.get("ENV_NAME", "unknown"))] += 1
            by_variant[str(r.config.get("META_POLICY_TYPE", "unknown"))] += 1
        lines.append("## Inventory")
        lines.append("")
        lines.append("By stage: " + json.dumps(dict(sorted(by_stage.items())), sort_keys=True))
        lines.append("By environment: " + json.dumps(dict(sorted(by_env.items())), sort_keys=True))
        lines.append("By variant: " + json.dumps(dict(sorted(by_variant.items())), sort_keys=True))
        lines.append("")

    if load_failures:
        lines.append("## Pickle load failures")
        lines.append("")
        for item in load_failures:
            lines.append(f"- `{item['path']}`: `{item['error']}`")
        lines.append("")

    if metric_errors:
        lines.append("## Metric extraction errors")
        lines.append("")
        for item in metric_errors[:100]:
            lines.append(f"- `{item.get('run_id')}` / `{item.get('metric')}`: `{item.get('error')}`")
        if len(metric_errors) > 100:
            lines.append(f"- ... {len(metric_errors) - 100} more")
        lines.append("")

    lines.append("## Finite-value checks")
    lines.append("")
    if long_df.empty:
        lines.append("- FATAL: no metrics were extracted.")
    else:
        values = pd.to_numeric(long_df["value"], errors="coerce").to_numpy(dtype=float)
        bad_rows = long_df.loc[~np.isfinite(values), ["run_id", "seed", "metric"]]
        if not bad_rows.empty:
            lines.append("- FATAL: non-finite metric values detected:")
            for row in bad_rows.drop_duplicates().head(100).itertuples(index=False):
                lines.append(f"  - {row.run_id} seed{row.seed} {row.metric}")
            if len(bad_rows) > 100:
                lines.append(f"  - ... {len(bad_rows) - 100} more")
        else:
            lines.append("- OK: no non-finite numeric metric values detected.")

    lines.append("")
    lines.append("## Skill usage checks")
    lines.append("")
    if not wide_df.empty:
        usage_cols = [c for c in wide_df.columns if c.startswith("skill_usage/")]
        if usage_cols:
            tmp = wide_df.copy()
            tmp["skill_usage_sum"] = tmp[usage_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            max_deviation = float((tmp["skill_usage_sum"] - 1.0).abs().max())
            lines.append(f"- Max |sum(skill_usage)-1|: {max_deviation:.6g}")
            if max_deviation > 0.05:
                lines.append("- WARNING: skill usage does not sum near 1.0 for at least one update.")
            if not summary_df.empty:
                low_usage_notes = []
                for col in [c for c in summary_df.columns if c.startswith("last10pct_mean/skill_usage/")]:
                    for row in summary_df.itertuples(index=False):
                        val = getattr(row, col.replace("/", "_"), None)
                    # Use iterrows because slash column names are awkward.
                    for _, row in summary_df.iterrows():
                        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                        if np.isfinite(val) and val < 0.01:
                            low_usage_notes.append((row["run_id"], row["seed"], col, val))
                if low_usage_notes:
                    lines.append("- WARNING: low final-window skill usage detected:")
                    for run_id, seed, col, val in low_usage_notes[:50]:
                        lines.append(f"  - {run_id} seed{seed} {col}: {val:.4g}")
                    if len(low_usage_notes) > 50:
                        lines.append(f"  - ... {len(low_usage_notes)-50} more")
        else:
            lines.append("- WARNING: no skill_usage metrics found.")

    lines.append("")
    lines.append("## Learning-signal checks")
    lines.append("")
    if not summary_df.empty:
        reward_metric = None
        for candidate in ("returns/env_reward_mean", "env/returned_episode_returns", "env/original_reward"):
            if f"first10pct_mean/{candidate}" in summary_df.columns and f"last10pct_mean/{candidate}" in summary_df.columns:
                reward_metric = candidate
                break
        if reward_metric is None:
            lines.append("- WARNING: no standard reward/return metric found.")
        else:
            for _, row in summary_df.iterrows():
                first = pd.to_numeric(pd.Series([row[f"first10pct_mean/{reward_metric}"]]), errors="coerce").iloc[0]
                last = pd.to_numeric(pd.Series([row[f"last10pct_mean/{reward_metric}"]]), errors="coerce").iloc[0]
                delta = last - first if np.isfinite(first) and np.isfinite(last) else math.nan
                status = "OK" if np.isfinite(delta) and delta >= 0 else "CHECK"
                lines.append(f"- {status}: {row['run_id']} seed{row['seed']} {reward_metric}: first={first:.4g}, last={last:.4g}, delta={delta:.4g}")

    lines.append("")
    lines.append("## Required run coverage checklist")
    lines.append("")
    required_main = {
        "flat_cartpole_balance",
        "cartpole_balance_neural",
        "cartpole_balance_nesy",
        "cartpole_balance_symbolic",
        "flat_cheetah_run",
        "cheetah_run_neural",
        "cheetah_run_nesy",
        "flat_walker_walk",
        "walker_walk_neural",
        "walker_walk_nesy",
        "flat_panda_pick_cube",
        "panda_pick_cube_neural",
        "panda_pick_cube_nesy",
        "panda_pick_cube_symbolic",
        "flat_go1_joystick",
        "go1_joystick_neural",
        "go1_joystick_nesy_phase2",
    }
    found = set(summary_df["run_id"].unique()) if not summary_df.empty and "run_id" in summary_df else set()
    for run_id in sorted(required_main):
        count = int((summary_df["run_id"] == run_id).sum()) if not summary_df.empty and "run_id" in summary_df else 0
        marker = "x" if run_id in found else " "
        status = "OK" if count >= 3 else "MISSING_OR_INCOMPLETE"
        lines.append(f"- [{marker}] {run_id}: {count} seed rows ({status})")
    go1_count = int((summary_df["run_id"] == "go1_joystick_nesy_phase2").sum()) if not summary_df.empty and "run_id" in summary_df else 0
    lines.append(f"- [{'x' if go1_count else ' '}] go1_joystick_nesy_phase2: {go1_count} seed rows")

    lines.append("")
    lines.append("## Checklist failure flags")
    lines.append("")
    if not wide_df.empty:
        flat_envs = set(
            summary_df.loc[summary_df["meta_policy_type"] == "flat", "env_name"].astype(str)
        )
        final_envs = set(summary_df["env_name"].astype(str)) if not summary_df.empty else set()
        missing_flat = sorted(final_envs - flat_envs)
        if missing_flat:
            lines.append("- FATAL: missing flat baseline for environments: " + ", ".join(missing_flat))
        else:
            lines.append("- OK: flat baseline present for every loaded environment.")

        raw_diag_cols = [c for c in wide_df.columns if c.startswith("policy_diag/")]
        if not raw_diag_cols:
            lines.append("- FATAL: missing raw feature diagnostics.")
        else:
            lines.append(f"- OK: raw feature diagnostics found ({len(raw_diag_cols)} metric columns).")

        if "train/critic_abs_td" in wide_df.columns:
            td = pd.to_numeric(wide_df["train/critic_abs_td"], errors="coerce")
            if td.notna().any() and float(td.max()) > 1e6:
                lines.append(f"- FATAL: critic TD instability detected; max={float(td.max()):.4g}.")
            else:
                lines.append("- OK: no monotonic-scale critic TD explosion detected by threshold.")

        usage_cols = [c for c in wide_df.columns if c.startswith("skill_usage/")]
        if usage_cols:
            tmp = wide_df.copy()
            tmp["skill_usage_sum"] = tmp[usage_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            bad_usage = tmp.loc[(tmp["skill_usage_sum"] - 1.0).abs() > 0.05]
            if not bad_usage.empty:
                lines.append("- FATAL: bad skill usage sums detected.")

        panda_rows = summary_df[
            summary_df["run_id"].astype(str).str.contains("panda_pick_cube", na=False)
        ] if not summary_df.empty else pd.DataFrame()
        if not panda_rows.empty:
            for skill in ("1_grasp_cube", "2_lift_cube"):
                col = f"last10pct_mean/skill_usage/{skill}"
                if col not in panda_rows.columns:
                    lines.append(f"- FATAL: Panda missing usage metric for {skill}.")
                else:
                    vals = pd.to_numeric(panda_rows[col], errors="coerce")
                    if vals.fillna(0.0).max() <= 0.0:
                        lines.append(f"- FATAL: Panda {skill} is never selected.")

        hopper_rows = summary_df[
            summary_df["run_id"].astype(str).str.contains("hopper_hop", na=False)
        ] if not summary_df.empty else pd.DataFrame()
        if not hopper_rows.empty:
            metric = _preferred_return_metric(summary_df.columns)
            if metric is not None:
                vals = pd.to_numeric(hopper_rows[f"last10pct_mean/{metric}"], errors="coerce")
                if vals.notna().any() and float(vals.max()) <= 1e-6:
                    lines.append("- FATAL: Hopper final reward is near zero and cannot count as success.")

    (out_dir / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(out_dir: Path, runs_root: Path, records: list[RunRecord]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs_root": str(runs_root.resolve()),
        "num_runs": len(records),
        "files": [],
        "runs": [
            {
                "run_id": r.run_id,
                "stage": r.stage,
                "path": r.rel_path,
                "env_name": r.config.get("ENV_NAME"),
                "policy": r.config.get("POLICY"),
                "meta_policy_type": r.config.get("META_POLICY_TYPE"),
                "seed": r.seed_from_name if r.seed_from_name is not None else r.config.get("SEED"),
                "total_timesteps": r.config.get("TOTAL_TIMESTEPS"),
            }
            for r in records
        ],
    }
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            manifest["files"].append(str(path.relative_to(out_dir)))
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")


def make_zip(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(out_dir)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/verification"), help="Root containing run .pkl files and logs.")
    parser.add_argument("--out", type=Path, default=Path("runs/verification_review"), help="Output review bundle directory.")
    parser.add_argument("--zip", type=Path, default=None, help="Optional zip path. Default: <out>.zip")
    parser.add_argument("--include-checkpoints", action="store_true", help="Copy raw .pkl checkpoints into the review bundle. This can be large.")
    args = parser.parse_args(argv)

    runs_root = args.runs
    out_dir = args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, load_failures = discover_runs(runs_root)
    long_df, wide_df, inventory_df, metric_errors = metrics_to_dataframes(records)
    (
        det_eval_episodes_df,
        det_eval_summary_df,
        task_success_summary_df,
        eval_metric_errors,
    ) = deterministic_eval_to_dataframes(records)
    summary_df = make_final_summary(wide_df)
    learning_trends_df = make_learning_trends(summary_df)
    baseline_comparison_df = make_deterministic_baseline_comparison(det_eval_summary_df)
    if baseline_comparison_df.empty:
        baseline_comparison_df = make_baseline_comparison(summary_df)
    skill_disentanglement_df = make_skill_disentanglement(summary_df)
    mask_diagnostics_df = make_mask_diagnostics(summary_df)
    raw_feature_diagnostics_df = make_raw_feature_diagnostics(wide_df)

    long_df.to_csv(out_dir / "metrics_long.csv", index=False)
    wide_df.to_csv(out_dir / "metrics_wide.csv", index=False)
    inventory_df.to_csv(out_dir / "run_inventory.csv", index=False)
    summary_df.to_csv(out_dir / "final_summary.csv", index=False)
    learning_trends_df.to_csv(out_dir / "learning_trends.csv", index=False)
    baseline_comparison_df.to_csv(out_dir / "baseline_comparison.csv", index=False)
    skill_disentanglement_df.to_csv(out_dir / "skill_disentanglement.csv", index=False)
    mask_diagnostics_df.to_csv(out_dir / "mask_diagnostics.csv", index=False)
    raw_feature_diagnostics_df.to_csv(out_dir / "raw_feature_diagnostics.csv", index=False)
    det_eval_episodes_df.to_csv(out_dir / "det_eval_episodes.csv", index=False)
    det_eval_summary_df.to_csv(out_dir / "det_eval_summary.csv", index=False)
    task_success_summary_df.to_csv(out_dir / "task_success_summary.csv", index=False)

    if load_failures:
        (out_dir / "pickle_load_failures.json").write_text(json.dumps(load_failures, indent=2), encoding="utf-8")
    all_metric_errors = metric_errors + eval_metric_errors
    if all_metric_errors:
        (out_dir / "metric_extraction_errors.json").write_text(
            json.dumps(all_metric_errors, indent=2),
            encoding="utf-8",
        )

    make_plots(wide_df, summary_df, out_dir / "plots")
    copy_supporting_files(runs_root, out_dir, records, include_checkpoints=args.include_checkpoints)
    write_diagnostics(out_dir, records, load_failures, all_metric_errors, long_df, wide_df, summary_df)
    write_manifest(out_dir, runs_root, records)

    zip_path = args.zip or out_dir.with_suffix(".zip")
    make_zip(out_dir, zip_path)
    print(f"Loaded {len(records)} runs from {runs_root}")
    print(f"Wrote review directory: {out_dir}")
    print(f"Wrote review zip: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
