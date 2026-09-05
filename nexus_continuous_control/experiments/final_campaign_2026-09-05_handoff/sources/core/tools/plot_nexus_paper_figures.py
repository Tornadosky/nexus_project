#!/usr/bin/env python3
"""Create paper-style figures from a collected NEXUS review directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_FILES = {
    "main_return_curves": "main_return_curves.png",
    "final_performance_vs_flat": "final_performance_vs_flat.png",
    "skill_reward_curves_by_env": "skill_reward_curves_by_env.png",
    "skill_usage_by_env_variant": "skill_usage_by_env_variant.png",
    "mask_availability_vs_selection": "mask_availability_vs_selection.png",
    "panda_phase_diagnostics": "panda_phase_diagnostics.png",
    "loss_and_td_diagnostics": "loss_and_td_diagnostics.png",
    "raw_feature_diagnostics": "raw_feature_diagnostics.png",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metric(candidates: Iterable[str], columns: Iterable[str]) -> str | None:
    cols = set(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def _placeholder(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _mean_curve_plot(
    df: pd.DataFrame,
    metrics: list[str],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    if df.empty or not metrics:
        _placeholder(path, title, "No matching metrics found.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    any_plotted = False
    for metric in metrics:
        local = df[["env_step", "env_name", "meta_policy_type", "seed", metric]].copy()
        local[metric] = pd.to_numeric(local[metric], errors="coerce")
        local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=["env_step", metric])
        if local.empty:
            continue
        grouped = (
            local.groupby(["env_name", "meta_policy_type", "env_step"], dropna=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        for (env, variant), group in grouped.groupby(["env_name", "meta_policy_type"], dropna=False):
            group = group.sort_values("env_step")
            label = f"{env}/{variant}" if len(metrics) == 1 else f"{env}/{variant}/{metric}"
            x = group["env_step"].to_numpy(dtype=float)
            mean = group["mean"].to_numpy(dtype=float)
            std = group["std"].fillna(0.0).to_numpy(dtype=float)
            plt.plot(x, mean, label=label, linewidth=1.7)
            plt.fill_between(x, mean - std, mean + std, alpha=0.12)
            any_plotted = True
    if not any_plotted:
        _placeholder(path, title, "Metrics were present but all values were non-numeric.")
        return
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_main_return_curves(wide: pd.DataFrame, out: Path) -> None:
    metric = _metric(
        ["env/returned_episode_returns", "returns/env_reward_mean", "env/original_reward"],
        wide.columns,
    )
    _mean_curve_plot(
        wide,
        [metric] if metric else [],
        out / PLOT_FILES["main_return_curves"],
        "Main Return Curves",
        metric or "return",
    )


def plot_final_performance_vs_flat(baseline: pd.DataFrame, out: Path) -> None:
    path = out / PLOT_FILES["final_performance_vs_flat"]
    if baseline.empty or "ratio_to_flat" not in baseline.columns:
        _placeholder(path, "Final Performance vs Flat", "No baseline comparison table found.")
        return
    df = baseline.copy()
    df["ratio_to_flat"] = pd.to_numeric(df["ratio_to_flat"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_to_flat"])
    if df.empty:
        _placeholder(path, "Final Performance vs Flat", "No finite ratio_to_flat values.")
        return
    labels = [f"{r.env_name}\n{r.meta_policy_type}" for r in df.itertuples(index=False)]
    x = np.arange(len(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(labels) * 0.7), 6))
    plt.bar(x, df["ratio_to_flat"].to_numpy(dtype=float))
    plt.axhline(0.8, color="black", linestyle="--", linewidth=1, label="80%")
    plt.axhline(0.7, color="gray", linestyle=":", linewidth=1, label="70%")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Final-window ratio to flat")
    plt.title("Final Performance vs Flat AC-PQN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_skill_curves(wide: pd.DataFrame, out: Path, prefix: str, filename_key: str, title: str) -> None:
    metrics = sorted([c for c in wide.columns if c.startswith(prefix)])
    _mean_curve_plot(wide, metrics, out / PLOT_FILES[filename_key], title, prefix.rstrip("/"))


def plot_mask(wide: pd.DataFrame, out: Path) -> None:
    metrics = sorted(
        [
            c
            for c in wide.columns
            if c.startswith("mask_available/") or c.startswith("mask_selected_given_available/")
        ]
    )
    _mean_curve_plot(
        wide,
        metrics,
        out / PLOT_FILES["mask_availability_vs_selection"],
        "NeSy Mask Availability vs Selection",
        "rate",
    )


def plot_panda(wide: pd.DataFrame, out: Path) -> None:
    panda = wide[wide.get("run_id", pd.Series(dtype=str)).astype(str).str.contains("panda_pick_cube")]
    metrics = [
        c
        for c in [
            "policy_diag/panda/dist_tcp_cube",
            "policy_diag/panda/grasped",
            "policy_diag/panda/cube_height",
            "env/returned_episode_returns",
            "returns/env_reward_mean",
        ]
        if c in panda.columns
    ]
    metrics += sorted([c for c in panda.columns if c.startswith("skill_usage/")])
    _mean_curve_plot(
        panda,
        metrics,
        out / PLOT_FILES["panda_phase_diagnostics"],
        "Panda Phase Diagnostics",
        "value",
    )


def plot_loss_td(wide: pd.DataFrame, out: Path) -> None:
    metrics = [
        c
        for c in [
            "train/actor_loss",
            "train/critic_loss",
            "train/meta_loss",
            "train/critic_abs_td",
            "train/meta_abs_td",
        ]
        if c in wide.columns
    ]
    _mean_curve_plot(
        wide,
        metrics,
        out / PLOT_FILES["loss_and_td_diagnostics"],
        "Loss and TD Diagnostics",
        "value",
    )


def plot_raw_features(raw: pd.DataFrame, out: Path) -> None:
    path = out / PLOT_FILES["raw_feature_diagnostics"]
    if raw.empty or "feature" not in raw.columns:
        _placeholder(path, "Raw Feature Diagnostics", "No raw feature diagnostics table found.")
        return
    df = raw.copy()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["mean"])
    if df.empty:
        _placeholder(path, "Raw Feature Diagnostics", "No finite raw feature means.")
        return
    grouped = df.groupby("feature", dropna=False)["mean"].mean().sort_values()
    labels = grouped.index.astype(str).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(labels) * 0.45), 6))
    plt.bar(np.arange(len(labels)), grouped.to_numpy(dtype=float))
    plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("Mean raw feature value")
    plt.title("Raw Feature Diagnostics")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    wide = _read_csv(args.review / "metrics_wide.csv")
    baseline = _read_csv(args.review / "baseline_comparison.csv")
    raw = _read_csv(args.review / "raw_feature_diagnostics.csv")
    args.out.mkdir(parents=True, exist_ok=True)

    plot_main_return_curves(wide, args.out)
    plot_final_performance_vs_flat(baseline, args.out)
    plot_skill_curves(
        wide,
        args.out,
        "skill_reward/",
        "skill_reward_curves_by_env",
        "Skill Reward Curves by Environment",
    )
    plot_skill_curves(
        wide,
        args.out,
        "skill_usage/",
        "skill_usage_by_env_variant",
        "Skill Usage by Environment and Variant",
    )
    plot_mask(wide, args.out)
    plot_panda(wide, args.out)
    plot_loss_td(wide, args.out)
    plot_raw_features(raw, args.out)
    print(f"Wrote paper figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
