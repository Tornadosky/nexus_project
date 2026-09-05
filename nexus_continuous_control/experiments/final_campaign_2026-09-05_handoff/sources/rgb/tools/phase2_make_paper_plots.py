#!/usr/bin/env python3
"""Create phase-2, paper-style plots for continuous-control NEXUS.

This script is intentionally standalone. Copy it into `nexus_continuous_control/tools/`
and run it after `collect_nexus_results.py` has produced a review directory. It also
uses deterministic-evaluation CSVs when Codex adds them in phase 2.

Expected inputs in --review:
  - metrics_wide.csv
  - baseline_comparison.csv
  - skill_disentanglement.csv
  - mask_diagnostics.csv
  - raw_feature_diagnostics.csv
  - det_eval_summary.csv                 optional but strongly recommended
  - det_eval_episodes.csv                optional
  - task_success_summary.csv             optional; det_eval_summary is used as fallback

Outputs under --out:
  - phase2_return_vs_flat.png
  - phase2_deterministic_task_success.png
  - phase2_training_return_curves_by_env.png
  - phase2_skill_usage_heatmap.png
  - phase2_panda_eval_phases.png
  - phase2_nesy_mask_diagnostics.png
  - phase2_td_stability.png
  - phase2_gate_summary.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN_ENVS = [
    "CartpoleBalance",
    "CheetahRun",
    "WalkerWalk",
    "PandaPickCube",
    "Go1JoystickFlatTerrain",
]
VARIANT_ORDER = ["flat", "neural", "nesy", "symbolic"]
VARIANT_LABEL = {"flat": "Flat AC-PQN", "neural": "NEXUS neural", "nesy": "NEXUS NeSy", "symbolic": "NEXUS symbolic"}
PANDA_METRIC_ORDER = [
    "panda/reach_success_rate",
    "panda/closed_near_cube_rate",
    "panda/lift_success_rate",
    "panda/place_success_rate",
]
PANDA_METRIC_LABEL = {
    "panda/reach_success_rate": "Reach",
    "panda/closed_near_cube_rate": "Close gripper near cube",
    "panda/lift_success_rate": "Lift",
    "panda/place_success_rate": "Place / target",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def finite_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def env_variant_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["env_name"] = pd.Categorical(out["env_name"], MAIN_ENVS, ordered=True)
    out["meta_policy_type"] = pd.Categorical(out["meta_policy_type"], VARIANT_ORDER, ordered=True)
    return out.sort_values(["env_name", "meta_policy_type"]).reset_index(drop=True)


def placeholder(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, values: Iterable[float], precision: int = 2) -> None:
    for patch, value in zip(ax.patches, values):
        if not np.isfinite(value):
            continue
        height = patch.get_height()
        ax.annotate(
            f"{value:.{precision}f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(ax.patches) > 12 else 0,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_return_vs_flat(baseline: pd.DataFrame, out: Path) -> None:
    path = out / "phase2_return_vs_flat.png"
    if baseline.empty or "ratio_to_flat" not in baseline.columns:
        placeholder(path, "Final deterministic/training return vs flat", "baseline_comparison.csv missing or incomplete")
        return
    df = baseline[baseline["env_name"].isin(MAIN_ENVS)].copy()
    df = env_variant_order(df)
    df["ratio_to_flat"] = finite_numeric(df["ratio_to_flat"])
    df = df.dropna(subset=["ratio_to_flat"])
    if df.empty:
        placeholder(path, "Final deterministic/training return vs flat", "No finite ratio_to_flat values")
        return

    envs = [e for e in MAIN_ENVS if e in set(df["env_name"].astype(str))]
    variants = [v for v in VARIANT_ORDER if v in set(df["meta_policy_type"].astype(str))]
    width = 0.78 / max(len(variants), 1)
    x = np.arange(len(envs))
    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(envs)), 5.5))
    plotted_values: list[float] = []
    for i, variant in enumerate(variants):
        vals = []
        for env in envs:
            rows = df[(df["env_name"].astype(str) == env) & (df["meta_policy_type"].astype(str) == variant)]
            vals.append(float(rows["ratio_to_flat"].iloc[0]) if not rows.empty else math.nan)
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=VARIANT_LABEL.get(variant, variant))
        plotted_values.extend(vals)
    ax.axhline(1.0, linestyle="-", linewidth=1.0, label="flat parity")
    ax.axhline(0.8, linestyle="--", linewidth=1.0, label="80% gate")
    ax.axhline(0.7, linestyle=":", linewidth=1.0, label="70% gate")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("Final-window return ratio to flat")
    ax.set_title("NEXUS performance relative to flat AC-PQN")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _success_source(review: Path) -> pd.DataFrame:
    task = read_csv(review / "task_success_summary.csv")
    if not task.empty:
        return task
    eval_summary = read_csv(review / "det_eval_summary.csv")
    if not eval_summary.empty:
        return eval_summary
    return pd.DataFrame()


def plot_task_success(review: Path, out: Path) -> None:
    path = out / "phase2_deterministic_task_success.png"
    df = _success_source(review)
    if df.empty:
        placeholder(path, "Deterministic task-success metrics", "No task_success_summary.csv or det_eval_summary.csv")
        return
    if "primary_success_rate" not in df.columns:
        placeholder(path, "Deterministic task-success metrics", "primary_success_rate column missing")
        return
    df = df[df["env_name"].isin(MAIN_ENVS)].copy()
    df["primary_success_rate"] = finite_numeric(df["primary_success_rate"])
    if "seed" in df.columns:
        grouped = df.groupby(["env_name", "meta_policy_type"], dropna=False)["primary_success_rate"].agg(["mean", "std"]).reset_index()
    else:
        grouped = df.rename(columns={"primary_success_rate": "mean"})
        grouped["std"] = 0.0
    grouped = env_variant_order(grouped).dropna(subset=["mean"])
    if grouped.empty:
        placeholder(path, "Deterministic task-success metrics", "No finite primary_success_rate values")
        return

    envs = [e for e in MAIN_ENVS if e in set(grouped["env_name"].astype(str))]
    variants = [v for v in VARIANT_ORDER if v in set(grouped["meta_policy_type"].astype(str))]
    width = 0.78 / max(len(variants), 1)
    x = np.arange(len(envs))
    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(envs)), 5.5))
    for i, variant in enumerate(variants):
        vals, errs = [], []
        for env in envs:
            rows = grouped[(grouped["env_name"].astype(str) == env) & (grouped["meta_policy_type"].astype(str) == variant)]
            vals.append(float(rows["mean"].iloc[0]) if not rows.empty else math.nan)
            errs.append(float(rows["std"].iloc[0]) if not rows.empty and np.isfinite(rows["std"].iloc[0]) else 0.0)
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=VARIANT_LABEL.get(variant, variant))
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("Deterministic primary success rate")
    ax.set_title("Aligned task-success metrics, deterministic evaluation")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_training_return_curves(wide: pd.DataFrame, out: Path) -> None:
    path = out / "phase2_training_return_curves_by_env.png"
    metric = None
    for candidate in ["env/returned_episode_returns", "returns/env_reward_mean", "env/original_reward"]:
        if candidate in wide.columns:
            metric = candidate
            break
    if wide.empty or metric is None:
        placeholder(path, "Training return curves", "No return metric found in metrics_wide.csv")
        return
    df = wide[wide["env_name"].isin(MAIN_ENVS)].copy()
    df[metric] = finite_numeric(df[metric])
    df["env_step"] = finite_numeric(df["env_step"])
    df = df.dropna(subset=["env_step", metric])
    if df.empty:
        placeholder(path, "Training return curves", "No finite curve values")
        return

    envs = [e for e in MAIN_ENVS if e in set(df["env_name"].astype(str))]
    ncols = 2
    nrows = int(math.ceil(len(envs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows), squeeze=False)
    for ax, env in zip(axes.ravel(), envs):
        local = df[df["env_name"].astype(str) == env]
        grouped = local.groupby(["meta_policy_type", "env_step"], dropna=False)[metric].agg(["mean", "std"]).reset_index()
        for variant in VARIANT_ORDER:
            g = grouped[grouped["meta_policy_type"].astype(str) == variant].sort_values("env_step")
            if g.empty:
                continue
            x = g["env_step"].to_numpy(dtype=float)
            mean = g["mean"].to_numpy(dtype=float)
            std = g["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mean, linewidth=1.6, label=VARIANT_LABEL.get(variant, variant))
            ax.fill_between(x, mean - std, mean + std, alpha=0.12)
        ax.set_title(env)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(envs):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle("Training return curves by environment", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_skill_usage_heatmap(summary: pd.DataFrame, out: Path) -> None:
    path = out / "phase2_skill_usage_heatmap.png"
    if summary.empty:
        placeholder(path, "Skill usage heatmap", "final_summary.csv missing")
        return
    usage_cols = [c for c in summary.columns if c.startswith("last10pct_mean/skill_usage/")]
    if not usage_cols:
        placeholder(path, "Skill usage heatmap", "No skill_usage columns found")
        return
    df = summary[summary["env_name"].isin(MAIN_ENVS)].copy()
    rows, labels, skills = [], [], []
    for _, row in env_variant_order(df).iterrows():
        row_vals = []
        row_skills = []
        for col in usage_cols:
            value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(value):
                row_vals.append(float(value))
                row_skills.append(col.split("skill_usage/", 1)[1])
        if row_vals:
            labels.append(f"{row['env_name']} / {row['meta_policy_type']}")
            skills = row_skills if len(row_skills) > len(skills) else skills
            # Pad to current max width later.
            rows.append(row_vals)
    if not rows:
        placeholder(path, "Skill usage heatmap", "No finite skill usage values")
        return
    maxw = max(len(r) for r in rows)
    mat = np.full((len(rows), maxw), np.nan)
    for i, row in enumerate(rows):
        mat[i, : len(row)] = row
    fig, ax = plt.subplots(figsize=(max(8, maxw * 1.1), max(6, len(rows) * 0.28)))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(maxw))
    ax.set_xticklabels([f"skill {i}" for i in range(maxw)], rotation=45, ha="right")
    ax.set_title("Final-window skill usage")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Usage probability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_panda_eval(review: Path, out: Path) -> None:
    path = out / "phase2_panda_eval_phases.png"
    df = _success_source(review)
    if df.empty:
        placeholder(path, "Panda deterministic phase metrics", "No deterministic/task success summary found")
        return
    panda = df[df["env_name"].astype(str) == "PandaPickCube"].copy()
    metrics = [m for m in PANDA_METRIC_ORDER if m in panda.columns]
    if panda.empty or not metrics:
        placeholder(path, "Panda deterministic phase metrics", "Panda phase columns missing")
        return
    records = []
    for variant in VARIANT_ORDER:
        rows = panda[panda["meta_policy_type"].astype(str) == variant]
        if rows.empty:
            continue
        for metric in metrics:
            vals = finite_numeric(rows[metric]).dropna()
            if len(vals):
                records.append({"variant": variant, "metric": metric, "mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)})
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        placeholder(path, "Panda deterministic phase metrics", "No finite Panda phase values")
        return
    variants = [v for v in VARIANT_ORDER if v in set(plot_df["variant"])]
    width = 0.78 / max(len(metrics), 1)
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(variants)), 5.5))
    for i, metric in enumerate(metrics):
        vals, errs = [], []
        for variant in variants:
            row = plot_df[(plot_df["variant"] == variant) & (plot_df["metric"] == metric)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else math.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=PANDA_METRIC_LABEL.get(metric, metric))
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL.get(v, v) for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Success / phase rate")
    ax.set_title("PandaPickCube deterministic phase metrics")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mask_diagnostics(mask: pd.DataFrame, out: Path) -> None:
    path = out / "phase2_nesy_mask_diagnostics.png"
    if mask.empty:
        placeholder(path, "NeSy mask diagnostics", "mask_diagnostics.csv missing")
        return
    df = mask[(mask["env_name"].isin(MAIN_ENVS)) & (mask["meta_policy_type"].astype(str) == "nesy")].copy()
    if df.empty:
        placeholder(path, "NeSy mask diagnostics", "No NeSy rows found")
        return
    df["last10pct_mean"] = finite_numeric(df["last10pct_mean"])
    df = df.dropna(subset=["last10pct_mean"])
    df = df[df["kind"].isin(["mask_available", "mask_selected_given_available"])]
    if df.empty:
        placeholder(path, "NeSy mask diagnostics", "No finite mask availability/selection rows")
        return
    grouped = df.groupby(["env_name", "kind"], dropna=False)["last10pct_mean"].mean().reset_index()
    envs = [e for e in MAIN_ENVS if e in set(grouped["env_name"].astype(str))]
    kinds = ["mask_available", "mask_selected_given_available"]
    x = np.arange(len(envs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(envs)), 5.0))
    for i, kind in enumerate(kinds):
        vals = []
        for env in envs:
            rows = grouped[(grouped["env_name"].astype(str) == env) & (grouped["kind"].astype(str) == kind)]
            vals.append(float(rows["last10pct_mean"].iloc[0]) if not rows.empty else math.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=kind)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("NeSy mask availability vs selection")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_td_stability(wide: pd.DataFrame, out: Path) -> None:
    path = out / "phase2_td_stability.png"
    metrics = [m for m in ["train/critic_abs_td", "train/meta_abs_td", "train/actor_grad_norm", "train/critic_grad_norm"] if m in wide.columns]
    if wide.empty or not metrics:
        placeholder(path, "TD / gradient stability", "No TD/gradient metrics found")
        return
    df = wide[wide["env_name"].isin(MAIN_ENVS)].copy()
    ncols = 2
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        df[metric] = finite_numeric(df[metric])
        local = df.dropna(subset=["env_step", metric])
        grouped = local.groupby(["meta_policy_type", "env_step"], dropna=False)[metric].agg(["mean", "std"]).reset_index()
        for variant in VARIANT_ORDER:
            g = grouped[grouped["meta_policy_type"].astype(str) == variant].sort_values("env_step")
            if g.empty:
                continue
            x = g["env_step"].to_numpy(dtype=float)
            mean = g["mean"].to_numpy(dtype=float)
            std = g["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mean, linewidth=1.5, label=VARIANT_LABEL.get(variant, variant))
            ax.fill_between(x, mean - std, mean + std, alpha=0.12)
        ax.set_title(metric)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle("Training stability diagnostics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_gate_summary(review: Path, out: Path) -> None:
    baseline = read_csv(review / "baseline_comparison.csv")
    success = _success_source(review)
    path = out / "phase2_gate_summary.md"
    lines = ["# Phase-2 Continuous NEXUS Gate Summary", ""]
    lines.append("## Main environment set")
    lines.extend(f"- {env}" for env in MAIN_ENVS)
    lines.append("")
    if not baseline.empty:
        df = baseline[baseline["env_name"].isin(MAIN_ENVS)].copy()
        df["ratio_to_flat"] = finite_numeric(df["ratio_to_flat"])
        lines.append("## Return ratio gates")
        for variant, threshold in [("neural", 0.8), ("nesy", 0.7)]:
            rows = df[df["meta_policy_type"].astype(str) == variant]
            passed = rows[rows["ratio_to_flat"] >= threshold]
            lines.append(f"- {variant}: {len(passed)}/{len(MAIN_ENVS)} envs >= {threshold:.2f} flat ratio")
            for _, row in rows.sort_values("env_name").iterrows():
                lines.append(f"  - {row['env_name']}: {row['ratio_to_flat']:.3f}")
        lines.append("")
    else:
        lines.append("## Return ratio gates\n- baseline_comparison.csv missing\n")

    if not success.empty and "primary_success_rate" in success.columns:
        lines.append("## Deterministic aligned success")
        df = success[success["env_name"].isin(MAIN_ENVS)].copy()
        df["primary_success_rate"] = finite_numeric(df["primary_success_rate"])
        grouped = df.groupby(["env_name", "meta_policy_type"], dropna=False)["primary_success_rate"].agg(["mean", "std", "count"]).reset_index()
        for _, row in env_variant_order(grouped).iterrows():
            lines.append(f"- {row['env_name']} / {row['meta_policy_type']}: {row['mean']:.3f} ± {row['std'] if np.isfinite(row['std']) else 0.0:.3f} ({int(row['count'])} seeds)")
        lines.append("")
    else:
        lines.append("## Deterministic aligned success\n- Missing deterministic success summary. This is a phase-2 blocker.\n")

    lines.append("## Hard blockers I will reject")
    lines.extend(
        [
            "- HopperHop included as a main success environment.",
            "- det_eval_summary.csv missing or lacking primary_success_rate.",
            "- Panda reports grasp/lift success only from the grasp proxy without cube-height evidence.",
            "- NeSy mask violations are not logged.",
            "- Source snapshot or final commit hash missing from the handoff bundle.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="Review directory from collect_nexus_results.py")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for phase-2 plots")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    wide = read_csv(args.review / "metrics_wide.csv")
    summary = read_csv(args.review / "final_summary.csv")
    baseline = read_csv(args.review / "baseline_comparison.csv")
    mask = read_csv(args.review / "mask_diagnostics.csv")

    plot_return_vs_flat(baseline, args.out)
    plot_task_success(args.review, args.out)
    plot_training_return_curves(wide, args.out)
    plot_skill_usage_heatmap(summary, args.out)
    plot_panda_eval(args.review, args.out)
    plot_mask_diagnostics(mask, args.out)
    plot_td_stability(wide, args.out)
    make_gate_summary(args.review, args.out)
    print(f"Wrote phase-2 plots and gate summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
