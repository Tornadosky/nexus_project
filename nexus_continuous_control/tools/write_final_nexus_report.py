#!/usr/bin/env python3
"""Write the final continuous-control NEXUS report from collector outputs."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> list[str]:
    lines = [f"python/platform: {platform.platform()}"]
    for name in ("jax", "mujoco", "mujoco_playground"):
        try:
            module = __import__(name)
            lines.append(f"{name}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{name}: unavailable ({exc!r})")
    return lines


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "_No matching columns._"
    out = df[cols].head(max_rows).copy()
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in cols]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _config_list(config_dir: Path) -> str:
    if not config_dir.exists():
        return "_No config directory found._"
    lines = []
    for path in sorted(config_dir.glob("*.yaml")):
        lines.append(f"- `{path.as_posix()}`")
    return "\n".join(lines) if lines else "_No configs found._"


def _policy_table() -> str:
    return "\n".join(
        [
            "| Environment | Skills | Rule summary |",
            "| --- | --- | --- |",
            "| CartpoleBalance | recover_balance, center_cart, damp_motion | angle/velocity recovery, cart centering, velocity damping |",
            "| CheetahRun | accelerate_forward, stabilize_posture, energy_efficient_run | speed, posture, and control-cost rewards |",
            "| WalkerWalk | stand_recover, walk_forward, stabilize_gait, energy_efficient | height/uprightness, target speed, gait stability, torque efficiency |",
            "| HopperHop | stand_recover, hop_forward, stabilize_landing, energy_efficient | survival/uprightness plus env-reward tracking skills |",
            "| PandaPickCube | reach_cube, grasp_cube, lift_cube, place_or_stabilize | distance-to-cube, grasp, height, target placement phases |",
            "| Go1JoystickFlatTerrain | stand, track_velocity, turn, recover | stance, command tracking, yaw tracking, fall recovery |",
        ]
    )


def _plot_links() -> str:
    plots = [
        "plots/paper/main_return_curves.png",
        "plots/paper/final_performance_vs_flat.png",
        "plots/paper/skill_reward_curves_by_env.png",
        "plots/paper/skill_usage_by_env_variant.png",
        "plots/paper/mask_availability_vs_selection.png",
        "plots/paper/panda_phase_diagnostics.png",
        "plots/paper/loss_and_td_diagnostics.png",
        "plots/paper/raw_feature_diagnostics.png",
    ]
    return "\n".join(f"- `{plot}`" for plot in plots)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = _read_csv(args.review / "final_summary.csv")
    baseline = _read_csv(args.review / "baseline_comparison.csv")
    trends = _read_csv(args.review / "learning_trends.csv")
    skills = _read_csv(args.review / "skill_disentanglement.csv")
    masks = _read_csv(args.review / "mask_diagnostics.csv")
    raw = _read_csv(args.review / "raw_feature_diagnostics.csv")
    diagnostics = (args.review / "diagnostics.md").read_text(encoding="utf-8") if (
        args.review / "diagnostics.md"
    ).exists() else "_No diagnostics file found._"

    report = [
        "# Continuous-Control NEXUS Results",
        "",
        f"Commit: `{_git_commit()}`",
        "",
        "## Environment Info",
        "",
        "\n".join(f"- {line}" for line in _versions()),
        "",
        "## Method Summary",
        "",
        (
            "Continuous-control NEXUS keeps the NEXUS meta-policy over interpretable skills, "
            "but replaces discrete option values with deterministic skill actors and skill "
            "critics. Skill critics train from shared rollouts on skill-specific rewards, "
            "while learned meta policies train on environment reward with masked max-Q "
            "bootstraps for NeSy."
        ),
        "",
        "## Exact Configs",
        "",
        _config_list(Path("configs")),
        "",
        "## Environment and Skill Table",
        "",
        _policy_table(),
        "",
        "## Main Performance",
        "",
        _markdown_table(
            summary,
            [
                "run_id",
                "seed",
                "env_name",
                "meta_policy_type",
                "final_env_step",
                "last10pct_mean/env/returned_episode_returns",
                "last10pct_mean/returns/env_reward_mean",
                "last10pct_mean/train/critic_abs_td",
            ],
        ),
        "",
        "## Baseline Comparison",
        "",
        _markdown_table(
            baseline,
            [
                "env_name",
                "meta_policy_type",
                "metric",
                "final_mean",
                "final_std",
                "num_seeds",
                "flat_final_mean",
                "ratio_to_flat",
            ],
        ),
        "",
        "## Learning Trends",
        "",
        _markdown_table(
            trends,
            [
                "run_id",
                "seed",
                "env_name",
                "meta_policy_type",
                "first10pct_mean",
                "last10pct_mean",
                "delta",
                "positive_learning_trend",
            ],
        ),
        "",
        "## Skill and Mask Diagnostics",
        "",
        _markdown_table(
            skills,
            [
                "run_id",
                "seed",
                "env_name",
                "meta_policy_type",
                "num_usage_skills",
                "usage_entropy",
                "skill_reward_std",
                "skill_rewards_nonconstant",
            ],
        ),
        "",
        _markdown_table(
            masks,
            ["run_id", "seed", "env_name", "meta_policy_type", "kind", "skill", "last10pct_mean"],
            max_rows=60,
        ),
        "",
        "## Raw Feature Diagnostics",
        "",
        _markdown_table(
            raw,
            ["run_id", "seed", "env_name", "meta_policy_type", "feature", "mean", "std", "min", "max"],
            max_rows=60,
        ),
        "",
        "## Plots",
        "",
        _plot_links(),
        "",
        "## Limitations and Failure Cases",
        "",
        diagnostics,
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
