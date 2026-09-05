#!/usr/bin/env python3
"""Write the phase-2 continuous-control NEXUS report from review CSVs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ENVS = [
    "CartpoleBalance",
    "CheetahRun",
    "WalkerWalk",
    "PandaPickCube",
    "Go1JoystickFlatTerrain",
]
VARIANTS = ["flat", "neural", "nesy", "symbolic"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _md_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "env_name" in out:
        out["env_name"] = pd.Categorical(out["env_name"], MAIN_ENVS, ordered=True)
    if "meta_policy_type" in out:
        out["meta_policy_type"] = pd.Categorical(out["meta_policy_type"], VARIANTS, ordered=True)
    sort_cols = [c for c in ["env_name", "meta_policy_type"] if c in out]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def _main_table(det: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    agg = (
        det.groupby(["env_name", "meta_policy_type"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            deterministic_return=("episode_return_mean", "mean"),
            deterministic_return_std=("episode_return_mean", "std"),
            primary_success=("primary_success_rate", "mean"),
        )
        .reset_index()
    )
    ratios = baseline[["env_name", "meta_policy_type", "ratio_to_flat"]].copy()
    return _ordered(agg.merge(ratios, on=["env_name", "meta_policy_type"], how="left"))


def _task_table(det: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for env in MAIN_ENVS:
        env_rows = det[det["env_name"] == env]
        for variant in VARIANTS:
            local = env_rows[env_rows["meta_policy_type"] == variant]
            if local.empty:
                continue
            row: dict[str, object] = {
                "env_name": env,
                "meta_policy_type": variant,
                "primary_success": local["primary_success_rate"].mean(),
                "primary_goal": local["primary_goal_metric"].mean(),
            }
            if env == "CartpoleBalance":
                row["diagnostic"] = f"upright={local['cartpole/upright_fraction'].mean():.3f}"
            elif env == "CheetahRun":
                row["diagnostic"] = f"speed={local['cheetah/forward_velocity_mean'].mean():.3f}"
            elif env == "WalkerWalk":
                row["diagnostic"] = f"stand={local['walker/stand_success_rate'].mean():.3f}"
            elif env == "PandaPickCube":
                row["diagnostic"] = (
                    f"reach={local['panda/reach_success_rate'].mean():.3f}, "
                    f"lift={local['panda/lift_success_rate'].mean():.3f}"
                )
            elif env == "Go1JoystickFlatTerrain":
                row["diagnostic"] = (
                    f"no_fall={local['go1/no_fall_rate'].mean():.3f}, "
                    f"vel_err={local['go1/velocity_tracking_error_mean'].mean():.3f}"
                )
            rows.append(row)
    return _ordered(pd.DataFrame(rows))


def _panda_table(det: pd.DataFrame) -> pd.DataFrame:
    panda = det[det["env_name"] == "PandaPickCube"]
    return _ordered(
        panda.groupby(["env_name", "meta_policy_type"], dropna=False)
        .agg(
            reach=("panda/reach_success_rate", "mean"),
            closed_near_cube=("panda/closed_near_cube_rate", "mean"),
            lift=("panda/lift_success_rate", "mean"),
            place=("panda/place_success_rate", "mean"),
            cube_height_delta=("panda/cube_height_delta_max_mean", "mean"),
        )
        .reset_index()
    )


def _go1_table(det: pd.DataFrame) -> pd.DataFrame:
    go1 = det[det["env_name"] == "Go1JoystickFlatTerrain"]
    return _ordered(
        go1.groupby(["env_name", "meta_policy_type"], dropna=False)
        .agg(
            primary_success=("primary_success_rate", "mean"),
            no_fall=("go1/no_fall_rate", "mean"),
            velocity_error=("go1/velocity_tracking_error_mean", "mean"),
            yaw_error=("go1/yaw_tracking_error_mean", "mean"),
            deterministic_return=("episode_return_mean", "mean"),
        )
        .reset_index()
    )


def _skill_table(skill: pd.DataFrame) -> pd.DataFrame:
    return _ordered(
        skill.groupby(["env_name", "meta_policy_type"], dropna=False)
        .agg(
            active_skills=("num_usage_skills", "mean"),
            usage_entropy=("usage_entropy", "mean"),
            skill_reward_std=("skill_reward_std", "mean"),
        )
        .reset_index()
    )


def _mask_table(mask: pd.DataFrame) -> pd.DataFrame:
    violations = mask[mask["metric"].astype(str).str.contains("violation", na=False)].copy()
    if violations.empty:
        return violations
    return _ordered(
        violations.groupby(["env_name", "meta_policy_type"], dropna=False)
        .agg(mask_violation_rate=("last10pct_mean", "mean"))
        .reset_index()
    )


def _env_info(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else "not captured"


def write_report(review: Path, out: Path) -> None:
    det = _read_csv(review / "det_eval_summary.csv")
    baseline = _read_csv(review / "baseline_comparison.csv")
    skill = _read_csv(review / "skill_disentanglement.csv")
    mask = _read_csv(review / "mask_diagnostics.csv")
    validation = (review / "phase2_validation.md").read_text(encoding="utf-8")
    env_info = _env_info(Path("runs/phase2_env_info_start.txt"))

    neural_pass = int(((baseline["meta_policy_type"] == "neural") & (baseline["ratio_to_flat"] >= 0.8)).sum())
    nesy_pass = int(((baseline["meta_policy_type"] == "nesy") & (baseline["ratio_to_flat"] >= 0.7)).sum())

    text = f"""# Continuous-Control NEXUS Phase-2 Results

Commit at report-generation time: `{_git(['rev-parse', 'HEAD'])}`
Branch: `{_git(['branch', '--show-current'])}`

## Claim Boundary

This phase-2 run is mechanically complete but not fully paper-ready under the strict research gates. HopperHop is intentionally absent from the main matrix. The final matrix contains 51 runs: 17 required configurations times 3 seeds.

The implementation now supports deterministic evaluation, aligned task metrics, task-specific diagnostics, NeSy mask-violation logging, and paper-style collection/plotting. Results are mixed: neural passes the return-ratio gate on {neural_pass}/5 environments; NeSy passes on {nesy_pass}/5, below the 4/5 research gate.

## Environment Info

```text
{env_info}
```

## Main Environment Set

1. CartpoleBalance
2. CheetahRun
3. WalkerWalk
4. PandaPickCube
5. Go1JoystickFlatTerrain

HopperHop was dropped from phase-2 success claims after Phase 1 showed failed behavior.

## Deterministic Return and Success Table

{_md_table(_main_table(det, baseline), ['env_name', 'meta_policy_type', 'seeds', 'deterministic_return', 'ratio_to_flat', 'primary_success'], ['Environment', 'Variant', 'Seeds', 'Deterministic return', 'Ratio to flat', 'Primary success'])}

## Aligned Task Metrics

{_md_table(_task_table(det), ['env_name', 'meta_policy_type', 'primary_success', 'primary_goal', 'diagnostic'], ['Environment', 'Variant', 'Primary success', 'Primary goal metric', 'Diagnostic'])}

## Skill Usage and Disentanglement

{_md_table(_skill_table(skill), ['env_name', 'meta_policy_type', 'active_skills', 'usage_entropy', 'skill_reward_std'], ['Environment', 'Variant', 'Active skills', 'Usage entropy', 'Skill reward std'])}

## Mask Violation Diagnostics

NeSy mask violation should be zero up to floating point tolerance. The final review includes explicit `mask/violation_rate` and per-skill `mask_violation/*` metrics.

{_md_table(_mask_table(mask), ['env_name', 'meta_policy_type', 'mask_violation_rate'], ['Environment', 'Variant', 'Mask violation rate'])}

## Panda Phase Diagnostics

Panda success is measured by cube height, not by the grasp proxy. `grasp_proxy` remains only a rule feature. Lift success is based on max cube height relative to initial/table height.

{_md_table(_panda_table(det), ['meta_policy_type', 'reach', 'closed_near_cube', 'lift', 'place', 'cube_height_delta'], ['Variant', 'Reach', 'Closed near cube', 'Lift', 'Place', 'Max height delta'])}

## Go1 Tuning Decision and Limitation

Go1 tuning selected the active-only actor update config in `configs/go1_joystick_nesy_phase2.yaml`, as documented in `docs/reports/go1_phase2_tuning_decision.md`. Go1 still misses the 0.35 primary-success research gate and should be presented as a weak robotics stress-test limitation.

{_md_table(_go1_table(det), ['meta_policy_type', 'primary_success', 'no_fall', 'velocity_error', 'yaw_error', 'deterministic_return'], ['Variant', 'Primary success', 'No-fall rate', 'Velocity error', 'Yaw error', 'Return'])}

## Figures

Generated figures are under `runs/phase2_review/plots/phase2_paper/` in the handoff bundle.

## Strict Validation Outcome

```text
{validation}
```

## Reproducibility

The handoff bundle includes `final_commit_hash.txt`, `git_status.txt`, `final_diff_stat.txt`, `source_snapshot_<commit>.zip`, `pip_freeze.txt`, `env_info_start.txt`, and final matrix `.out`/`.err` logs.

## Limitations

- The strict research validation fails 5 gates.
- Cartpole and Walker deterministic primary-success thresholds are not met.
- Panda lift success passes the best-NEXUS lift threshold but remains near the boundary; place success is low.
- Go1 remains weak and should be described as a stress-test limitation.
- The policy modules use privileged semantic state features rather than learned RGB features.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("runs/phase2_review"))
    parser.add_argument("--out", type=Path, default=Path("docs/reports/continuous_nexus_phase2_results.md"))
    args = parser.parse_args(argv)
    write_report(args.review, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
