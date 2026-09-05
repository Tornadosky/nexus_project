#!/usr/bin/env python3
"""Validate a phase-2 continuous-control NEXUS handoff bundle/review directory.

This validator deliberately separates hard mechanical gates from softer research-quality gates.
Use `--strict` for the final handoff before uploading results back to ChatGPT.
"""

from __future__ import annotations

import argparse
import math
import zipfile
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
REQUIRED_VARIANTS = {
    "CartpoleBalance": {"flat", "neural", "nesy", "symbolic"},
    "CheetahRun": {"flat", "neural", "nesy"},
    "WalkerWalk": {"flat", "neural", "nesy"},
    "PandaPickCube": {"flat", "neural", "nesy", "symbolic"},
    "Go1JoystickFlatTerrain": {"flat", "neural", "nesy"},
}
MIN_SEEDS = 3
PRIMARY_SUCCESS_THRESHOLDS = {
    "CartpoleBalance": 0.60,
    "CheetahRun": 0.40,
    "WalkerWalk": 0.35,
    "PandaPickCube": 0.20,
    "Go1JoystickFlatTerrain": 0.35,
}


def _extract_if_zip(path: Path, workdir: Path) -> Path:
    if path.is_dir():
        return path
    if not path.exists() or path.suffix.lower() != ".zip":
        raise SystemExit(f"--review must be a directory or .zip, got {path}")
    out = workdir / path.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out)
    # Most bundles put review files under final_research_review or phase2_review.
    for candidate in [out / "phase2_review", out / "final_research_review", out]:
        if (candidate / "final_summary.csv").exists() or (candidate / "metrics_wide.csv").exists():
            return candidate
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def add(lines: list[str], status: str, message: str) -> None:
    lines.append(f"- **{status}** {message}")


def check_files(review: Path, lines: list[str]) -> tuple[int, int]:
    errors = warnings = 0
    required = [
        "metrics_long.csv",
        "metrics_wide.csv",
        "final_summary.csv",
        "baseline_comparison.csv",
        "learning_trends.csv",
        "skill_disentanglement.csv",
        "mask_diagnostics.csv",
        "raw_feature_diagnostics.csv",
        "det_eval_summary.csv",
    ]
    for name in required:
        if (review / name).exists():
            add(lines, "PASS", f"found `{name}`")
        else:
            status = "FAIL" if name == "det_eval_summary.csv" else "WARN"
            add(lines, status, f"missing `{name}`")
            if status == "FAIL":
                errors += 1
            else:
                warnings += 1
    return errors, warnings


def check_matrix(review: Path, lines: list[str]) -> tuple[int, int]:
    errors = warnings = 0
    summary = read_csv(review / "final_summary.csv")
    if summary.empty:
        add(lines, "FAIL", "final_summary.csv is empty or missing")
        return 1, 0
    if "HopperHop" in set(summary.get("env_name", pd.Series(dtype=str)).astype(str)):
        add(lines, "FAIL", "HopperHop appears in the final main summary; phase 2 must drop Hopper from the success matrix")
        errors += 1
    else:
        add(lines, "PASS", "HopperHop absent from final main summary")
    for env, variants in REQUIRED_VARIANTS.items():
        env_df = summary[summary["env_name"].astype(str) == env]
        if env_df.empty:
            add(lines, "FAIL", f"missing environment `{env}`")
            errors += 1
            continue
        for variant in variants:
            rows = env_df[env_df["meta_policy_type"].astype(str) == variant]
            seeds = set(pd.to_numeric(rows.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
            if len(seeds) >= MIN_SEEDS:
                add(lines, "PASS", f"{env}/{variant}: {len(seeds)} seeds")
            else:
                add(lines, "FAIL", f"{env}/{variant}: expected >= {MIN_SEEDS} seeds, got {sorted(seeds)}")
                errors += 1
    return errors, warnings


def check_finite_metrics(review: Path, lines: list[str]) -> tuple[int, int]:
    errors = warnings = 0
    long = read_csv(review / "metrics_long.csv")
    if long.empty or "value" not in long.columns:
        add(lines, "WARN", "metrics_long.csv missing or has no value column")
        return 0, 1
    vals = finite(long["value"])
    bad = int(vals.isna().sum())
    # NaNs can appear in sparse columns after pivoting, but not in long metric values.
    if bad == 0:
        add(lines, "PASS", "metrics_long.csv has no non-finite numeric values")
    else:
        add(lines, "FAIL", f"metrics_long.csv has {bad} non-finite numeric values")
        errors += 1
    return errors, warnings


def check_baseline_ratios(review: Path, lines: list[str], strict: bool) -> tuple[int, int]:
    errors = warnings = 0
    baseline = read_csv(review / "baseline_comparison.csv")
    if baseline.empty:
        add(lines, "FAIL", "baseline_comparison.csv missing/empty")
        return 1, 0
    baseline = baseline[baseline["env_name"].isin(MAIN_ENVS)].copy()
    baseline["ratio_to_flat"] = finite(baseline["ratio_to_flat"])
    for variant, threshold, needed in [("neural", 0.80, 4), ("nesy", 0.70, 4)]:
        rows = baseline[baseline["meta_policy_type"].astype(str) == variant]
        passed = rows[rows["ratio_to_flat"] >= threshold]
        status = "PASS" if len(passed) >= needed else ("FAIL" if strict else "WARN")
        add(lines, status, f"{variant} return ratio gate: {len(passed)}/{len(MAIN_ENVS)} envs >= {threshold:.2f}; required {needed}/5")
        for _, row in rows.sort_values("env_name").iterrows():
            add(lines, "INFO", f"{variant} ratio {row['env_name']}: {row['ratio_to_flat']:.3f}")
        if status == "FAIL":
            errors += 1
        elif status == "WARN":
            warnings += 1
    return errors, warnings


def check_deterministic_eval(review: Path, lines: list[str], strict: bool) -> tuple[int, int]:
    errors = warnings = 0
    eval_summary = read_csv(review / "det_eval_summary.csv")
    if eval_summary.empty:
        add(lines, "FAIL", "det_eval_summary.csv is missing/empty")
        return 1, 0
    required_cols = ["env_name", "meta_policy_type", "seed", "episode_return_mean", "primary_success_rate"]
    for col in required_cols:
        if col not in eval_summary.columns:
            add(lines, "FAIL", f"det_eval_summary.csv missing required column `{col}`")
            errors += 1
    if errors:
        return errors, warnings
    if "HopperHop" in set(eval_summary["env_name"].astype(str)):
        add(lines, "FAIL", "HopperHop appears in deterministic eval; do not include it in phase-2 final eval")
        errors += 1
    eval_summary["primary_success_rate"] = finite(eval_summary["primary_success_rate"])
    grouped = eval_summary[eval_summary["env_name"].isin(MAIN_ENVS)].groupby(
        ["env_name", "meta_policy_type"], dropna=False
    )["primary_success_rate"].agg(["mean", "std", "count"]).reset_index()
    for env, threshold in PRIMARY_SUCCESS_THRESHOLDS.items():
        rows = grouped[(grouped["env_name"].astype(str) == env) & (grouped["meta_policy_type"].isin(["neural", "nesy", "symbolic"]))]
        best = float(rows["mean"].max()) if not rows.empty else math.nan
        status = "PASS" if np.isfinite(best) and best >= threshold else ("FAIL" if strict else "WARN")
        add(lines, status, f"{env}: best NEXUS primary success {best:.3f}; threshold {threshold:.2f}")
        if status == "FAIL":
            errors += 1
        elif status == "WARN":
            warnings += 1
    panda = eval_summary[eval_summary["env_name"].astype(str) == "PandaPickCube"]
    for col, threshold in [("panda/reach_success_rate", 0.50), ("panda/lift_success_rate", 0.20)]:
        if col not in panda.columns:
            status = "FAIL" if strict else "WARN"
            add(lines, status, f"Panda deterministic eval missing `{col}`")
            errors += int(status == "FAIL")
            warnings += int(status == "WARN")
            continue
        vals = finite(panda[panda["meta_policy_type"].isin(["neural", "nesy", "symbolic"])][col])
        best = float(vals.max()) if len(vals.dropna()) else math.nan
        status = "PASS" if np.isfinite(best) and best >= threshold else ("FAIL" if strict else "WARN")
        add(lines, status, f"Panda {col}: best {best:.3f}; threshold {threshold:.2f}")
        errors += int(status == "FAIL")
        warnings += int(status == "WARN")
    return errors, warnings


def check_mask_and_skills(review: Path, lines: list[str], strict: bool) -> tuple[int, int]:
    errors = warnings = 0
    summary = read_csv(review / "final_summary.csv")
    if not summary.empty:
        usage_cols = [c for c in summary.columns if c.startswith("last10pct_mean/skill_usage/")]
        if usage_cols:
            for _, row in summary[summary["env_name"].isin(MAIN_ENVS)].iterrows():
                vals = finite(pd.Series([row.get(c, np.nan) for c in usage_cols])).dropna().to_numpy(dtype=float)
                if vals.size:
                    usage_sum = float(vals.sum())
                    if abs(usage_sum - 1.0) > 1e-3:
                        add(lines, "FAIL", f"skill usage does not sum to 1 for {row['run_id']} seed {row['seed']}: {usage_sum:.6f}")
                        errors += 1
            add(lines, "PASS", "checked skill-usage sums where available")
    mask = read_csv(review / "mask_diagnostics.csv")
    if mask.empty:
        add(lines, "WARN", "mask_diagnostics.csv missing")
        warnings += 1
    else:
        # Prefer explicit violation metric if Codex adds it.
        violation_cols = [c for c in mask.columns if "violation" in c.lower()]
        if not violation_cols:
            add(lines, "WARN", "explicit mask violation metric not found; add mask_violation_rate in phase 2")
            warnings += 1
        else:
            add(lines, "PASS", f"mask violation columns present: {violation_cols}")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="Review directory or zipped handoff bundle")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/nexus_phase2_validate"))
    parser.add_argument("--strict", action="store_true", help="Fail on research-quality gates, not just mechanical gates")
    parser.add_argument("--out", type=Path, default=None, help="Optional markdown report path")
    args = parser.parse_args(argv)

    review = _extract_if_zip(args.review, args.workdir)
    lines: list[str] = ["# Phase-2 NEXUS Validation", "", f"Review directory: `{review}`", ""]
    total_errors = total_warnings = 0
    for title, fn in [
        ("Files", lambda: check_files(review, lines)),
        ("Matrix", lambda: check_matrix(review, lines)),
        ("Finite metrics", lambda: check_finite_metrics(review, lines)),
        ("Baseline ratios", lambda: check_baseline_ratios(review, lines, args.strict)),
        ("Deterministic eval", lambda: check_deterministic_eval(review, lines, args.strict)),
        ("Masks and skills", lambda: check_mask_and_skills(review, lines, args.strict)),
    ]:
        lines.extend(["", f"## {title}"])
        errors, warnings = fn()
        total_errors += errors
        total_warnings += warnings
    lines.extend(["", "## Summary", f"- errors: {total_errors}", f"- warnings: {total_warnings}", f"- strict: {args.strict}"])
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
