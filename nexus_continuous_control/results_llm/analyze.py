""" Analyze LLM results by extracting from <env>/manifest.json

Output:
tables/env_comparison_summary.csv   - hand vs llm(initial) vs refined(final), per env
tables/per_seed_results.csv         - one row per (env, condition, seed)
tables/skill_usage_handwritten.csv  - mean skill-usage fraction per hand-written skill
tables/skillset_sizes.csv           - skills: hand-written vs LLM-initial vs LLM-refined
tables/refinement_curve.csv         - one row per (env, refinement iteration)
tables/wall_clock.csv               - mean training wall-clock seconds per condition

plots/env_reward_comparison.png
plots/success_rate_comparison.png
plots/refinement_curves.png
plots/skill_usage_<env>.png          (one per environment)
plots/seed_variance_env_reward.png

report_tables.md                     - all tables above, rendered as Markdown
                                        (this is what README.md embeds)
 
Usage:
    python analyze_llm_results.py --root results_llm --out analysis_output
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
 
CONDITIONS = ("hand_written", "llm_initial", "llm_refined")
CONDITION_LABELS = {
    "hand_written": "Hand-written",
    "llm_initial": "LLM (initial)",
    "llm_refined": "LLM (refined)",
}
CONDITION_COLORS = {
    "hand_written": "#1A65DE",
    "llm_initial": "#C5210F",
    "llm_refined": "#149A34",
}


def discover_env_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"root directory not found: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("results_"))
    if not dirs:
        if (root / "manifest.json").exists() or any(root.glob("*_summary.json")):
            return [root]
        raise FileNotFoundError(f"no results_<env> subdirectories found under {root}")
    return dirs
 
 
def load_env_data(env_dir: Path) -> tuple[str, dict[str, Any]]:
    """Return (env_name, per-env dict) for one results_<env> folder."""
    
    manifest_path = env_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            envs = manifest.get("envs", {})
            if envs:
                (env_name, data), = envs.items() if len(envs) == 1 else list(envs.items())[:1]
                return env_name, data
        except (json.JSONDecodeError, ValueError):
            pass

    summary_candidates = list(env_dir.glob("*_summary.json"))
    if not summary_candidates:
        raise FileNotFoundError(f"no manifest.json or *_summary.json in {env_dir}")
    summary_path = summary_candidates[0]
    env_name = summary_path.name[: -len("_summary.json")]
    data = json.loads(summary_path.read_text())
    return env_name, data
 
 
def load_all(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for env_dir in discover_env_dirs(root):
        env_name, data = load_env_data(env_dir)
        out[env_name] = data
    return out
 
 
# Extraction
def first_backend(env_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    backends = env_data.get("backends", {})
    if not backends:
        raise ValueError("env data has no 'backends' entry")
    name = sorted(backends.keys())[0]
    return name, backends[name]
 
 
def skill_usage_from_run(run: dict[str, Any]) -> dict[str, float]:
    metrics = run.get("metrics", {})
    return {
        k.split("/", 1)[1]: float(v)
        for k, v in metrics.items()
        if k.startswith("skill_usage/")
    }
 
def mean_skill_usage(runs: list[dict[str, Any]]) -> dict[str, float]:
    per_run = [skill_usage_from_run(r) for r in runs]
    names = sorted({n for d in per_run for n in d})
    return {n: float(np.mean([d.get(n, 0.0) for d in per_run])) for n in names}
 
 
def per_seed_rows(env: str, condition: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in runs:
        eval_m = r.get("eval_metrics", {})
        metrics = r.get("metrics", {})
        rows.append(
            {
                "env": env,
                "condition": condition,
                "seed": r.get("seed"),
                "env_reward_mean": metrics.get("returns/env_reward_mean"),
                "skill_reward_mean": metrics.get("returns/skill_reward_mean"),
                "eval_episode_return_mean": eval_m.get("episode_return_mean"),
                "eval_success_rate": eval_m.get("primary_success_rate"),
                "eval_goal_metric": eval_m.get("primary_goal_metric"),
                "mask_violation_rate": metrics.get("mask/violation_rate"),
                "wall_s": r.get("wall_s"),
            }
        )
    return rows

# Tables 
def build_env_comparison_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        hw_summary = data["hand_written"]["summary"]
        backend_name, backend = first_backend(data)
        llm_summary = backend["llm"]["summary"]
        curve = backend["refinement"]["curve"]
        final_iter = curve[-1] if curve else None
        final_metrics = final_iter["metrics"] if final_iter else {}
 
        n_skills_hand = len(data["hand_written"]["runs"][0]["metrics"].keys() and mean_skill_usage(data["hand_written"]["runs"]))
        n_skills_llm_initial = len(backend["llm"]["skillset"]["skills"])
        n_skills_llm_refined = len(backend["refinement"]["final_skillset"]["skills"])
 
        rows.append(
            {
                "env": env,
                "backend": backend_name,
                "hand_env_reward_mean": hw_summary.get("env_reward_mean"),
                "hand_env_reward_std": hw_summary.get("env_reward_std"),
                "hand_success_rate_mean": hw_summary.get("success_rate_mean"),
                "hand_success_rate_std": hw_summary.get("success_rate_std"),
                "hand_n_skills": n_skills_hand,
                "llm_env_reward_mean": llm_summary.get("env_reward_mean"),
                "llm_env_reward_std": llm_summary.get("env_reward_std"),
                "llm_success_rate_mean": llm_summary.get("success_rate_mean"),
                "llm_success_rate_std": llm_summary.get("success_rate_std"),
                "llm_n_skills": n_skills_llm_initial,
                "refined_env_reward_mean": final_metrics.get("returns/env_reward_mean"),
                "refined_success_rate_mean": final_metrics.get("policy_diag/primary_success_rate"),
                "refined_n_skills": n_skills_llm_refined,
                "refinement_iterations": len(curve),
                "refinement_stopped_early": backend["refinement"].get("stopped_early"),
                "llm_vs_hand_reward_gap_pct": (
                    100.0 * (llm_summary.get("env_reward_mean", 0.0) - hw_summary.get("env_reward_mean", 0.0))
                    / abs(hw_summary["env_reward_mean"])
                    if hw_summary.get("env_reward_mean") else None
                ),
            }
        )
    return rows
 
 
def build_per_seed_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        rows += per_seed_rows(env, "hand_written", data["hand_written"]["runs"])
        _, backend = first_backend(data)
        rows += per_seed_rows(env, "llm_initial", backend["llm"]["runs"])
    return rows
 
 
def build_skill_usage_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        usage = mean_skill_usage(data["hand_written"]["runs"])
        for skill, frac in sorted(usage.items(), key=lambda kv: -kv[1]):
            rows.append({"env": env, "condition": "hand_written", "skill": skill, "mean_usage_fraction": round(frac, 4)})
    return rows

def build_llm_skill_usage_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        _, backend = first_backend(data)
        usage = mean_skill_usage(backend["llm"]["runs"])
        for skill, frac in sorted(usage.items(), key=lambda kv: -kv[1]):
            rows.append({"env": env, "condition": "llm_initial", "skill": skill, "mean_usage_fraction": round(frac, 4)})
    return rows 

def build_refined_skill_usage_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:

    rows = []
    for env, data in all_data.items():
        _, backend = first_backend(data)
        curve = backend["refinement"]["curve"]
        if not curve:
            continue
        final_metrics = curve[-1]["metrics"]
        usage = {
            k.split("/", 1)[1]: float(v)
            for k, v in final_metrics.items()
            if k.startswith("skill_usage/")
        }
        for skill, frac in sorted(usage.items(), key=lambda kv: -kv[1]):
            rows.append({"env": env, "condition": "llm_refined", "skill": skill, "mean_usage_fraction": round(frac, 4)})
    return rows
 
def build_skillset_size_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        hand_names = sorted(mean_skill_usage(data["hand_written"]["runs"]).keys())
        _, backend = first_backend(data)
        llm_names = [s["name"] for s in backend["llm"]["skillset"]["skills"]]
        refined_names = [s["name"] for s in backend["refinement"]["final_skillset"]["skills"]]
        rows.append(
            {
                "env": env,
                "hand_written_skills": len(hand_names),
                "hand_written_names": "; ".join(hand_names),
                "llm_initial_skills": len(llm_names),
                "llm_initial_names": "; ".join(llm_names),
                "llm_refined_skills": len(refined_names),
                "llm_refined_names": "; ".join(refined_names),
            }
        )
    return rows
 
 
def build_refinement_curve_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        _, backend = first_backend(data)
        for c in backend["refinement"]["curve"]:
            m = c["metrics"]
            rows.append(
                {
                    "env": env,
                    "iteration": c["iteration"],
                    "n_skills": len(c["skillset"]["skills"]),
                    "env_reward_mean": m.get("returns/env_reward_mean"),
                    "skill_reward_mean": m.get("returns/skill_reward_mean"),
                    "primary_success_rate": m.get("policy_diag/primary_success_rate"),
                    "mask_violation_rate": m.get("mask/violation_rate"),
                    "refinement_ok": c.get("refinement_ok"),
                    "refinement_error": c.get("refinement_error"),
                }
            )
    return rows
 
 
def build_wall_clock_table(all_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for env, data in all_data.items():
        hand_wall = [r.get("wall_s") for r in data["hand_written"]["runs"] if r.get("wall_s") is not None]
        _, backend = first_backend(data)
        llm_wall = [r.get("wall_s") for r in backend["llm"]["runs"] if r.get("wall_s") is not None]
        rows.append(
            {
                "env": env,
                "hand_written_wall_s_mean": round(float(np.mean(hand_wall)), 2) if hand_wall else None,
                "llm_initial_wall_s_mean": round(float(np.mean(llm_wall)), 2) if llm_wall else None,
            }
        )
    return rows


# CSV
def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
 
def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
 
 
def rows_to_markdown(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_no data_\n"
    cols = list(rows[0].keys())
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# Plots 
def plot_env_reward_comparison(env_table: list[dict[str, Any]], out_path: Path) -> None:
    envs = [r["env"] for r in env_table]
    hand = [r["hand_env_reward_mean"] for r in env_table]
    hand_std = [r["hand_env_reward_std"] or 0 for r in env_table]
    llm = [r["llm_env_reward_mean"] for r in env_table]
    llm_std = [r["llm_env_reward_std"] or 0 for r in env_table]
    refined = [r["refined_env_reward_mean"] for r in env_table]
 
    x = np.arange(len(envs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, len(envs) * 1.8), 5))
    ax.bar(x - width, hand, width, yerr=hand_std, capsize=4, label=CONDITION_LABELS["hand_written"],
           color=CONDITION_COLORS["hand_written"])
    ax.bar(x, llm, width, yerr=llm_std, capsize=4, label=CONDITION_LABELS["llm_initial"],
           color=CONDITION_COLORS["llm_initial"])
    ax.bar(x + width, refined, width, label=CONDITION_LABELS["llm_refined"],
           color=CONDITION_COLORS["llm_refined"])
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("Env reward mean (per-step)")
    ax.set_title("Environment reward: hand-written vs LLM (initial) vs LLM (refined)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def plot_success_rate_comparison(env_table: list[dict[str, Any]], out_path: Path) -> None:
    envs = [r["env"] for r in env_table]
    hand = [r["hand_success_rate_mean"] for r in env_table]
    hand_std = [r["hand_success_rate_std"] or 0 for r in env_table]
    llm = [r["llm_success_rate_mean"] for r in env_table]
    llm_std = [r["llm_success_rate_std"] or 0 for r in env_table]
    refined = [r["refined_success_rate_mean"] for r in env_table]
 
    x = np.arange(len(envs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, len(envs) * 1.8), 5))
    ax.bar(x - width, hand, width, yerr=hand_std, capsize=4, label=CONDITION_LABELS["hand_written"],
           color=CONDITION_COLORS["hand_written"])
    ax.bar(x, llm, width, yerr=llm_std, capsize=4, label=CONDITION_LABELS["llm_initial"],
           color=CONDITION_COLORS["llm_initial"])
    ax.bar(x + width, refined, width, label=CONDITION_LABELS["llm_refined"],
           color=CONDITION_COLORS["llm_refined"])
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("Primary success rate (deterministic eval)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task success rate: hand-written vs LLM (initial) vs LLM (refined)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def plot_refinement_curves(curve_table: list[dict[str, Any]], envs: list[str], out_path: Path) -> None:
    
    n = len(envs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    for idx, env in enumerate(envs):
        ax = axes[idx // ncols][idx % ncols]
        rows = [r for r in curve_table if r["env"] == env]
        rows.sort(key=lambda r: r["iteration"])
        iters = [r["iteration"] for r in rows]
        reward = [r["env_reward_mean"] for r in rows]
        n_skills = [r["n_skills"] for r in rows]
 
        ax.plot(iters, reward, marker="o", color="#4C72B0", label="env_reward_mean")
        ax.set_title(env)
        ax.set_xlabel("Refinement iteration")
        ax.set_ylabel("env_reward_mean", color="#4C72B0")
        ax.tick_params(axis="y", labelcolor="#4C72B0")
        ax.spines["top"].set_visible(False)
 
        ax2 = ax.twinx()
        ax2.plot(iters, n_skills, marker="s", linestyle="--", color="#55A868", label="# skills")
        ax2.set_ylabel("# skills", color="#55A868")
        ax2.tick_params(axis="y", labelcolor="#55A868")
        ax2.set_ylim(0, max(n_skills + [1]) + 1)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
 
    fig.suptitle("LLM refinement loop: reward and skillset size across iterations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def plot_skill_usage(env: str, usage: dict[str, float], out_path: Path) -> None:
    names = list(usage.keys())
    values = [usage[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(names, values, color="#4C72B0")
    ax.set_xlabel("Mean fraction of rollout steps selected")
    ax.set_xlim(0, 1)
    ax.set_title(f"{env}: hand-written skill usage (mean over seeds)")
    for b, v in zip(bars, values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
def plot_skill_usage_three_way(env: str, data: dict[str, Any], out_path: Path) -> None:
    """One figure, three horizontal-bar panels: hand-written / LLM-initial / LLM-refined."""
    _, backend = first_backend(data)
    hand_usage = mean_skill_usage(data["hand_written"]["runs"])
    llm_usage = mean_skill_usage(backend["llm"]["runs"])
    curve = backend["refinement"]["curve"]
    refined_usage = {}
    if curve:
        refined_usage = {
            k.split("/", 1)[1]: float(v)
            for k, v in curve[-1]["metrics"].items()
            if k.startswith("skill_usage/")
        }
 
    panels = [
        ("Hand-written", hand_usage, CONDITION_COLORS["hand_written"]),
        ("LLM (initial)", llm_usage, CONDITION_COLORS["llm_initial"]),
        ("LLM (refined)", refined_usage, CONDITION_COLORS["llm_refined"]),
    ]
    max_names = max(len(u) for _, u, _ in panels) or 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.5 * max_names + 1.8))
    for ax, (title, usage, color) in zip(axes, panels):
        names = list(usage.keys())
        values = [usage[n] for n in names]
        bars = ax.barh(names, values, color=color)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("usage fraction")
        for b, v in zip(bars, values):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{env}: skill usage across conditions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def plot_seed_variance(per_seed_table: list[dict[str, Any]], envs: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(envs) * 1.8), 5))
    x = np.arange(len(envs))
    width = 0.35
    for offset, cond in ((-width / 2, "hand_written"), (width / 2, "llm_initial")):
        means, mins, maxs = [], [], []
        for env in envs:
            vals = [
                r["env_reward_mean"]
                for r in per_seed_table
                if r["env"] == env and r["condition"] == cond and r["env_reward_mean"] is not None
            ]
            if not vals:
                means.append(0.0)
                mins.append(0.0)
                maxs.append(0.0)
                continue
            means.append(float(np.mean(vals)))
            mins.append(float(np.min(vals)))
            maxs.append(float(np.max(vals)))
        means = np.array(means)
        lower = means - np.array(mins)
        upper = np.array(maxs) - means
        ax.bar(x + offset, means, width, yerr=[lower, upper], capsize=4,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond])
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=20, ha="right")
    ax.set_ylabel("env_reward_mean (bar=seed mean, whiskers=seed min/max)")
    ax.set_title("Across-seed variance: hand-written vs LLM (initial)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default="results_llm", help="Path to the results_llm folder")
    parser.add_argument("--out", type=str, default="analysis_output", help="Output directory")
    args = parser.parse_args()
 
    root = Path(args.root)
    out = Path(args.out)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
 
    all_data = load_all(root)
    envs = sorted(all_data.keys())
    print(f"Loaded {len(envs)} environment(s): {', '.join(envs)}")
 
    env_table = build_env_comparison_table(all_data)
    per_seed_table = build_per_seed_table(all_data)
    skill_usage_table = (build_skill_usage_table(all_data) + build_llm_skill_usage_table(all_data) + build_refined_skill_usage_table(all_data))
    skillset_size_table = build_skillset_size_table(all_data)
    curve_table = build_refinement_curve_table(all_data)
    wall_table = build_wall_clock_table(all_data)
 
    write_csv(env_table, tables_dir / "env_comparison_summary.csv")
    write_csv(per_seed_table, tables_dir / "per_seed_results.csv")
    write_csv(skill_usage_table, tables_dir / "skill_usage_all_conditions.csv")
    write_csv(skillset_size_table, tables_dir / "skillset_sizes.csv")
    write_csv(curve_table, tables_dir / "refinement_curve.csv")
    write_csv(wall_table, tables_dir / "wall_clock.csv")
    print(f"Wrote CSV tables to {tables_dir}/")
 
    md = []
    md.append(rows_to_markdown(env_table, "Environment comparison summary (hand-written vs LLM vs refined)"))
    md.append(rows_to_markdown(skillset_size_table, "Skillset sizes and names"))
    md.append(rows_to_markdown(
        skill_usage_table,
        "Skill usage by condition (hand-written / LLM-initial / LLM-refined; "
        "hand-written and LLM-initial are averaged over 5 seeds, "
        "LLM-refined is the single final-iteration training run)",
    ))
    md.append(rows_to_markdown(curve_table, "LLM refinement loop, iteration by iteration"))
    md.append(rows_to_markdown(wall_table, "Mean training wall-clock time (seconds)"))
    md.append(rows_to_markdown(per_seed_table, "Per-seed raw results"))
    (out / "report_tables.md").write_text("\n\n".join(md))
    print(f"Wrote {out / 'report_tables.md'}")

    plot_env_reward_comparison(env_table, plots_dir / "env_reward_comparison.png")
    plot_success_rate_comparison(env_table, plots_dir / "success_rate_comparison.png")
    plot_refinement_curves(curve_table, envs, plots_dir / "refinement_curves.png")
    plot_seed_variance(per_seed_table, envs, plots_dir / "seed_variance_env_reward.png")
    for env, data in all_data.items():
        safe_env = env.replace("/", "_")
        plot_skill_usage_three_way(env, data, plots_dir / f"skill_usage_{safe_env}.png")
    print(f"Wrote plots to {plots_dir}/")
 
    print("\nDone. Summary:")
    for r in env_table:
        print(
            f"  {r['env']:<24} hand={r['hand_env_reward_mean']:.3f} "
            f"llm={r['llm_env_reward_mean']:.3f} refined={r['refined_env_reward_mean']:.3f} "
            f"(gap vs hand: {r['llm_vs_hand_reward_gap_pct']:.1f}%)"
        )
 
 
if __name__ == "__main__":
    main()