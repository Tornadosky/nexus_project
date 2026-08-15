""" Aggregate LLM extension results into a report.

Reads the minofest.json produced by run_llm_full_suite.py and
writes under the same results directory:

- summary.csv -> env, backend, kind, metric, mean, std 
- <env>_comparison.png -> hand-written vs LLM (per backend) vs refined chart
- <env>_refinement.png -> refinement loop progress curve, one line per backend
- REPORT.md -> Results, plots and comments

"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
_PALETTE = {"hand_written": "#4C72B0", "hf": "#DD8452", "vertex": "#55A868",
            "gemini": "#55A868", "openai": "#8172B2", "mock": "#937860"}


def load_manifest(results_dir: str) -> dict[str, Any]:
    path = Path(results_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run run_llm_full_suite.py first, or pass "
            "--results pointing at its output directory."
        )
    with open(path) as f:
        return json.load(f)
    
 
def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def build_table(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for env_name, env_entry in manifest["envs"].items():
        hand = env_entry["hand_written"]["summary"]
        
        for metric in ("env_reward", "success_rate", "goal_metric"):
            if f"{metric}_mean" in hand:
                rows.append({"env": env_name, "backend": "-", "kind": "hand_written",
                             "metric": metric, "mean": hand[f"{metric}_mean"],
                             "std": hand.get(f"{metric}_std")})
        
        for backend, b in env_entry["backends"].items():
            llm = b["llm"]["summary"]
            
            for metric in ("env_reward", "success_rate", "goal_metric"):
                if f"{metric}_mean" in llm:
                    rows.append({"env": env_name, "backend": backend, "kind": "llm",
                                 "metric": metric, "mean": llm[f"{metric}_mean"],
                                 "std": llm.get(f"{metric}_std")})
            curve = b["refinement"]["curve"]
            
            if curve:
                first, last = curve[0]["metrics"], curve[-1]["metrics"]
                
                for label, m in (("refined_first_iter", first), ("refined_last_iter", last)):
                    v = m.get("returns/env_reward_mean")
                    if v is not None:
                        rows.append({"env": env_name, "backend": backend, "kind": label,
                                     "metric": "env_reward", "mean": v, "std": None})
    return rows
 
 
def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "backend", "kind", "metric", "mean", "std"])
        w.writeheader()
        w.writerows(rows)


def plot_env_comparison(env_name: str, env_entry: dict[str, Any], out_path: Path) -> Path:
    labels, means, stds, colors = [], [], [], []
 
    hand = env_entry["hand_written"]["summary"]
    labels.append("Hand-written")
    means.append(hand.get("env_reward_mean"))
    stds.append(hand.get("env_reward_std", 0.0))
    colors.append(_PALETTE["hand_written"])
 
    for backend, b in env_entry["backends"].items():
        llm = b["llm"]["summary"]
        labels.append(f"LLM ({backend})")
        means.append(llm.get("env_reward_mean"))
        stds.append(llm.get("env_reward_std", 0.0))
        colors.append(_PALETTE.get(backend, "#8172B2"))
 
        curve = b["refinement"]["curve"]
        if curve:
            last_val = curve[-1]["metrics"].get("returns/env_reward_mean")
            labels.append(f"Refined ({backend})")
            means.append(last_val)
            stds.append(0.0)
            colors.append(_PALETTE.get(backend, "#8172B2"))
 
    means_plot = [m if m is not None else 0.0 for m in means]
    stds_plot = [s if s is not None else 0.0 for s in stds]
 
    fig, ax = plt.subplots(figsize=(1.7 * len(labels) + 1.5, 4.2))
    x = np.arange(len(labels))
    
    ax.bar(x, means_plot, yerr=stds_plot, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Env reward (mean over seeds)")
    ax.set_title(f"{env_name}: hand-written vs. LLM vs. refined")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    return out_path


def plot_refinement_curves(env_name: str, env_entry: dict[str, Any], out_path: Path) -> Path | None:
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    
    for backend, b in env_entry["backends"].items():
        curve = b["refinement"]["curve"]
        if not curve:
            continue
        xs = [c["iteration"] for c in curve]
        ys = [c["metrics"].get("returns/env_reward_mean") for c in curve]
        ax.plot(xs, ys, marker="o", label=backend, color=_PALETTE.get(backend, None))
        plotted = True
    
    if not plotted:
        plt.close(fig)
        return None
    
    ax.set_xlabel("Refinement iteration")
    ax.set_ylabel("returns/env_reward_mean")
    ax.set_title(f"{env_name}: refinement progress")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    return out_path
 
 
def _skill_usage_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Average skill_usage fractions across seeds -> a cheap 'behaviour'
    signal (which skills the policy actually relies on) without needing
    video rendering."""
    
    keys = set()
    for r in runs:
        keys.update(k for k in r.get("metrics", {}) if k.startswith("skill_usage/"))
    out = {}
    
    for k in sorted(keys):
        vals = [r["metrics"][k] for r in runs if k in r["metrics"]]
        if vals:
            out[k.split("/", 1)[1]] = float(np.mean(vals))
    
    return out


def _comment_for_env(env_name: str, env_entry: dict[str, Any]) -> list[str]:
    lines = []
    hand = env_entry["hand_written"]["summary"]
    hand_mean = hand.get("env_reward_mean")
    hand_sr = hand.get("success_rate_mean")
 
    for backend, b in env_entry["backends"].items():
        llm = b["llm"]["summary"]
        llm_mean = llm.get("env_reward_mean")
        
        if llm_mean is not None and hand_mean is not None:
            delta = llm_mean - hand_mean
            rel = (delta / abs(hand_mean) * 100) if hand_mean else float("nan")
            direction = "outperformed" if delta > 0 else "underperformed"
            lines.append(
                f"- **{backend}**: LLM-generated skills {direction} the hand-written "
                f"baseline by {_fmt(delta)} reward ({_fmt(rel, 1)}%)."
            )
        
        llm_sr = llm.get("success_rate_mean")
        if llm_sr is not None and hand_sr is not None:
            lines.append(
                f"  Success rate: hand-written {_fmt(hand_sr, 3)} vs. LLM {_fmt(llm_sr, 3)}."
            )
 
        curve = b["refinement"]["curve"]
        if len(curve) >= 2:
            first = curve[0]["metrics"].get("returns/env_reward_mean")
            last = curve[-1]["metrics"].get("returns/env_reward_mean")
            
            if first is not None and last is not None:
                trend = "improved" if last > first else "did not improve"
                stop_note = " Stopped early (an LLM call failed after retries)." \
                    if b["refinement"]["stopped_early"] else ""
                
                lines.append(
                    f"  Refinement loop {trend} the skillset over {len(curve)} iterations "
                    f"({_fmt(first)} -> {_fmt(last)}).{stop_note}"
                )
 
        usage = _skill_usage_summary(b["llm"]["runs"])
        if usage:
            top = max(usage, key=usage.get)
            lines.append(
                f"  Skill usage (LLM/{backend}, avg over seeds): "
                + ", ".join(f"{name}={_fmt(frac, 2)}" for name, frac in usage.items())
                + f" -- dominant skill: `{top}` ({_fmt(usage[top], 2)})."
            )
 
    hand_usage = _skill_usage_summary(env_entry["hand_written"]["runs"])
    if hand_usage:
        lines.append(
            "  Hand-written skill usage (avg over seeds): "
            + ", ".join(f"{name}={_fmt(frac, 2)}" for name, frac in hand_usage.items())
        )
    
    return lines

def write_report(manifest: dict[str, Any], out_dir: Path) -> Path:
    lines = [
        "# NEXUS LLM Extension - Results Report\n",
        f"Seeds: {manifest.get('seeds')}  |  Backends: {manifest.get('backends')}\n",
        "## Summary table\n",
        "| Env | Backend | Kind | Metric | Mean | Std |",
        "|---|---|---|---|---|---|",
    ]
    
    for row in build_table(manifest):
        std_str = _fmt(row["std"]) if row["std"] is not None else "-"
        lines.append(f"| {row['env']} | {row['backend']} | {row['kind']} | "
                      f"{row['metric']} | {_fmt(row['mean'])} | {std_str} |")
 
    for env_name, env_entry in manifest["envs"].items():
        lines.append(f"\n## {env_name}\n")
 
        cmp_path = out_dir / f"{env_name}_comparison.png"
        plot_env_comparison(env_name, env_entry, cmp_path)
        lines.append(f"![{env_name} comparison]({cmp_path.name})\n")
 
        ref_path = out_dir / f"{env_name}_refinement.png"
        if plot_refinement_curves(env_name, env_entry, ref_path):
            lines.append(f"![{env_name} refinement]({ref_path.name})\n")
 
        lines.append("**Comments:**\n")
        lines.extend(_comment_for_env(env_name, env_entry) or ["- (no comparable data)"])
 
    report_path = out_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    return report_path
 
 
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
 
    manifest = load_manifest(args.results)
    out_dir = Path(args.results)
 
    rows = build_table(manifest)
    write_csv(rows, out_dir / "summary.csv")
    report_path = write_report(manifest, out_dir)
 
    print("Wrote", out_dir / "summary.csv")
    print("Wrote", report_path)
 
 
if __name__ == "__main__":
    main()