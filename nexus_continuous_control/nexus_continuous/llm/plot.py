""" Plotting utilities for the LLM extension."""

from __future__ import annotations 
from typing import Any, Dict, Sequence 
import matplotlib 

matplotlib.use("Agg")
import matplotlib.pyplot as plt 

def plot_comparison(hand_summary: Dict[str, float], llm_summary: Dict[str, float], env_name: str, out_path: str) -> str:
    """Bar chart: handwritten vs. LLM env-reward mean +/- std.
    """
    labels = ["Handwritten", "NEXUS (LLM)"]
    means = [hand_summary["mean"], llm_summary["mean"]]
    stds = [hand_summary["std"], llm_summary["std"]]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    ax.set_ylabel("Env reward (mean over seeds)")
    ax.set_title(f"Handwritten vs. LLM-generated skills -- {env_name}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_refinement(
    history: Sequence[Any],
    out_path: str,
    metric_key: str = "returns/env_reward_mean",
    title: str = "LLM refinement loop",
) -> str:
    """Line chart of ``metric_key`` (and skillset size) across refinement iterations. 
    """
    iterations, values, n_skills = [], [], []
    for rec in history:
        metrics = rec.metrics if hasattr(rec, "metrics") else rec["metrics"]
        skillset = rec.skillset if hasattr(rec, "skillset") else rec["skillset"]
        iterations.append(rec.iteration if hasattr(rec, "iteration") else rec["iteration"])
        values.append(float(metrics.get(metric_key, float("nan"))))
        skills = skillset.get("skills") if isinstance(skillset, dict) else getattr(skillset, "skills", [])
        n_skills.append(len(skills) if skills is not None else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.plot(iterations, values, marker="o", color="#4C72B0")
    ax1.set_xlabel("Refinement iteration")
    ax1.set_ylabel(metric_key)
    ax1.set_title(title)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(iterations, n_skills, marker="s", color="#55A868")
    ax2.set_xlabel("Refinement iteration")
    ax2.set_ylabel("# skills in proposal")
    ax2.set_title("Skillset size over iterations")
    ax2.set_ylim(0, max(n_skills + [1]) + 1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_multi_seed_refinement(
    all_histories: Dict[int, Sequence[Any]],
    out_path: str,
    metric_key: str = "returns/env_reward_mean",
    title: str = "LLM refinement loop across seeds",
) -> str:
    """Overlay ``metric_key`` across iterations for several seeded runs.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for seed, history in sorted(all_histories.items()):
        iterations = [rec.iteration for rec in history]
        values = [float(rec.metrics.get(metric_key, float("nan"))) for rec in history]
        ax.plot(iterations, values, marker="o", label=f"seed {seed}")
        
    ax.set_xlabel("Refinement iteration")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
