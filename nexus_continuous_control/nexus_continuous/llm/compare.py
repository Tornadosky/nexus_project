"""
Compare manually-designed vs LLM-generated policies.
"""

from __future__ import annotations
import csv
import numpy as np

from nexus_continuous.llm.experiment import run_llm_experiment

def summarize_seed_metrics(results):
    returns = []
    
    for r in results:
        returns.append(float(r["returns/env_reward_mean"]))
        
    return{
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }

def compare_policies(
    env_name,
    train_handwritten_fn,
    train_llm_fn,
    num_seeds = 3,
):
    hand_runs = []
    llm_runs = []
    
    for seed in range(num_seeds):
        hand_runs.append(train_handwritten_fn(env_name, seed))
    for seed in range(num_seeds):
        llm_runs.append(train_llm_fn(env_name, seed))
    
    hand_stats = summarize_seed_metrics(hand_runs)
    llm_stats = summarize_seed_metrics(llm_runs)
    
    row = {
        "env": env_name,
        "hand_mean": hand_stats["mean"],
        "hand_std": hand_stats["std"],
        "llm_mean": llm_stats["mean"],
        "llm_std": llm_stats["std"],
    }
    
    with open("comparison.csv", "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    
    return row