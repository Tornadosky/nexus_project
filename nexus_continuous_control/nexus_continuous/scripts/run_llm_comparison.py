"""
Compare handwritten NEXUS vs LLM generated NEXUS.

Usage:

python scripts/run_llm_comparison.py \
    --env WalkerWalk \
    --config configs/walker.yaml \
    --seeds 3
"""


from __future__ import annotations
import argparse
import json
from typing import Any
import numpy as np

from nexus_continuous.envs.env_registry import ENV_REGISTRY
from nexus_continuous.llm.pipeline import generate_skillset
from nexus_continuous.llm.client import LLMClient

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.utils import load_config
from nexus_continuous.policies.registry import canonicalize_policy_name

def train_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    result = run_training(cfg)
    return result.metrics

def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(r["returns/env_reward_mean"]) for r in results]
    return {
        "mean":float(np.mean(values)),
        "std":float(np.std(values)),
        "min":float(np.min(values)),
        "max":float(np.max(values))
    }

def generate_llm_policy(env: str, client: LLMClient | None = None):
    meta = ENV_REGISTRY[env]
    skillset = generate_skillset(
        env_name=env,
        observation_schema="\n".join(meta["fields"]),
        task_description=meta["task"],
        client = client
    )

    return skillset

def compare_policies(
    env_name: str, 
    config: str,
    num_seeds: int = 3,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """ Train the hand-written policy and an LLM-generated policy. 
    """
    base_cfg = load_config(config)
    obs_fields = tuple(ENV_REGISTRY[env_name]["fields"])
        
    hand=[]
    for seed in range(num_seeds):
        cfg=dict(base_cfg)
        cfg["ENV_NAME"] = env_name
        cfg["SEED"] = seed
        cfg["POLICY_MODULE"] = canonicalize_policy_name(env_name)

        hand.append(train_policy(cfg))

    llm=[]
    skillset = generate_llm_policy(env_name, client = client)
    for seed in range(num_seeds):
        cfg=dict(base_cfg)
        cfg["ENV_NAME"] = env_name
        cfg["SEED"] = seed
        cfg["USE_LLM_SKILLS"] = True
        cfg["LLM_SKILLSET"] = skillset.__dict__
        cfg["OBS_FIELDS"] = obs_fields

        llm.append(train_policy(cfg))
    
    hand_summary = summarize(hand)
    llm_summary = summarize(llm)
    return{
        "env": env_name,
        "hand_mean": hand_summary["mean"],
        "hand_std": hand_summary["std"],
        "llm_mean": llm_summary["mean"],
        "llm_std": llm_summary["std"],
        "handwritten": hand_summary,
        "llm": llm_summary,
        "skillset": skillset
    }
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args=parser.parse_args()
    
    result = compare_policies(args.env, args.config, num_seeds = args.seeds)

    print("\n========== RESULTS ==========")
    print("Handwritten:")
    print(result["handwritten"])
    print("\nLLM:")
    print(result["llm"])

    with open(f"{args.env}_comparison.json","w") as f:
        json.dump(
            {
                "environment": args.env,
                "handwritten": result["handwritten"],
                "llm": result["llm"],
                "skills": result["skillset"].__dict__
            },
            f,
            indent=2,
        )

if __name__=="__main__":
    main()