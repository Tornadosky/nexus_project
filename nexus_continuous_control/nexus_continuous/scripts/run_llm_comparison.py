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
import numpy as np

from nexus_continuous.envs.env_registry import ENV_REGISTRY
from nexus_continuous.llm.pipeline import generate_skillset
from nexus_continuous.llm.interpreter import make_policy_module


from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.utils import load_config
from nexus_continuous.policies.registry import load_policy_module


def train_policy(cfg):
    result = run_training(cfg)
    return result.metrics

def summarize(results):
    values = [float(r["returns/env_reward_mean"]) for r in results]

    return {
        "mean":
            float(np.mean(values)),
        "std":
            float(np.std(values)),
        "min":
            float(np.min(values)),
        "max":
            float(np.max(values))
    }

def generate_llm_policy(env):
    meta = ENV_REGISTRY[env]

    skillset = generate_skillset(
        env_name=env,
        observation_schema="\n".join(meta["fields"]),
        task_description=meta["task"]
    )

    return skillset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args=parser.parse_args()


    base_cfg = load_config(args.config)
    
    hand=[]
    for seed in range(args.seeds):
        cfg=dict(base_cfg)
        cfg["ENV_NAME"]=args.env
        cfg["SEED"]=seed
        cfg["POLICY_MODULE"] = (load_policy_module(args.env))

        hand.append(train_policy(cfg))

    llm=[]
    skillset = generate_llm_policy(args.env)
    for seed in range(args.seeds):
        cfg=dict(base_cfg)
        cfg["ENV_NAME"]=args.env
        cfg["SEED"]=seed
        cfg["USE_LLM_SKILLS"]=True
        cfg["LLM_SKILLSET"]=(skillset.__dict__)
        cfg["POLICY_MODULE"]=(make_policy_module(skillset.__dict__))

        llm.append(train_policy(cfg))


    print("\n========== RESULTS ==========")
    print("Handwritten:")
    print(summarize(hand))

    print("\nLLM:")
    print(summarize(llm))

    with open(f"{args.env}_comparison.json","w") as f:
        json.dump(
            {
                "environment":args.env,
                "handwritten":summarize(hand),
                "llm":summarize(llm),
                "skills":skillset.__dict__
            },
            f,
            indent=2
        )

if __name__=="__main__":
    main()