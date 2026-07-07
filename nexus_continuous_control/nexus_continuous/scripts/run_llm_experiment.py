"""
Run complete NEXUS LLM skill experiment.

Pipeline:

Environment name
        |
        v
ENV_REGISTRY
        |
        v
LLM skill generation
        |
        v
schema validation
        |
        v
interpreter -> JAX policy module
        |
        v
AC-PQN training
        |
        v
save checkpoint + skill json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle

from nexus_continuous.envs.env_registry import ENV_REGISTRY
from nexus_continuous.llm.pipeline import generate_skillset, save_skillset
from nexus_continuous.llm.client import LLMClient

from nexus_continuous.llm.interpreter import make_policy_module
from nexus_continuous.utils import load_config
from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training

def get_env_metadata(env_name):
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"""
            Unknown environment {env_name}
            Available: {list(ENV_REGISTRY.keys())}
            """
        )
    return ENV_REGISTRY[env_name]


def generate_llm_skills(env_name):
    meta = get_env_metadata(env_name)
    fields = "\n".join(meta["fields"])
    
    print("\nGenerating LLM skills...")
    print("Environment:", env_name)
    print("Fields:")
    print(fields)

    skillset = generate_skillset(
        env_name = env_name,
        observation_schema = fields,
        task_description = meta["task"],
    )

    return skillset


def build_llm_config(base_cfg, skillset):
    cfg = dict(base_cfg)
    cfg["USE_LLM_SKILLS"] = True
    cfg["LLM_SKILLSET"] = skillset.__dict__
    cfg["POLICY_MODULE"] = make_policy_module(skillset.__dict__)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Environment name from ENV_REGISTRY")
    parser.add_argument("--config", required=True, help="Training config yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    out = Path(args.output)
    out.mkdir(parents = True, exist_ok = True)
    
    skillset = generate_llm_skills(args.env)
    skill_path = (out/f"{args.env}_llm_skills.json")
    save_skillset(skillset, str(skill_path))
    print("Saved skills:", skill_path)
    
    cfg = load_config(args.config)
    cfg["ENV_NAME"] = args.env 
    cfg["SEED"] = args.seed 
    
    cfg = build_llm_config(cfg, skillset)
    
    print("\n=== Training LLM NEXUS policy ===")
    result = run_training(cfg)
    save_file = (out/f"{args.env}_llm_results.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "config": cfg,
                "skills": skillset,
                "metrics": result.metrics,
                "eval_metrics": result.eval_metrics
            }, 
            f
        )
    print("\nSaved:", save_file)
    
    if __name__ == "__main__":
        main()

