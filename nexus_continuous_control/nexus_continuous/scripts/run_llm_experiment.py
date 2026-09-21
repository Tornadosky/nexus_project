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
from typing import Any
from pathlib import Path
import pickle

from nexus_continuous.envs.env_registry import ENV_REGISTRY
from nexus_continuous.llm.pipeline import generate_skillset, save_skillset
from nexus_continuous.llm.client import LLMClient

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


def generate_llm_skills(env_name: str, client: LLMClient | None = None):
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
        client = client
    )

    return skillset


def build_llm_config(base_cfg: dict[str, Any], env_name: str, skillset) -> dict[str, Any]:
    """ Train a base training config into one that trains the LLM generated skillset. 
    """
    cfg = dict(base_cfg)
    cfg["USE_LLM_SKILLS"] = True
    cfg["LLM_SKILLSET"] = skillset.__dict__ if hasattr(skillset, "__dict__") else skillset
    cfg["OBS_FIELDS"] = tuple(get_env_metadata(env_name)["fields"])
    return cfg

def run_llm_experiment(
    env_name: str,
    config: str,
    seed: int = 0,
    output: str = "results",
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """ Generate a LLM skillset for env_name, train and save results.
    
        Returns a dict with the generated skillset, training metrics and eval metrics.
    """
    out = Path(output)
    out.mkdir(parents = True, exist_ok = True)
    
    skillset = generate_llm_skills(env_name, client = client)
    skill_path = out/f"{env_name}_llm_skills.json"
    save_skillset(skillset, str(skill_path))
    print("Saved skills:", skill_path) 
    
    cfg = load_config(config)
    cfg["ENV_NAME"] = env_name
    cfg["SEED"] = seed 
    cfg = build_llm_config(cfg, env_name, skillset)
    
    print("\n Training LLM NEXUS policy")
    result = run_training(cfg)
    
    save_file = out/f"{env_name}_llm_results.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(
            {
                "config": cfg,
                "skills": skillset,
                "metrics": result.metrics,
                "eval_metrics": result.eval_metrics,
            },
            f,
        )
    print("\nSaved:", save_file)
    
    return{
        "skillset": skillset, 
        "metrics": result.metrics,
        "eval_metrics": result.eval_metrics,
    }

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--env", required=True, help="Environment name from ENV_REGISTRY")
    parser.add_argument("--config", required=True, help="Training config yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    
    run_llm_experiment(
        env_name = args.env,
        config = args.config,
        seed = args.seed, 
        output = args.output
    )
    
if __name__ == "__main__":
    main()

