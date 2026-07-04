"""    
Run LLM-generated skill experiments using the standard NEXUS training pipeline.

1. Loads a config
2. Injects LLM-generates skill modules (via interpreter)
3. Runs hierarchical training (as hand-written baselines)
4. Saves outputs for comparison
"""

from __future__ import annotations 
import argparse 
import json
import jax
from pathlib import Path 

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.utils import load_config
from nexus_continuous.llm.interpreter import make_policy_module


def load_llm_skillset(path: str):
    """ Load LLM-generated JSON skill set."""
    with open(path, "r") as f:
        return json.load(f)

def patch_config_with_llm(cfg: dict, llm_skillset: dict) -> dict:
    """ 
    Inject LLM skill module into config so that:
    load_policy_module(cfg["POLICY"]) picks up interpreted skills.
    """
    
    cfg = dict(cfg)
    cfg["LLM_SKILLSET"] = llm_skillset
    cfg["USE_LLM_SKILLS"] = True 
    
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    parser.add_argument("--llm-skills", type = str, required = True)
    parser.add_argument("--save", type = str, default = "llm_run.pkl")
    parser.add_argument("--seed", type = int, default = 0)
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    llm_skillset = load_llm_skillset(args.llm_skills)
    cfg = patch_config_with_llm(cfg, llm_skillset)
    
    print("\n=== Running LLM Skill Experiment ===")
    print(f"Env: {cfg['ENV_NAME']}")
    print(f"Skills file: {args.llm_skills}")
    print(f"Seed: {args.seed}")
    
    rng = jax.random.PRNGKey(args.seed)
    output = run_training(cfg)
    print("\n=== Training Finished ===")
    
    Path(args.save).parent.mkdir(parents = True, exist_ok = True)
    
    import pickle
    with open(args.save, "wb") as f:
        pickle.dump(
            {
                "config": cfg,
                "metrics": output.metrics,
                "eval_metrics": output.eval_metrics,
                "eval_table": output.eval_episode_table
            },
            f,
        )
    print(f"Saved to: {args.save}")

if __name__ == "__main__":
    main()