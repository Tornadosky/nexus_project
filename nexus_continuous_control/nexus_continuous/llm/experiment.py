"""
Generate, compile and train an LLM-created policy
"""

from __future__ import annotations
import json
from pathlib import Path

from nexus_continuous.llm.pipeline import generate_skillset
from nexus_continuous.llm.env_registry import ENV_REGISTRY
from nexus_continuous.llm.interpreter import make_policy_module

def run_llm_experiment(
    env_name: str,
    train_fn, 
    save_dir: str = "llm_runs",
):
    meta = ENV_REGISTRY[env_name]
    observation_schema = "\n".join(meta["fields"])
    
    skillset = generate_skillset(
        env_name = env_name,
        observation_schema = observation_schema,
        task_description = meta["task"],
    )
    Path(save_dir).mkdir(parents = True, exist_ok = True)
    
    with open(
        f"{save_dir}/{env_name}_skillset.json",
        "w",
    )as f:
        json.dump(skillset.__dict__, f, indent = 2, default = str)
    
    policy_module = make_policy_module(
        skillset = skillset.__dict__,
        field_names = meta["fields"],
        name = f"llm_{env_name}",
    )
    metrics = train_fn(policy_module)
    
    return{
        "skillset": skillset,
        "metrics": metrics
    }