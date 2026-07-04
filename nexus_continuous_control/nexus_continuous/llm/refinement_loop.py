""" 
Iterative LLM refinement loop for NEXUS skill discovery.
Propose -> train -> evaluate -> feedback -> revise

Mechanism for LLM vs hand-written comparison, iterative skill imporvement 
and multi-seed evaluation loops (paper setting)
"""

from __future__ import annotations

import json
from typing import Any, Dict
from dataclasses import dataclass

from nexus_continuous.llm.pipeline import LLMSkillPipeline, validate_and_build
from nexus_continuous.llm.client import LLMClient

def summarize_metrics(metrics: Dict[str, Any]) -> str:
    """ Convert raw training metrics into a compact LLM-readabli summary """
    
    keys_of_interest = [
        "returns/env_reward_mean",
        "returns/skill_reward_mean",
        "policy_diag/primary_success_rate",
        "policy_diag/primary_goal_metric",
        "env/returned_episode_returns"
    ]
    
    summary = []
    for k in keys_of_interest:
        if k in metrics:
            v = metrics[k]
            try:
                v = float(v)
            except Exception:
                pass
            summary.append(f"{k}: {v}")
    
    return "\n".join(summary) if summary else str(metrics)

def build_feedback_prompt(
    env_name: str,
    previous_skillset: Dict[str, Any],
    metrics_summary: str
) -> tuple[str, str]:
    
    """ Ask LLM to improve previous skillset based on performance """
    
    system_prompt = (
        "You are an expert reinforcement learning rresearcher improving hierarchicaal skills "
        "for continuous control agents."
    )
    
    user_prompt = f"""
    We trained a hierarchical RL agent on environment: {env_name}
    Previous skillset: {json.dumps(previous_skillset, indent = 2)}
    Training performance metrics: {metrics_summary}
    
    Task: Improve the skill decomposition.
    
    Return a new improved JSON skillset with:
    - better reward shaping
    - better activation rules
    - improved skill progression
    - same schema as before
       
    Rules:
    - Keep 3-5 skills
    - Make skills more disctinct and non-overlapping
    - Improve weakest-performing skills
    - Avoid overly strict activation rules
    - Ensure reward signals are dense and learnable
    
    Return ONLY valid JSON.
    """
    return system_prompt, user_prompt

@dataclass
class RefinementConfig: 
    env_name: str
    observation_schema: str
    task_decription: str
    num_iterations: int = 3
    
class LLMRefinementLoop:
    def __init__(self, pipeline: LLMSkillPipeline, client: LLMClient):
        self.pipeline = pipeline
        self.client = client
    
    def propose_initial(self, cfg: RefinementConfig):
        """ Initial proposal """
        return self.pipeline.generate_skillset(
            env_name = cfg.env_name,
            observation_schema = cfg.observation_schema,
            task_description = cfg.task_decription
        )
        
    def refine_once(
        self,
        cfg:RefinementConfig,
        previous_skillset,
        metrics:Dict[str, Any]
    ):
        """ Refinement step """
        metrics_summary = summarize_metrics(metrics)
        
        system_prompt, user_prompt = build_feedback_prompt(
            cfg.env_name,
            previous_skillset.__dict__ if hasattr(previous_skillset, "__dict__") else previous_skillset,
            metrics_summary
        )
        
        return self.client.generate_json(system_prompt, user_prompt)
    
    def run(self, cfg:RefinementConfig, train_fn, eval_fn = None):
        """ 
        Full loop
        train_fn(skillset) -> metrics
        eval_fn optional extra evaluation stage
        """
        
        skillset = self.propose_initial(cfg)
        history = []
        
        for i in range(cfg.num_iterations):
            print(f"\n[LLM LOOP] iteration {i}")
            
            metrics = train_fn(skillset)
            if eval_fn is not None:
                metrics = {**metrics, **eval_fn(skillset)}
            
            history.append(
                {
                    "iteration": i,
                    "metrics": metrics,
                    "skillset": skillset
                }
            )
            print("Metrics:")
            print(summarize_metrics(metrics))
            
            try:
                raw = self.refine_once(cfg, skillset, metrics)
                skillset = validate_and_build(raw)
            except Exception as e:
                print(f"[LLM LOOP] refinement failed: {e}")
                break
        
        return skillset, history