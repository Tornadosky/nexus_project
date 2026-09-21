""" 
LLM -> Skillset pipeline for NEXUS.

Builds prompts for the LLM and calls the LLM client. The output
is validated and normalized into NexusSkillSet schema and is prepared
for compilation by interpreter.py
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional
from dataclasses import asdict
from nexus_continuous.llm.client import LLMClient
from nexus_continuous.llm.schema import NexusSkillSet, SkillSpec, RewardTerm

def skillset_to_dict(skillset: NexusSkillSet) -> Dict[str, Any]:
    """ Deeply convert a NexusSkillSet to a plain JSON-safe dict. """
    return asdict(skillset)

def build_skill_prompt(
    env_name: str,
    observation_schema: str,
    task_description: str
) -> tuple[str, str]:
    """ Returns (system_prompt, user_prompt) """
    
    system_prompt = (
        "You are an expert reinforcement learning researcher designing "
        "interpretable hierarchical skill policies for continuous control."
    )
    user_prompt = f"""
    You are designing skills fro the MuJoCo Playground environment: {env_name}
    OBSERVATION SCHEMA: {observation_schema}
    TASK DESCRIPTION: {task_description}
    
    Return ONLY valid JSON matching this structure:
    {{
        "environment": "{env_name}",
        "observation_schema": "...",
        "skills":[
            {{
                "name": "string",
                "description": "string",
                "activation_rule": "boolean expression over fields",
                "reward_terms": [
                    {{
                        "type": "negative_distance | positive_velocity | target_height | binary_bonus | action_penalty | posture_penalty",
                        "weight": 1.0,
                        "lhs": "observation_field_or_null",
                        "rhs": "observation_field_or_numeric_constant_or_null",
                        "threshold": 0.0
                    }}
                ]
            }}
        ],
        "meta_policy_notes": "string"
    }}
    
    Rules:
    - Use ONLY fields from observation schema
    - lhs MUST be an observation field name or null
    - rhs MUST be either an observation field name, a numeric constant, or null
    - 3-5 skills only
    - Skills should form a progression (safe -> locomotion -> optimal performance)
    - activation_rule must be a boolean expression (and/or/not, comparisons)
    - reward must be stable and dense (avoid sparse-only rewards)
    - Do not include markdown fences.
    - Do not explain anything.
    - Output only the JSON.
    
    """
    return system_prompt, user_prompt

def _coerce_numeric(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value

def _parse_reward_terms(raw_terms: list[dict]) -> list[RewardTerm]:
    terms = []
    for t in raw_terms:
        terms.append(
            RewardTerm(
                type = t["type"],
                weight = float(t.get("weight", 1.0)),
                lhs = _coerce_numeric(t.get("lhs")),
                rhs = _coerce_numeric(t.get("rhs")),
                threshold = t.get("threshold"),
                description = t.get("description", "")
            )
        )
    return terms

def _parse_skill(spec: dict) -> SkillSpec:
    return SkillSpec(
        name = spec["name"],
        description = spec.get("description", ""),
        activation_rule = spec.get("activation_rule", "True"),
        reward_terms = _parse_reward_terms(spec.get("reward_terms", []))
    )
    
def _check_fields(skills: list[SkillSpec], allowed_fields: Optional[set[str]]) -> None:
    if not allowed_fields:
        return
    for s in skills:
        for term in s.reward_terms:
            for side_name, side_val in (("lhs", term.lhs), ("rhs", term.rhs)):
                if side_val is None or isinstance(side_val, (int, float)):
                    continue
                if side_val not in allowed_fields:
                    warnings.warn(
                        f"Skill '{s.name}' reward term references unknown field "
                        f"{side_name}={side_val!r}; not in observation schema "
                        f"{sorted(allowed_fields)}. This term will contribute 0 "
                        "reward at runtime (interpreter falls back to a default "
                        "of 0 for unknown fields).",
                        stacklevel=3,
                    )
    
def validate_and_build(raw: Dict[str, Any], allowed_fields: Optional[set[str]] = None) -> NexusSkillSet:
    """ Converts raw LLM JSON -> typed schema. Raises Value Error if invalid. """
    
    if "skills" not in raw:
        raise ValueError("missing 'skills' in LLM output")
    
    if not raw["skills"]:
        raise ValueError("LLM output has an empty 'skills' list")
    
    skills = [_parse_skill(s) for s in raw["skills"]]
    _check_fields(skills, allowed_fields)
    
    return NexusSkillSet(
        environment = raw.get("environment", "unknown"),
        observation_schema = raw.get("observation_schema", ""),
        skills = skills,
        meta_policy_notes = raw.get("meta_policy_notes", ""),
    )
    
    
class LLMSkillPipeline:
    def __init__(self, client: LLMClient):
        self.client = client
    
    def generate_skillset(
        self, 
        env_name: str, 
        observation_schema: str,
        task_description: str,
        max_retries: int = 2,
        allowed_fields: Optional[set[str]] = None,
    ) -> NexusSkillSet:
        """ Main entrypoint: LLM -> JSON -> validated NexusSkillSet """
        
        system_prompt, user_prompt = build_skill_prompt(env_name, observation_schema, task_description)
        last_error = None
        raw = None
        
        for _ in range(max_retries + 1):
            try:
                raw = self.client.generate_json(system_prompt, user_prompt)
                return validate_and_build(raw, allowed_fields=allowed_fields)
            except Exception as e:
                last_error = str(e)
                # strengthen prompt on retry
                user_prompt += f"\n\nIMPORTANT: Previous output was invalid JSON: {last_error}\nReturn STRICT JSON ONLY."
        
        raise RuntimeError(
            f"Failed to generate valid skillset after retries.\nLast error: {last_error}\nRaw: {raw}"
        )


def generate_skillset(
    env_name: str,
    observation_schema: str,
    task_description: str,
    client: LLMClient | None = None,
    allowed_fields: Optional[set[str]] = None
) -> NexusSkillSet:
    """ Simple functional API """
    
    client = client or LLMClient()
    pipeline = LLMSkillPipeline(client)
    
    return pipeline.generate_skillset(
        env_name = env_name,
        observation_schema = observation_schema,
        task_description = task_description,
        allowed_fields = allowed_fields
    )
    
def save_skillset(skillset, path):
    import json 
    import os
    
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(asdict(skillset), f, indent = 2)