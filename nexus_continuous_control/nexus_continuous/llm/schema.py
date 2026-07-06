"""Safe schema for optional LLM-generated skills.

The training code does not execute arbitrary Python returned by an LLM. Instead,
use this JSON-friendly schema and convert it into reviewed JAX reward/meta-policy
functions in `nexus_continuous.policies`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# RewardTermType = Literal[
#     "negative_distance",
#     "positive_velocity",
#     "target_height",
#     "binary_bonus",
#     "action_penalty",
#     "posture_penalty",
# ]


@dataclass(frozen=True)
class RewardTerm:
    type: str
    weight: float
    lhs: Optional[str] = None
    rhs: Optional[str] = None
    threshold: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    activation_rule: str
    reward_terms: list[RewardTerm]


@dataclass(frozen=True)
class NexusSkillSet:
    environment: str
    observation_schema: str
    skills: list[SkillSpec]
    meta_policy_notes: str
