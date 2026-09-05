"""Safe schema for optional LLM-generated skills.

The training code does not execute arbitrary Python returned by an LLM. Instead,
use this JSON-friendly schema and convert it into reviewed JAX reward/meta-policy
functions in `nexus_continuous.policies`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RewardTermType = Literal[
    "negative_distance",
    "positive_velocity",
    "target_height",
    "binary_bonus",
    "action_penalty",
    "posture_penalty",
]


@dataclass(frozen=True)
class RewardTerm:
    type: RewardTermType
    weight: float = 1.0
    lhs: str | None = None
    rhs: str | None = None
    threshold: float | None = None
    description: str = ""


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    activation_rule: str
    reward_terms: list[RewardTerm] = field(default_factory=list)


@dataclass(frozen=True)
class NexusSkillSet:
    environment: str
    observation_schema: str
    skills: list[SkillSpec]
    meta_policy_notes: str = ""
