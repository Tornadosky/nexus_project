import pytest
from nexus_continuous.llm.pipeline import (
    build_skill_prompt,
    validate_and_build,
    _parse_reward_terms,
    _parse_skill,
    LLMSkillPipeline
)
from nexus_continuous.llm.client import (LLMClient, LLMConfig)
from nexus_continuous.llm.schema import (NexusSkillSet, SkillSpec, RewardTerm)

# Prompt generation
def test_build_skill_prompt_contains_environment():
    system, user = build_skill_prompt(
        env_name="WalkerWalk",
        observation_schema="height, velocity",
        task_description="Walk forward"
    )

    assert "WalkerWalk" in user
    assert "height" in user
    assert "Walk forward" in user
    assert "JSON" in user
    assert system != ""

# Reward parsing
def test_parse_reward_terms():
    raw = [
        {
            "type": "positive_velocity",
            "weight": 2,
            "lhs": "forward_velocity",
            "threshold": None
        },
        {
            "type": "action_penalty",
            "weight":0.1
        }
    ]
    result = _parse_reward_terms(raw)

    assert len(result) == 2
    assert isinstance(result[0], RewardTerm)
    assert result[0].type == "positive_velocity"
    assert result[0].weight == 2.0
    assert result[0].lhs == "forward_velocity"


def test_parse_reward_defaults():
    raw = [{"type":"posture_penalty"}]
    result = _parse_reward_terms(raw)

    assert result[0].weight == 1.0
    assert result[0].lhs is None

# Skill parsing
def test_parse_skill():
    raw = {
        "name":"walk",
        "description":"Move forward",
        "activation_rule":"height > 0.8",
        "reward_terms":[
            {
                "type":"positive_velocity",
                "weight":1,
                "lhs":"forward_velocity"
            }
        ]
    }
    skill = _parse_skill(raw)

    assert isinstance(skill, SkillSpec)
    assert skill.name == "walk"
    assert skill.activation_rule == "height > 0.8"
    assert len(skill.reward_terms)==1

def test_parse_skill_missing_optional_fields():
    raw = {"name":"recover"}
    skill = _parse_skill(raw)

    assert skill.name == "recover"
    assert skill.description == ""
    assert skill.activation_rule == "True"
    assert skill.reward_terms == []
    
# Schema validation
def test_validate_valid_skillset():
    raw = {
        "environment":"WalkerWalk",
        "observation_schema":
            "height,velocity",
        "skills":[
            {
                "name":"walk",
                "description":
                    "forward movement",
                "activation_rule":
                    "height > 0.8",
                "reward_terms":[
                    {
                        "type":
                            "positive_velocity",
                        "weight":1.0,
                        "lhs":
                            "forward_velocity"
                    }
                ]
            }
        ],
        "meta_policy_notes":
            "progressive skills"
    }
    result = validate_and_build(raw)

    assert isinstance(result, NexusSkillSet)
    assert result.environment == "WalkerWalk"
    assert len(result.skills)==1
    assert result.skills[0].name=="walk"

def test_validate_missing_skills():
    raw = {"environment":"WalkerWalk"}

    with pytest.raises(ValueError):
        validate_and_build(raw)

# Pipeline integration
def test_pipeline_with_mock_client():
    client = LLMClient(LLMConfig(backend="mock"))
    pipeline = LLMSkillPipeline(client)
    result = pipeline.generate_skillset(
        env_name="WalkerWalk",
        observation_schema="""
        height,
        forward_velocity
        """,
        task_description=
        "Walk forward"
    )

    assert isinstance(result, NexusSkillSet)
    assert len(result.skills)>0
    assert result.skills[0].name == "stable_move"
    
# Retry behaviour
class BrokenClient:
    def __init__(self):
        self.calls = 0

    def generate_json(self, system_prompt, user_prompt):
        self.calls += 1
        raise ValueError("bad json")

def test_pipeline_failure_after_retries():
    pipeline = LLMSkillPipeline(BrokenClient())

    with pytest.raises(RuntimeError):
        pipeline.generate_skillset(
            env_name="WalkerWalk",
            observation_schema="height",
            task_description="walk",
            max_retries=2
        )

# Invalid LLM output
class InvalidJSONClient:
    def generate_json(self, system_prompt, user_prompt):
        return {"hello":"world"}

def test_pipeline_rejects_invalid_output():
    pipeline = LLMSkillPipeline(InvalidJSONClient())

    with pytest.raises(RuntimeError):
        pipeline.generate_skillset(
            env_name="WalkerWalk",
            observation_schema="height",
            task_description="walk",
            max_retries=0
        )