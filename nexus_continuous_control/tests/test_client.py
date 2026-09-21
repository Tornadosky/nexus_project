import pytest

from nexus_continuous.llm.client import (LLMClient, LLMConfig)

def test_mock_backend_returns_valid_json():
    client = LLMClient(LLMConfig(backend="mock"))

    out = client.generate_json("", "")

    assert isinstance(out, dict)
    assert "skills" in out
    assert len(out["skills"]) == 1
    assert out["skills"][0]["name"] == "stable_move"


def test_extract_json_plain():
    client = LLMClient(LLMConfig(backend="mock"))
    text = """
    {
        "a":1,
        "b":2
    }
    """
    out = client.extract_json(text)

    assert out == {"a": 1, "b": 2}


def test_extract_json_with_prefix():
    client = LLMClient(LLMConfig(backend="mock"))
    text = """
Some explanation...

{
    "hello":"world",
    "number":5
}
"""
    out = client.extract_json(text)

    assert out["hello"] == "world"
    assert out["number"] == 5


def test_extract_json_nested():
    client = LLMClient(LLMConfig(backend="mock"))
    text = """
LLM response
{
    "skills":[
        {
            "name":"walk",
            "reward":[1,2,3]
        }
    ]
}
"""
    out = client.extract_json(text)

    assert len(out["skills"]) == 1
    assert out["skills"][0]["name"] == "walk"


def test_extract_invalid_json():
    client = LLMClient(LLMConfig(backend="mock"))
    with pytest.raises(RuntimeError):
        client.extract_json("there is no json here")


def test_mock_skill_schema():
    client = LLMClient(LLMConfig(backend="mock"))
    result = client._mock_response()
    skill = result["skills"][0]

    assert "name" in skill
    assert "description" in skill
    assert "activation_rule" in skill
    assert "reward_terms" in skill