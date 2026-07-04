""" 
LLM client for generating NEXUS skillset and meta-policies.

Swaps OpenAI / local models, mocks outputs for debugging and 
enforces structured JSON generation.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    
class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.api_key = os.getenv("OPENAI_API_KEY", None)
        self._use_mock = self.api_key is None
        
        if not self._use_mock:
            from openai import OpenAI
            self.client = OpenAI()
    
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """ Returns parsed JSON from the model. """
        
        if self._use_mock:
            return self._mock_response()
        
        response = self.client.chat.completions.create(
            model = self.config.model,
            temperature = self.config.temperature,
            max_tokens = self.config.max_tokens,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content

        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            raise RuntimeError(f"LLM did not return valid JSON:\n{text}")
    
    def _mock_response(self) -> Dict[str, Any]:
        """ Safe fallback when no API key present """
        return{
            "skills":[
                {
                    "name": "stable_move",
                    "description": "Default safe movement skill",
                    "activation_rule": "true",
                    "reward_terms":[
                        {"type": "positive_velocity", "weight": 1.0, "lhs": "forward_velocity"}
                    ],
                }
            ],
            "meta_policy_notes": "mock fallback policy"
        }