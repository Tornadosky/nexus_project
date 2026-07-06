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
import transformers
import accelerate
import torch
import sentencepiese

@dataclass
class LLMConfig:
    backend: str = "hf"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 1000
    
class LLMClient:
    def __init__(self, config = None):
        self.config = config or LLMConfig()
        self.backend = self.config.backend 
        
        if self.backend == "openai":
            from openai import OpenAI 
            self.cliet = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
            
        elif self.backend == "hf":
            from transformers import pipeline
            
            self.generator = pipeline(
                "text-generation",
                model = self.config.model,
                device_map = "auto"
            )
    
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """ Returns parsed JSON from the model. """
        
        if self.backend == "mock":
            return self._mock_response()
        
        if self.backend == "openai":
        
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
        
        elif self.backend == "hf":
            prompt = f""" 
            {system_prompt}
            {user_prompt}
            Return only JSON.
            """
            
            output = self.generator(
                prompt,
                max_new_tokens = self.config.max_tokens,
                temperature = self.config.temperature
            )
            
            text = output[0]["generated_text"]
            
        return self.extract_json(text)
    
    def extract_json(self, text):
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        raise RuntimeError("No JSON found: \n" + text)
    
    def _mock_response(self) -> Dict[str, Any]:
        """ Safe fallback when no API key present """
        return{
            "skills":[
                {
                    "name": "stable_move",
                    "description": "Default safe movement skill",
                    "activation_rule": "true",
                    "reward_terms":[]
                }
            ],
            "meta_policy_notes": "mock fallback policy"
        }