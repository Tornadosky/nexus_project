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
    backend: str = "hf"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 500
    
class LLMClient:
    def __init__(self, config = None):
        self.config = config or LLMConfig()
        self.backend = self.config.backend 
        
        if self.backend == "openai":
            from openai import OpenAI 
            self.client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
            
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = self.generator.tokenizer.apply_chat_template(
                messages, 
                tokenize = False,
                add_generation_prompt = True
            )
            
            output = self.generator(
                prompt,
                max_new_tokens = self.config.max_tokens,
                do_sample = False,
                eos_token_id = self.generator.tokenizer.eos_token_id,
            )
            
            text = output[0]["generated_text"][len(prompt):]
            
            print("=" * 80)
            print(text)
            print("=" * 80)
            
        return self.extract_json(text)
    
    def extract_json(self, text):
        
        from json import JSONDecoder 
        
        start = text.find("{")
        if start == -1:
            raise RuntimeError("No JSON found:\n" + text)
        
        decoder = JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        
        return obj
    
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