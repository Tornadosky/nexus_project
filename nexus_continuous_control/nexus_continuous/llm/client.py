""" 
LLM client for generating NEXUS skillset and meta-policies.

Swaps OpenAI / local models, mocks outputs for debugging and 
enforces structured JSON generation.
"""

from __future__ import annotations
import json
import os
import random
import shutil
import subprocess
import urllib.error
import urllib.request 
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class LLMConfig:
    backend: str = "hf"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 1000
    seed: int = 0
    
    # Vertex AI (Gemini) only
    region: str = "us-central1"
    project:Optional[str] = None
    
    
def _vertex_gcloud() -> str:
    path = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not path:
        raise RuntimeError(
            "gcloud CLI needed on PATH and authenticated"
        )
    return path 

def _vetex_get_token() -> str:
    out = subprocess.run([_vertex_gcloud(), "auth", "print-access-token"],
                         capture_output = True, text = True)
    lines = [l for l in (out.stdout or "").splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(
            "No gcloud access token. Check authentication."
            "stderr =" + (out.stderr or "")[:200]
        )
    return lines[-1].strip()

def _vertex_get_project() -> str:
    out = subprocess.run([_vertex_gcloud(), "config", "get-value", "project"],
                         capture_output = True, text = True)
    lines = [l for l in (out.stdout or "").splitlines() if l.strip() and "(unset)" not in l]
    return lines[-1].strip() if lines else ""

def _call_vertex_gemini(
    system_prompt: str,
    user_prompt: str,
    token: str,
    project: str,
    model: str,
    region: str,
    temperature: float,
) -> str:
    """Call Gemini on Vertex AI in JSON mode and return the raw text response.
    """
    uri = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/{region}/publishers/google/models/{model}:generateContent")
    body = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        uri, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Vertex AI HTTP {e.code}: {e.read().decode()[:300]}") from e
    return resp["candidates"][0]["content"]["parts"][0]["text"]
    
class LLMClient:
    def __init__(self, config = None, mock_generator: Optional["MockSkillGenerator"] = None):
        self.config = config or LLMConfig()
        self.backend = self.config.backend 
        self.mock_generator = mock_generator
        
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
            
        elif self.backend == "vertex":
            if self.config.model == "Qwen/Qwen2.5-1.5B-Instruct":
                self.config.model = "gemini-2.5-flash"
            if self.config.project is None:
                self.config.project = _vertex_get_project()
            if not self.config.project:
                raise RuntimeError(
                    "backend='vertex' needs a GCP project; pass "
                    "LLMConfig(project=...) or run `gcloud config set project <PROJECT>`."
                )
            
    
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """ Returns parsed JSON from the model. """
        
        if self.backend == "mock":
            if self.mock_generator is not None:
                return self.mock_generator(system_prompt, user_prompt)
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
            
            text = output[0]["generated_text"]
            
            print("=" * 80)
            print(text)
            print("=" * 80)
            
        elif self.backend == "vertex":
            token = _vetex_get_token()
            text = _call_vertex_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                token=token,
                project=self.config.project,
                model=self.config.model,
                region=self.config.region,
                temperature=self.config.temperature,
            )
        
        else:
            raise ValueError(
                f"Unknown LLM backend {self.backend!r}. Expected one of: "
                "'openai', 'hf', 'vertex', 'mock'."
            )
            
        return self.extract_json(text)
    
    def extract_json(self, text) -> Dict[str, Any]:
        
        from json import JSONDecoder 
        
        start = text.find("{")
        if start == -1:
            raise RuntimeError("No JSON found:\n" + text)
        
        decoder = JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON from LLM output ({exc}):\n{text!r}"
            ) from exc
        
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
        
class MockSkillGenerator:
    """ Deterministic, seeded stand-in for a real LLM, for tests, demos and notebooks.
    """
    def __init__(self, fields: tuple[str, ...], seed: int = 0):
        self.fields = tuple(fields)
        self.rng = random.Random(seed)
        self._call_count = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        del system_prompt
        self._call_count += 1
        is_refinement = "Previous skillset" in user_prompt

        n_skills = 3 if not is_refinement else min(5, 3 + (self._call_count // 2))
        skills = []
        for i in range(n_skills):
            field = self.fields[i % len(self.fields)]
            weight = round(0.5 + self.rng.random(), 2)
            threshold = round(0.1 * (i + 1) + 0.05 * self._call_count, 3)
            skills.append(
                {
                    "name": f"skill_{i}_{field}",
                    "description": f"Auto-generated mock skill around '{field}'.",
                    "activation_rule": f"abs({field}) > {threshold}",
                    "reward_terms": [
                        {
                            "type": "negative_distance",
                            "weight": weight,
                            "lhs": field,
                            "rhs": None,
                            "threshold": 0.0,
                        },
                        {
                            "type": "action_penalty",
                            "weight": 0.01,
                            "lhs": None,
                            "rhs": None,
                            "threshold": None,
                        },
                    ],
                }
            )

        return {
            "environment": "mock_env",
            "observation_schema": "\n".join(self.fields),
            "skills": skills,
            "meta_policy_notes": (
                f"mock generator call #{self._call_count}, refinement={is_refinement}"
            ),
        }
