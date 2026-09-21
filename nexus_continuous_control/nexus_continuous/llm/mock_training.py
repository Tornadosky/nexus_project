"""Deterministic, seeded fate training for tests, notebooks and demos.
"""

from __future__ import annotations 
import hashlib 
import random 
from dataclasses import dataclass, asdict, is_dataclass 
from typing import Any, Dict 

def _skillset_dict(skillset: Any) -> Dict[str, Any]:
    if is_dataclass(skillset):
        return asdict(skillset)
    return skillset 

def _stable_seed(skillset_dict: Dict[str, Any], base_seed: int) -> int:
    """ Stable per-skillset seed. Re-training the exact same 
    skillset + seed is reproducible across processes.     
    """
    payload = repr(sorted(skillset_dict.get("skills", []), key = lambda s: s.get("name", "")))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) ^ base_seed) & 0xFFFFFFFF 

def _proxy_quality(skillset_dict: Dict[str, Any]) -> float:
    skills = skillset_dict.get("skills", [])
    n = len(skills)
    if n == 0: 
        return 0.0 
    richness = min(n, 5) / 5.0 
    rules = [s.get("activation_rule", "") for s in skills]
    distinctness = len(set(rules)) / max(1, n)
    has_signal = 0 
    for s in skills: 
        terms = s.get("reward_terms", [])
        if any(abs(float(t.get("weight", 0.0) or 0.0)) > 1e-6 for t in terms):
            has_signal += 1
    signal_ratio = has_signal / n 
    return 0.4 * richness + 0.35 * distinctness + 0.25 * signal_ratio

@dataclass 
class MockTrainer:
    """ Seeded, determininstic 'train_fn' replacement. """
    seed: int = 0
    noise_scale: float = 0.05 
    
    def __call__(self, skillset: Any) -> Dict[str, float]:
        d = _skillset_dict(skillset)
        quality = _proxy_quality(d)
        rng = random.Random(_stable_seed(d, self.seed))
        noise = lambda scale: rng.gauss(0.0, scale)
        
        env_reward = 10.0 * quality + noise(self.noise_scale * 10.0)
        skill_reward = 5.0 * quality + noise(self.noise_scale * 5.0) 
        success_rate = max(0.0, min(1.0, 0.1 + 0.8 * quality + noise(self.noise_scale)))
        goal_metric = max(0.0, min(1.0, quality + noise(self.noise_scale)))
        
        return {
            "returns/env_reward_mean": env_reward,
            "returns/skill_reward_mean": skill_reward,
            "policy_diag/primary_success_rate": success_rate,
            "policy_diag/primary_goal_metric": goal_metric, 
            "env/returned_episode_returns": env_reward
        }
        
def mock_train_fn(skillset: Any, seed: int = 0) -> Dict[str, float]:
    """ Functional convenience wrapper around MockTrainer. """
    return MockTrainer(seed = seed)(skillset)

def hand_written_baseline_metrics(env_name: str, seed: int = 0, quality: float = 0.8) -> Dict[str, float]:
    """ Deterministic stand-in for a hand-written-policy training run. 
    """
    stable = int(hashlib.sha256(env_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random((stable ^ seed) & 0xFFFFFFFF)
    noise = lambda scale: rng.gauss(0.0, scale)  # noqa: E731
    env_reward = 10.0 * quality + noise(0.5)
    success_rate = max(0.0, min(1.0, 0.1 + 0.8 * quality + noise(0.05)))
    return {
        "returns/env_reward_mean": env_reward,
        "policy_diag/primary_success_rate": success_rate,
    }
    