"""Tests for the LLM NexusSkillSet -> JAX interpreter."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nexus_continuous.llm.interpreter import eval_rule, make_policy_module


def _fields():
    return {
        "pole_angle": jnp.asarray([0.0, 0.3, -0.5, 0.1]),
        "cart_position": jnp.asarray([0.0, 0.6, -0.2, 0.4]),
    }


def test_eval_rule_comparison():
    f = _fields()
    out = np.asarray(eval_rule("abs(pole_angle) > 0.2", f))
    assert out.tolist() == [False, True, True, False]


def test_eval_rule_and_or_caseinsensitive():
    f = _fields()
    out = np.asarray(eval_rule("abs(pole_angle) > 0.2 AND abs(cart_position) > 0.3", f))
    assert out.tolist() == [False, True, False, False]
    out2 = np.asarray(eval_rule("abs(pole_angle) > 0.4 or abs(cart_position) > 0.5", f))
    assert out2.tolist() == [False, True, True, False]


def test_eval_rule_constant_true_broadcasts():
    f = _fields()
    out = np.asarray(eval_rule("True", f))
    assert out.shape == (4,) and out.all()


def test_eval_rule_missing_field_is_safe():
    f = _fields()
    # unknown field -> treated as 0, no crash, shape preserved
    out = np.asarray(eval_rule("not_a_field > 1.0", f))
    assert out.shape == (4,) and not out.any()


def test_make_policy_module_shapes_and_finite():
    ss = {
        "skills": [
            {"name": "a", "activation_rule": "abs(pole_angle) > 0.2",
             "reward_terms": [{"type": "negative_distance", "weight": 1.0, "lhs": "pole_angle"},
                              {"type": "action_penalty", "weight": 0.01}]},
            {"name": "b", "activation_rule": "True",
             "reward_terms": [{"type": "posture_penalty", "weight": 0.5, "lhs": "pole_angle"}]},
            {"name": "c", "activation_rule": "abs(cart_position) > 0.3",
             "reward_terms": [{"type": "binary_bonus", "weight": 1.0}]},
        ]
    }
    mod = make_policy_module(ss, ("pole_angle", "cart_position"))
    assert mod.NUM_SKILLS == 3
    B = 4
    obs = {"raw_actor": jnp.zeros((B, 4)), "policy_info": {
        "pole_angle": jnp.asarray([0.0, 0.3, -0.5, 0.1]),
        "cart_position": jnp.asarray([0.0, 0.6, -0.2, 0.4])}}
    action = jnp.zeros((B, 2))
    done = jnp.zeros((B,))
    sr = np.asarray(mod.skill_rewards(obs, obs, action, jnp.zeros((B,)), done))
    assert sr.shape == (B, 3) and np.all(np.isfinite(sr))
    mask = np.asarray(mod.skill_mask(obs))
    assert mask.shape == (B, 3)
    # skill b ("True") always available
    assert mask[:, 1].all()
    sym = np.asarray(mod.symbolic_meta_policy(obs))
    assert sym.shape == (B,) and sym.min() >= 0 and sym.max() < 3


def test_progressive_mask_allows_next_skill():
    # progression skills where only skill 0 ("reach") is active everywhere.
    ss = {"skills": [
        {"name": "reach", "activation_rule": "d > 0.1", "reward_terms": [{"type": "action_penalty", "weight": 0.01}]},
        {"name": "grasp", "activation_rule": "d <= 0.01", "reward_terms": [{"type": "action_penalty", "weight": 0.01}]},
        {"name": "lift", "activation_rule": "d <= 0.005", "reward_terms": [{"type": "action_penalty", "weight": 0.01}]},
    ]}
    obs = {"raw_actor": jnp.zeros((4, 4)), "policy_info": {"d": jnp.asarray([0.2, 0.2, 0.2, 0.2])}}
    strict = make_policy_module(ss, ("d",), mask_mode="strict")
    prog = make_policy_module(ss, ("d",), mask_mode="progressive")
    ms = np.asarray(strict.skill_mask(obs))
    mp = np.asarray(prog.skill_mask(obs))
    # strict: only reach (idx 0) available
    assert ms[:, 0].all() and not ms[:, 1].any() and not ms[:, 2].any()
    # progressive: reach + grasp (idx 0,1), not lift (idx 2)
    assert mp[:, 0].all() and mp[:, 1].all() and not mp[:, 2].any()
