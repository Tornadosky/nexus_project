"""Compile an LLM-proposed NexusSkillSet (JSON) into runnable JAX skill rewards,
masks and a symbolic meta-policy, so LLM-generated skills can be trained and
compared head-to-head against the hand-written policies.

Only the *typed* JSON spec is consumed -- no LLM code is executed. Activation
rules are evaluated with a restricted AST walker (names -> named state fields,
abs(), comparisons, and/or/not, +-*/ ) over JAX arrays; reward terms are mapped
from the fixed vocabulary (negative_distance, positive_velocity, target_height,
binary_bonus, action_penalty, posture_penalty) to JAX operations on the named
state fields.

This module targets the scalar-feature environments (cartpole, cheetah, walker,
hopper) whose LLM field names are exactly the adapter's semantic keys. Vector
fields (panda/go1 positions) are out of scope here.
"""
from __future__ import annotations

import ast
import json
from types import ModuleType
from typing import Any

import jax.numpy as jnp

from nexus_continuous.llm.jax_bootstrap import ensure_jax 

ensure_jax()

try:
    from nexus_continuous.policies.common import actor_obs, feature_info, info_value, safe_index
except Exception: 
    from nexus_continuous.llm.common_fallback import actor_obs, feature_info, info_value, safe_index


# Default obs-index fallbacks per env field (used only if the semantic key is
# absent). Indices follow the hand-written policies' conventions.
_FALLBACK_INDEX = {
    "cart_position": 0, "pole_angle": 1, "cart_velocity": -2, "pole_angular_velocity": -1,
    "torso_pitch": 1, "pitch": 1, "torso_height": 0, "height": 0,
    "forward_velocity": -1, "x_velocity": -1,
}


def _field(name: str, obs: Any, semantic: Any, x: jnp.ndarray) -> jnp.ndarray:
    """Resolve a named scalar field from semantic info, with index fallback."""
    if name == "joint_speed":
        default = jnp.mean(jnp.abs(x[..., x.shape[-1] // 2:]), axis=-1)
        return info_value(semantic, ("joint_speed", "metrics/joint_speed"), default)
    idx = _FALLBACK_INDEX.get(name, 0)
    default = safe_index(x, idx)
    # try the name and a few common aliases
    aliases = (name,)
    if name in ("torso_height",):
        aliases = ("torso_height", "height")
    elif name in ("torso_pitch",):
        aliases = ("torso_pitch", "pitch")
    elif name in ("forward_velocity",):
        aliases = ("forward_velocity", "x_velocity")
    return info_value(semantic, aliases, default)


def _build_fields(field_names: tuple[str, ...], obs: Any, info: Any) -> dict[str, jnp.ndarray]:
    x = actor_obs(obs)
    semantic = feature_info(obs, info)
    return {n: jnp.asarray(_field(n, obs, semantic, x)) for n in field_names}


# ---- restricted AST evaluator for activation rules ----
_ALLOWED_CALLS = {"abs": jnp.abs, "min": jnp.minimum, "max": jnp.maximum}


def _normalize(rule: str) -> str:
    # LLMs emit AND/OR/NOT in various cases; Python needs lowercase keywords.
    import re
    rule = re.sub(r"\bAND\b", "and", rule)
    rule = re.sub(r"\bOR\b", "or", rule)
    rule = re.sub(r"\bNOT\b", "not", rule)
    return rule


def _ev(node: ast.AST, f: dict[str, jnp.ndarray]) -> jnp.ndarray:
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, f) for v in node.values]
        op = jnp.logical_and if isinstance(node.op, ast.And) else jnp.logical_or
        out = vals[0]
        for v in vals[1:]:
            out = op(out, v)
        return out
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_ev(node.operand, f)
        if isinstance(node.op, ast.Not):
            return jnp.logical_not(_ev(node.operand, f))
        return _ev(node.operand, f)
    if isinstance(node, ast.BinOp):
        l, r = _ev(node.left, f), _ev(node.right, f)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
        raise ValueError(f"unsupported binop {node.op}")
    if isinstance(node, ast.Compare):
        left = _ev(node.left, f)
        result = None
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, f)
            if isinstance(op, ast.Gt):
                c = left > right
            elif isinstance(op, ast.GtE):
                c = left >= right
            elif isinstance(op, ast.Lt):
                c = left < right
            elif isinstance(op, ast.LtE):
                c = left <= right
            elif isinstance(op, ast.Eq):
                c = jnp.abs(left - right) < 1e-6
            elif isinstance(op, ast.NotEq):
                c = jnp.abs(left - right) >= 1e-6
            else:
                raise ValueError(f"unsupported compare {op}")
            result = c if result is None else jnp.logical_and(result, c)
            left = right
        return result
    if isinstance(node, ast.Call):
        fn = _ALLOWED_CALLS.get(getattr(node.func, "id", ""))
        args = [_ev(a, f) for a in node.args]
        if fn is None:
            return args[0] if args else jnp.asarray(0.0)
        return fn(*args)
    if isinstance(node, ast.Name):
        return f.get(node.id, jnp.asarray(0.0))
    if isinstance(node, ast.Constant):
        return jnp.asarray(float(node.value)) if isinstance(node.value, (int, float)) else jnp.asarray(0.0)
    raise ValueError(f"unsupported node {type(node).__name__}")


def eval_rule(rule: str, fields: dict[str, jnp.ndarray]) -> jnp.ndarray:
    shape = jnp.shape(next(iter(fields.values())))
    if not rule or not rule.strip():
        return jnp.ones(shape, dtype=bool)
    try:
        tree = ast.parse(_normalize(rule), mode="eval")
        out = _ev(tree.body, fields)
        # constants ("True") or missing-field results are scalars -> broadcast.
        return jnp.broadcast_to(jnp.asarray(out, dtype=bool), shape)
    except Exception:
        return jnp.zeros(shape, dtype=bool)


# ---- reward-term -> JAX ----
def _term_reward(term: dict, fields: dict[str, jnp.ndarray], action: jnp.ndarray) -> jnp.ndarray:
    t = term.get("type")
    w = float(term.get("weight", 1.0) or 1.0)
    lhs = fields.get(term.get("lhs")) if term.get("lhs") in fields else None
    rhs = fields.get(term.get("rhs")) if term.get("rhs") in fields else None
    thr = term.get("threshold")
    thr = float(thr) if thr is not None else None
    zero = jnp.zeros(action.shape[:-1], dtype=action.dtype)
    if t == "action_penalty":
        return -w * jnp.sum(jnp.square(action), axis=-1)
    if t == "posture_penalty":
        v = lhs if lhs is not None else zero
        return -w * jnp.abs(v)
    if t == "negative_distance":
        if lhs is None:
            return zero
        if rhs is not None:
            return -w * jnp.abs(lhs - rhs)
        if thr is not None:
            return -w * jnp.abs(lhs - thr)
        return -w * jnp.abs(lhs)
    if t == "positive_velocity":
        v = lhs if lhs is not None else zero
        return w * v
    if t == "target_height":
        v = lhs if lhs is not None else zero
        if thr and thr > 0:
            return w * jnp.clip(v / thr, 0.0, 1.0)
        return w * v
    if t == "binary_bonus":
        if lhs is not None and thr is not None:
            return w * (lhs > thr).astype(action.dtype)
        return w * jnp.ones(action.shape[:-1], dtype=action.dtype)
    return zero


def make_policy_module(skillset: dict, field_names: tuple[str, ...],
                       task_metrics_fn=None, name: str = "llm_generated",
                       field_fn=None, mask_mode: str = "strict") -> ModuleType:
    """Return a policy-module-like object the trainer can load.

    field_fn: optional callable(obs, info) -> {name: array} for envs that need
    derived fields (e.g. panda distances). Defaults to scalar semantic lookup.

    mask_mode: how to compile the NeSy mask from per-skill activation rules.
      "strict"      -- mask = activation rule (faithful, but the LLM's mutually
                       exclusive rules can collapse to one skill).
      "progressive" -- for progression-style skills (reach->grasp->lift->...),
                       also allow every skill up to (highest active index + 1),
                       so the meta can try the next step instead of being stuck.
    """
    skills = skillset.get("skills", [])
    skill_names = tuple(s.get("name", f"skill_{i}") for i, s in enumerate(skills))
    num_skills = len(skills)
    if num_skills == 0:
        raise ValueError()
    build = field_fn if field_fn is not None else (lambda obs, info: _build_fields(field_names, obs, info))

    def skill_rewards(prev_obs, obs, action, env_reward, done, info=None):
        fields = build(obs, info)
        batch = action.shape[:-1]
        cols = []
        for s in skills:
            terms = s.get("reward_terms", [])
            r = sum((_term_reward(t, fields, action) for t in terms),
                    jnp.zeros(batch, dtype=action.dtype))
            cols.append(jnp.broadcast_to(jnp.asarray(r, dtype=action.dtype), batch))
        rewards = jnp.stack(cols, axis=-1)
        return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)

    def _activations(obs, info=None):
        fields = build(obs, info)
        return [eval_rule(s.get("activation_rule", ""), fields) for s in skills]

    def skill_mask(obs, info=None):
        acts = _activations(obs, info)
        mask = jnp.stack(acts, axis=-1)  # [..., num_skills] bool
        if mask_mode == "progressive" and num_skills > 1:
            idx = jnp.arange(num_skills)
            # highest active skill index (0 if none active)
            active_idx = jnp.where(mask, idx, -1)
            highest = jnp.max(active_idx, axis=-1, keepdims=True)
            highest = jnp.maximum(highest, 0)
            allow = idx <= (highest + 1)  # allow up to one step past the frontier
            mask = mask | allow
        any_active = jnp.any(mask, axis=-1, keepdims=True)
        default = jnp.zeros_like(mask).at[..., 0].set(True)
        return jnp.where(any_active, mask, default)

    def symbolic_meta_policy(obs, info=None):
        acts = _activations(obs, info)
        out = jnp.zeros(acts[0].shape, dtype=jnp.int32)
        # priority: last listed skill wins ties only if earlier inactive
        for i in reversed(range(num_skills)):
            out = jnp.where(acts[i], i, out)
        return out.astype(jnp.int32)

    mod = ModuleType(name)
    mod.SKILL_NAMES = skill_names
    mod.NUM_SKILLS = num_skills
    mod.skill_rewards = skill_rewards
    mod.skill_mask = skill_mask
    mod.symbolic_meta_policy = symbolic_meta_policy
    if task_metrics_fn is not None:
        mod.task_metrics = task_metrics_fn
    mod.explain_policy = lambda: f"LLM-generated skills: {', '.join(skill_names)}"
    return mod


def load_skillset_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
