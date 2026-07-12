"""CPU shape tests for the RGB skill-actor modules (vision.py).

Confirms the vision encoder + vision actor produce correctly shaped, in-range
actions for pixel + proprioception inputs. Run:
    python -m pytest -q tests/test_vision_shapes.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nexus_continuous.vision import RGBEncoder, VisionSkillActor


def test_rgb_encoder_output_shape():
    enc = RGBEncoder(embedding_dim=64)
    pixels = jnp.zeros((4, 64, 64, 3), dtype=jnp.uint8)
    params = enc.init(jax.random.PRNGKey(0), pixels)
    z = enc.apply(params, pixels)
    assert z.shape == (4, 64)


def test_vision_actor_shape_and_range():
    action_dim = 6
    scale = jnp.ones((action_dim,)) * 2.0
    bias = jnp.zeros((action_dim,))
    actor = VisionSkillActor(action_dim=action_dim, action_scale=scale, action_bias=bias,
                             hidden_sizes=(64, 64), embedding_dim=32)
    pixels = jnp.zeros((4, 48, 48, 3), dtype=jnp.float32)
    proprio = jnp.zeros((4, 10), dtype=jnp.float32)
    params = actor.init(jax.random.PRNGKey(0), pixels, proprio)
    a = actor.apply(params, pixels, proprio)
    assert a.shape == (4, action_dim)
    # tanh * scale + bias => within [bias-scale, bias+scale] = [-2, 2]
    assert np.all(np.asarray(a) <= 2.0 + 1e-4) and np.all(np.asarray(a) >= -2.0 - 1e-4)


def test_vision_actor_vmap_over_skills():
    """It must vmap over a leading skill axis exactly like SkillActor does."""
    action_dim = 3
    scale = jnp.ones((action_dim,)); bias = jnp.zeros((action_dim,))
    actor = VisionSkillActor(action_dim=action_dim, action_scale=scale, action_bias=bias,
                             hidden_sizes=(32,), embedding_dim=16)
    pixels = jnp.zeros((5, 32, 32, 3)); proprio = jnp.zeros((5, 8))
    n_skills = 3
    keys = jax.random.split(jax.random.PRNGKey(0), n_skills)
    skill_params = jax.vmap(lambda k: actor.init(k, pixels, proprio)["params"])(keys)
    out = jax.vmap(lambda p: actor.apply({"params": p}, pixels, proprio))(skill_params)
    assert out.shape == (n_skills, 5, action_dim)
