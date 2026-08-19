"""CPU shape tests for the RGB skill-actor modules (vision.py).

Confirms the vision encoder + vision actor produce correctly shaped, in-range
actions for pixel + proprioception inputs. Run:
    python -m pytest -q tests/test_vision_shapes.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nexus_continuous.vision import (
    RGBEncoder,
    SharedRGBTrunk,
    VisionSkillActor,
    VisionSkillHead,
    build_rgb_actor_fns,
)


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


# --------------------------------------------------------------------------- #
# Shared-encoder modules (RGB_SHARED_ENCODER).                                 #
# --------------------------------------------------------------------------- #


def test_shared_trunk_shapes_and_aux():
    trunk = SharedRGBTrunk(embedding_dim=32, aux_state_dim=5)
    pixels = jnp.zeros((4, 32, 32, 3), dtype=jnp.float32)
    params = trunk.init(jax.random.PRNGKey(0), pixels)
    z = trunk.apply(params, pixels)
    assert z.shape == (4, 32)
    z2, state_pred = trunk.apply(params, pixels, return_aux=True)
    assert z2.shape == (4, 32) and state_pred.shape == (4, 5)
    # The aux head lives on the trunk, next to the encoder.
    assert set(params["params"]) == {"RGBEncoder_0", "aux_state"}


def test_shared_trunk_without_aux_has_no_aux_head():
    trunk = SharedRGBTrunk(embedding_dim=16)
    pixels = jnp.zeros((2, 16, 16, 3), dtype=jnp.float32)
    params = trunk.init(jax.random.PRNGKey(0), pixels)
    assert set(params["params"]) == {"RGBEncoder_0"}
    assert trunk.apply(params, pixels, return_aux=True)[1] is None


def test_vision_skill_head_shape_range_and_vmap():
    action_dim = 3
    scale = jnp.ones((action_dim,)) * 2.0
    bias = jnp.zeros((action_dim,))
    head = VisionSkillHead(action_dim=action_dim, action_scale=scale, action_bias=bias,
                           hidden_sizes=(32,))
    latent = jnp.zeros((5, 16)); proprio = jnp.zeros((5, 8))
    params = head.init(jax.random.PRNGKey(0), latent, proprio)
    a = head.apply(params, latent, proprio)
    assert a.shape == (5, action_dim)
    assert np.all(np.abs(np.asarray(a)) <= 2.0 + 1e-4)
    # The head params must have the SAME submodule name as the unshared actor's
    # MLP, so a per-skill slice of either tree is interchangeable.
    assert set(params["params"]) == {"MLP_0"}
    # And it must stack over a leading skill axis exactly like VisionSkillActor.
    n_skills = 4
    keys = jax.random.split(jax.random.PRNGKey(1), n_skills)
    stacked = jax.vmap(lambda k: head.init(k, latent, proprio)["params"])(keys)
    out = jax.vmap(lambda p: head.apply({"params": p}, latent, proprio))(stacked)
    assert out.shape == (n_skills, 5, action_dim)


def test_shared_trunk_plus_heads_equals_vision_skill_actor():
    """trunk.apply + vmap(head.apply) must reproduce vmap(VisionSkillActor.apply).

    Built by taking N independently-initialized VisionSkillActor parameter sets
    and OVERWRITING every encoder with skill 0's encoder -- i.e. an unshared
    actor that happens to share its CNN. The factored (trunk + heads) form must
    then compute exactly the same actions from the correspondingly-sliced params.
    """
    action_dim = 2
    embed = 16
    hidden = (32, 32)
    scale = jnp.full((action_dim,), 1.5)
    bias = jnp.full((action_dim,), 0.25)
    actor = VisionSkillActor(action_dim=action_dim, action_scale=scale, action_bias=bias,
                             hidden_sizes=hidden, embedding_dim=embed)
    head = VisionSkillHead(action_dim=action_dim, action_scale=scale, action_bias=bias,
                           hidden_sizes=hidden)
    trunk = SharedRGBTrunk(embedding_dim=embed)

    pixels = jax.random.uniform(jax.random.PRNGKey(7), (6, 24, 24, 3), minval=-0.5, maxval=0.5)
    proprio = jax.random.normal(jax.random.PRNGKey(8), (6, 3))
    n_skills = 3
    keys = jax.random.split(jax.random.PRNGKey(0), n_skills)
    stacked = jax.vmap(lambda k: actor.init(k, pixels, proprio)["params"])(keys)
    # Force one common encoder (skill 0's) into all N actors.
    shared_enc = jax.tree_util.tree_map(lambda leaf: leaf[0], stacked["RGBEncoder_0"])
    stacked = dict(stacked)
    stacked["RGBEncoder_0"] = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf[None], (n_skills,) + leaf.shape), shared_enc
    )

    reference = jax.vmap(lambda p: actor.apply({"params": p}, pixels, proprio))(stacked)

    # Corresponding slices: encoder -> the trunk, MLP_0 -> the stacked heads.
    trunk_params = {"RGBEncoder_0": shared_enc}
    head_params = {"MLP_0": stacked["MLP_0"]}
    latent = trunk.apply({"params": trunk_params}, pixels)
    factored = jax.vmap(lambda p: head.apply({"params": p}, latent, proprio))(head_params)

    assert factored.shape == reference.shape == (n_skills, 6, action_dim)
    np.testing.assert_allclose(np.asarray(factored), np.asarray(reference), rtol=1e-6, atol=1e-6)


def test_build_rgb_actor_fns_layouts_and_equivalence():
    """The factory's two layouts, and that the shared one runs ONE CNN pass."""
    kwargs = dict(action_dim=2, action_scale=jnp.ones((2,)), action_bias=jnp.zeros((2,)),
                  hidden_sizes=(32,), embedding_dim=16)
    dummy_px = jnp.zeros((1, 20, 20, 3), dtype=jnp.float32)
    dummy_pr = jnp.zeros((1, 0), dtype=jnp.float32)
    px = jnp.zeros((7, 20, 20, 3), dtype=jnp.float32)
    pr = jnp.zeros((7, 0), dtype=jnp.float32)
    n_skills = 3

    unshared = build_rgb_actor_fns(**kwargs, aux_state_dim=4, shared_encoder=False)
    shared = build_rgb_actor_fns(**kwargs, aux_state_dim=4, shared_encoder=True)

    up = unshared.init(jax.random.PRNGKey(0), n_skills, dummy_px, dummy_pr)
    sp = shared.init(jax.random.PRNGKey(0), n_skills, dummy_px, dummy_pr)

    assert set(up) == {"RGBEncoder_0", "MLP_0", "aux_state"}
    assert set(sp) == {"encoder", "heads"}
    assert set(sp["encoder"]) == {"RGBEncoder_0", "aux_state"}
    assert set(sp["heads"]) == {"MLP_0"}

    assert unshared.apply(up, px, pr).shape == (n_skills, 7, 2)
    assert shared.apply(sp, px, pr).shape == (n_skills, 7, 2)

    # The aux prediction loses the redundant skill axis when the trunk is shared.
    _, up_pred = unshared.apply_aux(up, px, pr)
    _, sp_pred = shared.apply_aux(sp, px, pr)
    assert up_pred.shape == (n_skills, 7, 4) and unshared.aux_has_skill_axis
    assert sp_pred.shape == (7, 4) and not shared.aux_has_skill_axis

    # Latents: N redundant copies vs one.
    assert unshared.encode(up, px).shape == (n_skills, 7, 16)
    assert shared.encode(sp, px).shape == (7, 16)

    # Sharing is strictly smaller for the same config.
    n_up = sum(x.size for x in jax.tree_util.tree_leaves(up))
    n_sp = sum(x.size for x in jax.tree_util.tree_leaves(sp))
    assert n_sp < n_up, (n_sp, n_up)


def test_build_rgb_actor_fns_unshared_matches_legacy_init_and_apply():
    """The OFF path must be operation-for-operation the historical code.

    Same module, same ``jax.random.split(rng, num_skills)`` key consumption,
    same vmapped apply -> byte-identical parameters and outputs.
    """
    kwargs = dict(action_dim=2, action_scale=jnp.ones((2,)) * 3.0,
                  action_bias=jnp.zeros((2,)), hidden_sizes=(32, 32), embedding_dim=16)
    n_skills = 4
    dummy_px = jnp.zeros((1, 20, 20, 3), dtype=jnp.float32)
    dummy_pr = jnp.zeros((1, 2), dtype=jnp.float32)
    rng = jax.random.PRNGKey(1234)

    actor = VisionSkillActor(**kwargs, aux_state_dim=0)
    legacy = jax.vmap(lambda k: actor.init(k, dummy_px, dummy_pr)["params"])(
        jax.random.split(rng, n_skills)
    )
    fns = build_rgb_actor_fns(**kwargs, aux_state_dim=0, shared_encoder=False)
    new = fns.init(rng, n_skills, dummy_px, dummy_pr)

    assert jax.tree_util.tree_structure(legacy) == jax.tree_util.tree_structure(new)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), legacy, new)
    )

    px = jax.random.uniform(jax.random.PRNGKey(5), (6, 20, 20, 3), minval=-0.5, maxval=0.5)
    pr = jax.random.normal(jax.random.PRNGKey(6), (6, 2))
    legacy_out = jax.vmap(lambda p: actor.apply({"params": p}, px, pr))(legacy)
    assert bool(jnp.array_equal(legacy_out, fns.apply(new, px, pr)))
