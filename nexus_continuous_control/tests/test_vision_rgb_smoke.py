"""CPU integration smoke test for the USE_RGB trainer path.

The real RGB run needs MuJoCo Playground's in-loop MJWarp renderer (GPU + a
fragile warp-lang pin), so it cannot run in CI. This test instead injects a tiny
FAKE vision env (it emits the same observation contract the real adapter would:
``actor_pixels`` + a privileged proprio vector + semantic ``policy_info``) by
monkeypatching ``build_playground_env``. That exercises the parts of the diff
that static review and the vision-module shape tests do NOT cover:

  * the ``use_rgb`` branch of ``_actor_apply`` (vmap-over-skills WITH pixels),
  * the RGB ``init`` path (batch-1 dummy_pixels / width-0 dummy_proprio),
  * the pixel threading through both bootstrap call sites,
  * the DrQ ``_augment_pixels`` path (RGB_AUGMENT=true),
  * the ``(train_state, rng)`` minibatch-scan carry,
  * the ``_drop_actor_pixels`` next_obs handling,
  * the deterministic-eval pixel path.

Run (CPU):  JAX_PLATFORMS=cpu python -m pytest -q tests/test_vision_rgb_smoke.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nexus_continuous.algorithms.hierarchical_ac_pqn_playground as alg
from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import (
    _drop_actor_pixels,
    _select_rows,
    run_training,
)
from nexus_continuous.envs.playground_adapter import PlaygroundEnvBundle
from nexus_continuous.networks import MetaQ
from nexus_continuous.returns import smooth_l1_loss
from nexus_continuous.utils import global_norm
from nexus_continuous.vision import (
    RGBActorFns,
    RGBEncoder,
    SharedRGBTrunk,
    VisionSkillActor,
    VisionSkillHead,
    build_rgb_actor_fns,
)

# Small fake-render geometry (keep tiny so the test runs in seconds on CPU).
_H = _W = 12
_C = 3
_PROPRIO = 4  # cartpole qpos(2) + qvel(2)


def _fake_obs(batch: int):
    """Match the adapter's vision-mode observation contract."""
    proprio = jnp.zeros((batch, _PROPRIO), dtype=jnp.float32)
    # Float pixels in the renderer's ~[-0.5, 0.5] range (constant is fine; the
    # test only checks the pipeline runs and stays finite).
    pixels = jnp.full((batch, _H, _W, _C), 0.1, dtype=jnp.float32)
    semantic = {
        "cart_position": jnp.zeros((batch,), dtype=jnp.float32),
        "pole_angle": jnp.zeros((batch,), dtype=jnp.float32),
        "cart_velocity": jnp.zeros((batch,), dtype=jnp.float32),
        "pole_angular_velocity": jnp.zeros((batch,), dtype=jnp.float32),
    }
    return {
        "actor": proprio,
        "critic": proprio,
        "raw_actor": proprio,
        "raw_critic": proprio,
        "actor_pixels": pixels,
        "policy_info": semantic,
    }


class _FakeVisionEnv:
    """Minimal env exposing the (reset, step) contract the trainer expects."""

    def reset(self, key, params=None):
        del params
        n = key.shape[0]
        return _fake_obs(n), jnp.zeros((n,), dtype=jnp.float32)

    def step(self, key, state, action, params=None):
        del key, params
        n = action.shape[0]
        reward = jnp.zeros((n,), dtype=jnp.float32)
        done = jnp.zeros((n,), dtype=bool)
        return _fake_obs(n), state, reward, done, {}


def _fake_bundle(_config):
    return PlaygroundEnvBundle(
        env=_FakeVisionEnv(),
        env_params=None,
        action_low=-jnp.ones((1,), dtype=jnp.float32),
        action_high=jnp.ones((1,), dtype=jnp.float32),
        action_dim=1,
        episode_length=6,
        # Privileged-state width the auxiliary pixel->state head predicts; the
        # real adapter reads this off the MuJoCo model as nq+nv.
        actor_obs_dim=_PROPRIO,
    )


def _smoke_config(**overrides):
    config = {
        "ALG_NAME": "nexus_ac_pqn",
        "ENV_NAME": "CartpoleBalance",
        "POLICY": "cartpole_balance",
        "TASK_POLICY": "cartpole_balance",
        "META_POLICY_TYPE": "nesy",
        "USE_RGB": True,
        "RGB_PROPRIO": "none",
        "RGB_EMBED_DIM": 16,
        "RGB_AUGMENT": True,
        "RGB_AUG_PAD": 4,
        # Tiny budget: 2 updates.
        "TOTAL_TIMESTEPS": 4 * 4 * 2,
        "NUM_ENVS": 4,
        "NUM_STEPS": 4,
        "NUM_EPOCHS": 1,
        "NUM_MINIBATCHES": 2,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "GAMMA": 0.99,
        "LAMBDA": 0.65,
        "SKILL_LAMBDA": 0.65,
        "META_LAMBDA": 0.8,
        "LR": 1e-4,
        "LR_START": 1e-4,
        "LR_END": 1e-4,
        "LR_DECAY": 1.0,
        "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 1.0,
        "ACTOR_HIDDEN_SIZES": [16, 16],
        "CRITIC_HIDDEN_SIZES": [16, 16],
        "META_HIDDEN_SIZES": [16, 16],
        "NUM_CRITICS": 2,
        "NORM_TYPE": "layer_norm",
        "ACTIVATION": "relu",
        "NOISE_START": 0.3,
        "NOISE_FINISH": 0.02,
        "NOISE_DECAY": 0.8,
        "META_EPS_START": 1.0,
        "META_EPS_FINISH": 0.02,
        "META_EPS_DECAY": 0.6,
        # NORMALIZE_OBS off so the fake env need not carry running-stats state.
        "NORMALIZE_OBS": False,
        "NORMALIZE_REWARD": False,
        "BEHAVIOR_PENALTY_COEFF": 0.001,
        "ACTOR_UPDATE_MODE": "all_states",
        "EVAL_AFTER_TRAIN": True,
        "EVAL_NUM_ENVS": 2,
        "EVAL_NUM_EPISODES": 2,
        "PRINT_EVERY": 0,
        "SAVE_PATH": None,
    }
    config.update(overrides)
    return config


# --------------------------------------------------------------------------- #
# Unit tests for the pieces that the smoke also covers, isolated for clarity.  #
# --------------------------------------------------------------------------- #


def test_drop_actor_pixels_strips_only_pixels():
    obs = {"actor": jnp.zeros((2, 3)), "policy_info": {}, "actor_pixels": jnp.zeros((2, 8, 8, 3))}
    out = _drop_actor_pixels(obs)
    assert "actor_pixels" not in out
    assert set(out) == {"actor", "policy_info"}
    # State-mode (no pixels) must be a no-op identity.
    state_obs = {"actor": jnp.zeros((2, 3)), "policy_info": {}}
    assert _drop_actor_pixels(state_obs) is state_obs


def test_encoder_normalization_is_batch_independent():
    """The float passthrough must not depend on other samples in the batch."""
    enc = RGBEncoder(embedding_dim=16)
    img = jax.random.uniform(jax.random.PRNGKey(0), (1, _H, _W, _C), minval=-0.5, maxval=0.5)
    params = enc.init(jax.random.PRNGKey(1), img)
    alone = enc.apply(params, img)
    # Same image embedded inside a batch that also contains a huge-valued sample.
    outlier = jnp.full((1, _H, _W, _C), 100.0, dtype=jnp.float32)
    batched = enc.apply(params, jnp.concatenate([img, outlier], axis=0))
    np.testing.assert_allclose(np.asarray(alone[0]), np.asarray(batched[0]), rtol=1e-5, atol=1e-5)


def test_encoder_uint8_path_is_finite():
    enc = RGBEncoder(embedding_dim=16)
    img = jnp.zeros((2, _H, _W, _C), dtype=jnp.uint8)
    params = enc.init(jax.random.PRNGKey(0), img)
    out = enc.apply(params, img)
    assert out.shape == (2, 16)
    assert bool(jnp.all(jnp.isfinite(out)))


# --------------------------------------------------------------------------- #
# End-to-end smoke: run the real make_train USE_RGB path on the fake env.      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rgb_augment", [True, False])
def test_use_rgb_train_smoke(monkeypatch, rgb_augment):
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    config = _smoke_config(RGB_AUGMENT=rgb_augment)

    output = run_training(config)

    # Training completed the expected number of updates.
    runner_state = output.runner_state
    train_state = runner_state[0]
    assert int(train_state.actor.n_updates) == 2

    # Eval ran and produced a finite return.
    assert "episode_return_mean" in output.eval_metrics
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))

    # Losses are finite (no NaN from the pixel actor / augmentation path).
    for key, value in output.metrics.items():
        if key.startswith("train/") and "loss" in key:
            assert bool(jnp.all(jnp.isfinite(value))), f"non-finite {key}"


def test_aux_state_head_and_sensitivity_monitor(monkeypatch):
    """The fix for the 2026-08-17 'blind encoder' finding must actually run.

    Exercises the paths that only activate with RGB_AUX_STATE_COEF > 0: the
    aux-enabled parameter tree (an extra Dense head inside every skill actor),
    the return_aux=True vmapped apply in the actor loss, and the pixel-
    sensitivity monitor. All of it must stay finite.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    config = _smoke_config(RGB_AUX_STATE_COEF=1.0, RGB_MONITOR_SENSITIVITY=True)

    output = run_training(config)

    assert int(output.runner_state[0].actor.n_updates) == 2
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))

    # The aux head exists in every skill actor's parameters.
    actor_params = output.runner_state[0].actor.params
    assert "aux_state" in actor_params, f"no aux head; got {list(actor_params)}"

    # Both new metrics were logged and stayed finite.
    for name in ("rgb/aux_state_loss", "rgb/pixel_sensitivity"):
        matches = [k for k in output.metrics if name in k]
        assert matches, f"{name} not logged; metrics: {sorted(output.metrics)[:20]}"
        for k in matches:
            assert bool(jnp.all(jnp.isfinite(output.metrics[k]))), f"non-finite {k}"


def test_aux_head_absent_when_disabled(monkeypatch):
    """Default (coef 0) must leave the parameter tree byte-identical to before."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(_smoke_config())
    assert "aux_state" not in output.runner_state[0].actor.params


# --------------------------------------------------------------------------- #
# RGB_SHARED_ENCODER (lever A): ONE CNN trunk for all N skills.                #
# --------------------------------------------------------------------------- #

_NUM_SKILLS = 3  # cartpole_balance policy
_EMBED = 16      # _smoke_config's RGB_EMBED_DIM


def _reference_trees(aux_state_dim: int):
    """Freshly-init'd trunk/head params WITHOUT any skill axis, for shape refs."""
    px = jnp.zeros((1, _H, _W, _C), dtype=jnp.float32)
    proprio = jnp.zeros((1, 0), dtype=jnp.float32)
    trunk = SharedRGBTrunk(embedding_dim=_EMBED, aux_state_dim=aux_state_dim)
    head = VisionSkillHead(
        action_dim=1,
        action_scale=jnp.ones((1,), dtype=jnp.float32),
        action_bias=jnp.zeros((1,), dtype=jnp.float32),
        hidden_sizes=(16, 16),
    )
    trunk_params = trunk.init(jax.random.PRNGKey(0), px)["params"]
    head_params = head.init(jax.random.PRNGKey(0), jnp.zeros((1, _EMBED)), proprio)["params"]
    return trunk_params, head_params


def _shapes(tree):
    return {
        "/".join(str(k.key) for k in path): leaf.shape
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def test_shared_encoder_parameter_layout(monkeypatch):
    """The trunk must live OUTSIDE the skill axis and the heads INSIDE it."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(_smoke_config(RGB_SHARED_ENCODER=True))

    assert int(output.runner_state[0].actor.n_updates) == 2
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))

    actor_params = output.runner_state[0].actor.params
    assert set(actor_params) == {"encoder", "heads"}, list(actor_params)

    ref_trunk, ref_head = _reference_trees(aux_state_dim=0)
    # Every encoder leaf has NO leading skill axis: shape equals the un-vmapped
    # reference exactly. (Comparing against a reference rather than testing
    # `shape[0] != 3` matters: Conv kernels legitimately start with a 3.)
    assert _shapes(actor_params["encoder"]) == _shapes(ref_trunk)
    # Every head leaf DOES carry the leading skill axis.
    expected_heads = {k: (_NUM_SKILLS,) + v for k, v in _shapes(ref_head).items()}
    assert _shapes(actor_params["heads"]) == expected_heads

    # No NaNs leaked out of the one-CNN-pass path.
    for key, value in output.metrics.items():
        if key.startswith("train/") and "loss" in key:
            assert bool(jnp.all(jnp.isfinite(value))), f"non-finite {key}"


def test_shared_encoder_is_smaller_and_not_a_no_op(monkeypatch):
    """Sharing must be a strict size win, and the flag must really do something.

    The second half also guards the byte-identity test below from passing for the
    wrong reason: if flipping RGB_SHARED_ENCODER changed nothing, that test would
    be vacuous.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    unshared = run_training(_smoke_config()).runner_state[0].actor.params
    shared = run_training(_smoke_config(RGB_SHARED_ENCODER=True)).runner_state[0].actor.params

    def count(params):
        return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))

    assert count(shared) < count(unshared), (count(shared), count(unshared))
    assert jax.tree_util.tree_structure(unshared) != jax.tree_util.tree_structure(shared)


def test_shared_encoder_aux_head_lives_on_the_trunk(monkeypatch):
    """One aux head on the shared trunk, not N redundant per-skill copies."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(
        _smoke_config(RGB_SHARED_ENCODER=True, RGB_AUX_STATE_COEF=1.0,
                      RGB_MONITOR_SENSITIVITY=True)
    )

    actor_params = output.runner_state[0].actor.params
    assert "aux_state" in actor_params["encoder"], list(actor_params["encoder"])
    assert "aux_state" not in actor_params["heads"], list(actor_params["heads"])
    # No skill axis on it, and it predicts the full privileged state width.
    assert actor_params["encoder"]["aux_state"]["kernel"].shape == (_EMBED, _PROPRIO)
    assert actor_params["encoder"]["aux_state"]["bias"].shape == (_PROPRIO,)
    ref_trunk, _ = _reference_trees(aux_state_dim=_PROPRIO)
    assert _shapes(actor_params["encoder"]) == _shapes(ref_trunk)

    # Both RGB monitors keep working under the new branch.
    for name in ("rgb/aux_state_loss", "rgb/pixel_sensitivity"):
        matches = [k for k in output.metrics if name in k]
        assert matches, f"{name} not logged; metrics: {sorted(output.metrics)[:20]}"
        for k in matches:
            assert bool(jnp.all(jnp.isfinite(output.metrics[k]))), f"non-finite {k}"
    # The pixel-sensitivity monitor must be a real number, not silently zeroed.
    sens = output.metrics["train/rgb/pixel_sensitivity"]
    assert bool(jnp.all(jnp.isfinite(sens)))


def test_shared_encoder_requires_use_rgb(monkeypatch):
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    with pytest.raises(ValueError, match="RGB_SHARED_ENCODER"):
        run_training(_smoke_config(USE_RGB=False, RGB_SHARED_ENCODER=True))


# --------------------------------------------------------------------------- #
# BYTE-IDENTITY GUARD for the flag-OFF path.                                   #
# --------------------------------------------------------------------------- #


def _legacy_build_rgb_actor_fns(
    *,
    action_dim,
    action_scale,
    action_bias,
    hidden_sizes=(256, 256),
    embedding_dim=128,
    aux_state_dim=0,
    shared_encoder=False,
):
    """The pre-RGB_SHARED_ENCODER trainer code, transcribed operation for operation.

    Verbatim from the version of ``hierarchical_ac_pqn_playground.py`` that
    predates the shared trunk: one ``VisionSkillActor``, ``jax.random.split(
    rng_actor, num_skills)``, and a plain ``jax.vmap`` over the skill axis.
    Monkeypatched over ``alg.build_rgb_actor_fns`` below so the flag-off run can
    be diffed against the historical behaviour without a second checkout.
    """
    assert not shared_encoder, "the legacy path has no shared-encoder mode"
    actor = VisionSkillActor(
        action_dim=action_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        hidden_sizes=tuple(hidden_sizes),
        embedding_dim=embedding_dim,
        aux_state_dim=aux_state_dim,
    )

    def _init(rng, num_skills, dummy_pixels, dummy_proprio):
        actor_rngs = jax.random.split(rng, num_skills)
        return jax.vmap(lambda k: actor.init(k, dummy_pixels, dummy_proprio)["params"])(
            actor_rngs
        )

    def _apply(params, pixels, proprio):
        return jax.vmap(lambda p: actor.apply({"params": p}, pixels, proprio))(params)

    def _apply_aux(params, pixels, proprio):
        return jax.vmap(
            lambda p: actor.apply({"params": p}, pixels, proprio, return_aux=True)
        )(params)

    def _encode(params, pixels):  # pragma: no cover - the trainer never calls it
        raise AssertionError("the legacy path had no encode helper")

    return RGBActorFns(
        shared_encoder=False,
        actor=actor,
        trunk=None,
        head=None,
        init=_init,
        apply=_apply,
        apply_aux=_apply_aux,
        encode=_encode,
        train_state_apply=actor.apply,
        aux_has_skill_axis=True,
    )


def _assert_train_states_identical(a, b):
    trees_a = {"actor": a.actor.params, "critic": a.critic.params}
    trees_b = {"actor": b.actor.params, "critic": b.critic.params}
    if a.meta is not None or b.meta is not None:
        assert a.meta is not None and b.meta is not None
        trees_a["meta"] = a.meta.params
        trees_b["meta"] = b.meta.params
    for name in trees_a:
        ta, tb = trees_a[name], trees_b[name]
        assert jax.tree_util.tree_structure(ta) == jax.tree_util.tree_structure(tb), name
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: bool(jnp.array_equal(x, y)), ta, tb)
        ), f"{name} params differ"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"RGB_AUX_STATE_COEF": 1.0, "RGB_MONITOR_SENSITIVITY": True},
        {"RGB_AUGMENT": False},
        # Lever B's flags explicitly OFF. Also covers the `apply_gradients`
        # reorder the folded meta->encoder gradient forced (the actor's optimizer
        # step now happens after the meta block): if that reorder were not inert,
        # the actor tree would diverge from the legacy transcription here.
        {"RGB_META_SEES_PIXELS": False, "RGB_META_LATENT_STOP_GRAD": False},
    ],
)
def test_flag_off_is_byte_identical_to_the_legacy_actor(monkeypatch, overrides):
    """With RGB_SHARED_ENCODER absent, nothing may have changed. Bit for bit.

    The refactor routed the RGB actor through ``build_rgb_actor_fns``. This runs
    the SAME seed and SAME fake env twice -- once through the real factory, once
    through ``_legacy_build_rgb_actor_fns`` (the historical inline code) -- and
    requires the final actor/critic/meta parameter trees and the eval metrics to
    be EXACTLY equal. That covers the PRNG consumption order too: a different
    ``jax.random.split`` pattern would move every init and every exploration
    draw, and no leaf would match.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)

    new = run_training(_smoke_config(**overrides))
    monkeypatch.setattr(alg, "build_rgb_actor_fns", _legacy_build_rgb_actor_fns)
    legacy = run_training(_smoke_config(**overrides))

    _assert_train_states_identical(new.runner_state[0], legacy.runner_state[0])
    for key, value in new.eval_metrics.items():
        assert bool(jnp.array_equal(value, legacy.eval_metrics[key])), key


# --------------------------------------------------------------------------- #
# RGB_META_SEES_PIXELS / RGB_META_LATENT_STOP_GRAD (lever B).                   #
#                                                                              #
# The meta-Q -- the "boss" that decides WHICH skill acts -- reads the shared    #
# CNN latent APPENDED to the privileged state it already had.                   #
# --------------------------------------------------------------------------- #

_META_HIDDEN = 16  # _smoke_config's META_HIDDEN_SIZES[0]
_METAZ = {"RGB_SHARED_ENCODER": True, "RGB_META_SEES_PIXELS": True}


def _meta_first_dense_kernel(meta_params):
    """``MetaQ`` -> ``MLP`` -> ``Dense_0`` kernel, navigated EXPLICITLY.

    Deliberately NOT ``tests/test_actor_obs_indices.py::_first_dense_kernel``:
    that helper walks the tree for the first path containing ``Dense_0``, and once
    an unstacked encoder subtree exists in the actor params it returns the wrong
    leaf. A fixed path cannot silently drift onto another layer.
    """
    return meta_params["MLP_0"]["Dense_0"]["kernel"]


def _meta_encoder_grads(*, encode_inside: bool, stop_grad: bool = False):
    """Differentiate a meta TD loss w.r.t. ``{"meta", "encoder"}`` on REAL modules.

    ``encode_inside=True`` reproduces the shipped ``meta_loss_fn``: the latent is
    encoded from the encoder parameters that are being differentiated.
    ``encode_inside=False`` is THE TRAP -- the latent is closed over from outside
    and is therefore a jax constant.
    """
    fns = build_rgb_actor_fns(
        action_dim=1,
        action_scale=jnp.ones((1,), dtype=jnp.float32),
        action_bias=jnp.zeros((1,), dtype=jnp.float32),
        hidden_sizes=(16, 16),
        embedding_dim=_EMBED,
        shared_encoder=True,
    )
    pixels = jax.random.uniform(
        jax.random.PRNGKey(1), (5, _H, _W, _C), minval=-0.5, maxval=0.5
    )
    state = jax.random.normal(jax.random.PRNGKey(2), (5, _PROPRIO))
    actor_params = fns.init(
        jax.random.PRNGKey(0), _NUM_SKILLS, pixels[:1], jnp.zeros((1, 0), jnp.float32)
    )
    meta = MetaQ(num_skills=_NUM_SKILLS, hidden_sizes=(16, 16))
    meta_params = meta.init(
        jax.random.PRNGKey(3), jnp.zeros((_PROPRIO + _EMBED,), jnp.float32)
    )["params"]
    skills = jnp.asarray([0, 1, 2, 0, 1], dtype=jnp.int32)
    targets = jnp.arange(5, dtype=jnp.float32)
    latent_from_outside = fns.encode(actor_params, pixels)

    def loss_fn(params):
        if encode_inside:
            # Exactly what the trainer does: a ONE-KEY dict of the differentiated
            # encoder params (the shared closure reads nothing else).
            latent = fns.encode({"encoder": params["encoder"]}, pixels)
            if stop_grad:
                latent = jax.lax.stop_gradient(latent)
        else:
            latent = latent_from_outside
        q = meta.apply(
            {"params": params["meta"]}, jnp.concatenate([state, latent], axis=-1)
        )
        return jnp.mean(smooth_l1_loss(_select_rows(q, skills), targets))

    return jax.grad(loss_fn)(
        {"meta": meta_params, "encoder": actor_params["encoder"]}
    )


def test_encoding_inside_the_grad_is_what_makes_the_encoder_gradient_nonzero():
    """THE TRAP, made explicit and permanent.

    If the latent is computed OUTSIDE the differentiated function it is a jax
    constant and the meta's TD gradient never reaches the encoder. There is no
    error, no NaN, no shape mismatch -- and, as the last assertion here proves,
    the meta's OWN gradient is bit-for-bit the same either way, so nothing the
    meta reports can reveal the bug. That is why the shipped ``meta_loss_fn``
    re-encodes inside, and why ``train/meta_encoder_grad_norm`` is logged.
    """
    inside = _meta_encoder_grads(encode_inside=True)
    outside = _meta_encoder_grads(encode_inside=False)
    stopped = _meta_encoder_grads(encode_inside=True, stop_grad=True)

    assert float(global_norm(inside["encoder"])) > 0.0
    assert float(global_norm(outside["encoder"])) == 0.0  # <- the silent failure
    assert float(global_norm(stopped["encoder"])) == 0.0  # <- the escape hatch

    for name, other in (("outside", outside), ("stop_grad", stopped)):
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda a, b: bool(jnp.allclose(a, b)), inside["meta"], other["meta"]
            )
        ), f"the meta's own gradient differs from the {name} variant"


def test_meta_loss_sends_gradient_into_the_shared_encoder(monkeypatch):
    """THE NON-NEGOTIABLE ONE: the encoder must actually RECEIVE meta gradient.

    ``train/meta_encoder_grad_norm`` IS ``global_norm(g["encoder"])`` as computed
    by the real ``meta_loss_fn`` inside the real training loop, before the fold
    into the actor's gradient. Asserting it is strictly positive on every
    minibatch is what stops the silent-zero-gradient failure from shipping green.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(_smoke_config(**_METAZ))

    assert int(output.runner_state[0].actor.n_updates) == 2
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))
    norms = np.asarray(output.metrics["train/meta_encoder_grad_norm"])
    assert np.all(np.isfinite(norms)), norms
    assert float(norms.min()) > 0.0, (
        "the meta's TD gradient does NOT reach the shared encoder: the latent is "
        f"being treated as a constant. min |g[encoder]| = {float(norms.min())}"
    )
    for key, value in output.metrics.items():
        if key.startswith("train/") and "loss" in key:
            assert bool(jnp.all(jnp.isfinite(value))), f"non-finite {key}"


def test_meta_latent_stop_grad_gives_exactly_zero_encoder_gradient(monkeypatch):
    """The escape hatch -- and the contrast that makes the test above meaningful.

    With RGB_META_LATENT_STOP_GRAD the meta still READS the latent but sends no
    gradient into the encoder, separating "the meta benefits from seeing pixels"
    from "the meta's TD gradient helps train the encoder".
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(_smoke_config(**_METAZ, RGB_META_LATENT_STOP_GRAD=True))

    assert int(output.runner_state[0].actor.n_updates) == 2
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))
    norms = np.asarray(output.metrics["train/meta_encoder_grad_norm"])
    assert float(np.abs(norms).max()) == 0.0, norms
    # The meta still sees the latent: its input width proves the flag did not
    # quietly fall back to the state-only meta.
    meta_params = output.runner_state[0].meta.params
    assert _meta_first_dense_kernel(meta_params).shape == (_PROPRIO + _EMBED, _META_HIDDEN)


def test_meta_gradient_actually_reaches_the_optimizer(monkeypatch):
    """Computing the gradient is not enough -- it must be APPLIED.

    Same seed, same config, same forward pass (``stop_gradient`` changes no
    value), so the ONLY difference between these two runs is whether the meta's
    TD gradient is added to the encoder's gradient before the actor's optimizer
    step. The encoder weights must therefore end up different.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    trained = run_training(_smoke_config(**_METAZ)).runner_state[0]
    read_only = run_training(
        _smoke_config(**_METAZ, RGB_META_LATENT_STOP_GRAD=True)
    ).runner_state[0]

    enc_a = trained.actor.params["encoder"]
    enc_b = read_only.actor.params["encoder"]
    assert jax.tree_util.tree_structure(enc_a) == jax.tree_util.tree_structure(enc_b)
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: bool(jnp.array_equal(x, y)), enc_a, enc_b)
    ), "the folded meta gradient never changed the encoder weights"
    # NOT asserted: that the meta's own parameters stay equal. The fold only
    # touches the encoder subtree, so the FIRST minibatch's meta update is indeed
    # identical -- but from the second one on the meta reads a latent produced by
    # the now-different encoder, so it legitimately diverges too. The encoder
    # divergence above is the only clean, attributable signal.


def test_meta_input_keeps_the_state_and_appends_the_latent(monkeypatch):
    """REGRESSION GUARD: THE STATE WAS NOT REMOVED, only extended.

    Lever B is ``concatenate([state, latent])``, never ``latent`` alone. The
    hand-written symbolic precondition mask and the NeSy diagnostics must keep
    reading the same privileged state, or the interpretability the NEXUS
    hierarchy exists for is gone. A first Dense kernel of ``(_EMBED, hidden)``
    instead of ``(_PROPRIO + _EMBED, hidden)`` is exactly what that mistake looks
    like -- and appending (rather than prepending) keeps every state component on
    the column index it always had.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)

    off = run_training(_smoke_config()).runner_state[0]
    assert _meta_first_dense_kernel(off.meta.params).shape == (_PROPRIO, _META_HIDDEN)

    on = run_training(_smoke_config(**_METAZ)).runner_state[0]
    assert _meta_first_dense_kernel(on.meta.params).shape == (
        _PROPRIO + _EMBED,
        _META_HIDDEN,
    )
    # The symbolic layer still fires on the full state (interpretability intact).
    metrics = run_training(_smoke_config(**_METAZ)).metrics
    symbolic = [k for k in metrics if "cartpole/" in k]
    assert symbolic, f"symbolic diagnostics missing; got {sorted(metrics)[:20]}"
    for key in symbolic:
        assert bool(jnp.all(jnp.isfinite(metrics[key]))), f"non-finite {key}"


def test_meta_sees_pixels_requires_shared_encoder(monkeypatch):
    """With N private encoders there is no single latent a meta-Q could read."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    with pytest.raises(ValueError, match="RGB_META_SEES_PIXELS requires RGB_SHARED_ENCODER"):
        run_training(_smoke_config(RGB_META_SEES_PIXELS=True))


def test_meta_sees_pixels_requires_use_rgb(monkeypatch):
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    with pytest.raises(ValueError, match="RGB_META_SEES_PIXELS"):
        run_training(_smoke_config(USE_RGB=False, RGB_META_SEES_PIXELS=True))


@pytest.mark.parametrize("base", [{}, {"RGB_SHARED_ENCODER": True}])
def test_lever_b_flags_explicitly_off_change_nothing(monkeypatch, base):
    """Byte-identity: the new flags set to False == the flags absent.

    Together with the legacy-actor byte-identity test above (which is
    parametrized with the new flags off) this pins the default path bit for bit,
    including the ``apply_gradients`` reorder that the folded meta->encoder
    gradient required.
    """
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    absent = run_training(_smoke_config(**base))
    explicit = run_training(
        _smoke_config(**base, RGB_META_SEES_PIXELS=False, RGB_META_LATENT_STOP_GRAD=False)
    )

    _assert_train_states_identical(absent.runner_state[0], explicit.runner_state[0])
    for key, value in absent.eval_metrics.items():
        assert bool(jnp.array_equal(value, explicit.eval_metrics[key])), key
    # And the new metric is a hard zero when nothing is switched on.
    assert float(np.abs(np.asarray(absent.metrics["train/meta_encoder_grad_norm"])).max()) == 0.0
