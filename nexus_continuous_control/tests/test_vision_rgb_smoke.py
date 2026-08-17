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
    run_training,
)
from nexus_continuous.envs.playground_adapter import PlaygroundEnvBundle
from nexus_continuous.vision import RGBEncoder

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
