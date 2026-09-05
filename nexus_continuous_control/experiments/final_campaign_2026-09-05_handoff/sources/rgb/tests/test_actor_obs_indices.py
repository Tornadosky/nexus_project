"""CPU integration test for ACTOR_OBS_INDICES (the "restricted state" actor).

This is arm (B) of the actor-information ablation:

  (A) actor sees the FULL privileged state          -- default, no config key
  (B) actor sees a RESTRICTED subset of the state   -- ACTOR_OBS_INDICES
  (C) actor sees camera pixels                      -- USE_RGB (tested in
      tests/test_vision_rgb_smoke.py)

In all three the meta-Q, the critics, the symbolic rules and the skill rewards
must keep the FULL privileged state, so the actor's observation is the only
variable. That is the property this file pins down, because the trainer feeds
the SAME array (``get_actor_obs(obs)``) to both the skill actors and the
meta-Q: a naive restriction would silently blind the meta-policy too.

Like tests/test_vision_rgb_smoke.py this injects a tiny FAKE env by
monkeypatching ``build_playground_env``, so it runs in seconds on CPU.

Run (CPU):  JAX_PLATFORMS=cpu python -m pytest -q tests/test_actor_obs_indices.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

import nexus_continuous.algorithms.hierarchical_ac_pqn_playground as alg
from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.envs.playground_adapter import PlaygroundEnvBundle

# 8-dim toy state: [0:4] = "robot proprioception", [4:8] = privileged extras
# (stand-in for the manipulated object's pose the blind baseline must not see).
_STATE = 8
_PROPRIO_IDX = [0, 1, 2, 3]
_NUM_SKILLS = 3  # cartpole_balance policy


def _fake_obs(batch: int):
    """State-mode observation contract (no ``actor_pixels`` key)."""
    state = jnp.zeros((batch, _STATE), dtype=jnp.float32)
    # Semantic channel so the symbolic rules/masks/rewards never depend on the
    # raw-vector layout -- they must be unaffected by the actor restriction.
    semantic = {
        "cart_position": jnp.zeros((batch,), dtype=jnp.float32),
        "pole_angle": jnp.zeros((batch,), dtype=jnp.float32),
        "cart_velocity": jnp.zeros((batch,), dtype=jnp.float32),
        "pole_angular_velocity": jnp.zeros((batch,), dtype=jnp.float32),
    }
    return {
        "actor": state,
        "critic": state,
        "raw_actor": state,
        "raw_critic": state,
        "policy_info": semantic,
    }


class _FakeStateEnv:
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
        env=_FakeStateEnv(),
        env_params=None,
        action_low=-jnp.ones((1,), dtype=jnp.float32),
        action_high=jnp.ones((1,), dtype=jnp.float32),
        action_dim=1,
        episode_length=6,
        actor_obs_dim=_STATE,
    )


def _smoke_config(**overrides):
    config = {
        "ALG_NAME": "nexus_ac_pqn",
        "ENV_NAME": "CartpoleBalance",
        "POLICY": "cartpole_balance",
        "TASK_POLICY": "cartpole_balance",
        "META_POLICY_TYPE": "nesy",
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


def _first_dense_kernel(params):
    """Return the ``Dense_0`` kernel of the first MLP inside ``params``."""
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        names = [getattr(k, "key", None) for k in path]
        if "Dense_0" in names and names[-1] == "kernel":
            return leaf
    raise AssertionError(f"no Dense_0 kernel in {jax.tree_util.tree_structure(params)}")


def _assert_finite_losses(output):
    for key, value in output.metrics.items():
        if key.startswith("train/") and "loss" in key:
            assert bool(jnp.all(jnp.isfinite(value))), f"non-finite {key}"


# --------------------------------------------------------------------------- #


def test_actor_obs_indices_restricts_actor_only(monkeypatch):
    """(B) Restricted actor: training runs, actor is narrow, meta stays full."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(_smoke_config(ACTOR_OBS_INDICES=_PROPRIO_IDX))

    train_state = output.runner_state[0]
    assert int(train_state.actor.n_updates) == 2

    # (1) End-to-end and finite.
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))
    _assert_finite_losses(output)

    # (2) The restriction reached the network: actor first layer is [N, 4, 16].
    actor_kernel = _first_dense_kernel(train_state.actor.params)
    assert actor_kernel.shape == (_NUM_SKILLS, len(_PROPRIO_IDX), 16), actor_kernel.shape

    # The critical one: the meta-Q is fed the SAME get_actor_obs array, and it
    # must still be the FULL 8-dim privileged state.
    assert train_state.meta is not None
    meta_kernel = _first_dense_kernel(train_state.meta.params)
    assert meta_kernel.shape == (_STATE, 16), meta_kernel.shape

    # Critics keep the full state too (obs + 1 action dim).
    critic_kernel = _first_dense_kernel(train_state.critic.params)
    assert critic_kernel.shape == (_NUM_SKILLS, 2, _STATE + 1, 16), critic_kernel.shape


def test_default_keeps_full_actor_width(monkeypatch):
    """(A) No regression: without the option the actor keeps the full state."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    output = run_training(_smoke_config())

    train_state = output.runner_state[0]
    assert int(train_state.actor.n_updates) == 2
    assert bool(jnp.isfinite(output.eval_metrics["episode_return_mean"]))
    _assert_finite_losses(output)

    actor_kernel = _first_dense_kernel(train_state.actor.params)
    assert actor_kernel.shape == (_NUM_SKILLS, _STATE, 16), actor_kernel.shape
    meta_kernel = _first_dense_kernel(train_state.meta.params)
    assert meta_kernel.shape == (_STATE, 16), meta_kernel.shape


def test_symbolic_layer_unaffected_by_restriction(monkeypatch):
    """The NeSy diagnostics (rules/masks over the full state) still fire."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    restricted = run_training(_smoke_config(ACTOR_OBS_INDICES=_PROPRIO_IDX))
    keys = [k for k in restricted.metrics if "cartpole/" in k]
    assert keys, f"symbolic diagnostics missing; got {sorted(restricted.metrics)[:20]}"
    for k in keys:
        assert bool(jnp.all(jnp.isfinite(restricted.metrics[k]))), f"non-finite {k}"


@pytest.mark.parametrize(
    "bad,message",
    [
        ([], "non-empty"),
        ([-1, 0], "non-negative"),
        ([0, 99], "only has 8 components"),
    ],
)
def test_invalid_indices_raise(monkeypatch, bad, message):
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    with pytest.raises(ValueError, match=message):
        run_training(_smoke_config(ACTOR_OBS_INDICES=bad))


def test_rgb_mode_rejects_actor_obs_indices(monkeypatch):
    """USE_RGB has its own knob; combining the two must fail loudly, not silently."""
    monkeypatch.setattr(alg, "build_playground_env", _fake_bundle)
    with pytest.raises(ValueError, match="RGB_PROPRIO"):
        run_training(_smoke_config(USE_RGB=True, ACTOR_OBS_INDICES=_PROPRIO_IDX))
