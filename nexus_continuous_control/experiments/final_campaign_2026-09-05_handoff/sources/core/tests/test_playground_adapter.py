import jax
import jax.numpy as jnp

from nexus_continuous.envs.playground_adapter import _LogVecWrapper


class _NonFiniteRewardEnv:
    def reset(self, key, params=None):
        del key, params
        return {"actor": jnp.zeros((4, 1), dtype=jnp.float32)}, jnp.asarray(0)

    def step(self, key, state, action, params=None):
        del key, action, params
        reward = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, 1.0], dtype=jnp.float32)
        done = jnp.asarray([False, True, False, True])
        obs = {"actor": jnp.zeros((4, 1), dtype=jnp.float32)}
        return obs, state + 1, reward, done, {}


def test_log_wrapper_sanitizes_nonfinite_rewards_before_return_accumulation():
    wrapper = _LogVecWrapper(_NonFiniteRewardEnv())
    _obs, state = wrapper.reset(jax.random.PRNGKey(0))

    _obs, next_state, reward, _done, info = wrapper.step(
        jax.random.PRNGKey(1),
        state,
        jnp.zeros((4, 1), dtype=jnp.float32),
    )

    assert jnp.isfinite(reward).all()
    assert jnp.allclose(reward, jnp.asarray([0.0, 0.0, 0.0, 1.0]))
    assert jnp.allclose(info["nonfinite_reward"], jnp.asarray([1.0, 1.0, 1.0, 0.0]))
    assert jnp.isfinite(info["returned_episode_returns"]).all()
    assert jnp.isfinite(next_state.episode_returns).all()
