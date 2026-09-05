import jax.numpy as jnp

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import (
    masked_meta_bootstrap_value,
    skill_actor_bootstrap_values,
)
from nexus_continuous.returns import q_lambda_returns


def test_q_lambda_returns_uses_next_timestep_values_and_final_bootstrap():
    rewards = jnp.zeros((3, 1), dtype=jnp.float32)
    dones = jnp.zeros((3, 1), dtype=bool)
    values = jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float32)
    last_value = jnp.asarray([40.0], dtype=jnp.float32)
    targets = q_lambda_returns(rewards, dones, values, last_value, gamma=1.0, lambda_=0.0)
    assert jnp.allclose(targets[:, 0], jnp.asarray([20.0, 30.0, 40.0]))


def test_skill_actor_bootstrap_values_shape_is_env_by_skill():
    num_skills = 3
    num_envs = 5
    num_critics = 2
    critic_params = jnp.arange(num_skills * num_critics, dtype=jnp.float32).reshape(
        num_skills,
        num_critics,
    )
    actor_params = jnp.arange(num_skills, dtype=jnp.float32)
    obs_actor = jnp.zeros((num_envs, 2), dtype=jnp.float32)
    obs_critic = jnp.ones((num_envs, 2), dtype=jnp.float32)

    def actor_apply(params, obs):
        del obs
        return jnp.broadcast_to(params[:, None, None], (num_skills, num_envs, 1))

    def critic_apply(param, obs, action):
        return param + obs[:, 0] + action[:, 0]

    values = skill_actor_bootstrap_values(
        critic_params,
        actor_params,
        obs_actor,
        obs_critic,
        actor_apply,
        critic_apply,
    )
    assert values.shape == (num_envs, num_skills)
    assert jnp.isfinite(values).all()


def test_meta_bootstrap_value_is_masked_max_not_selected_or_zeroed():
    q_values = jnp.asarray([[-5.0, -2.0, -1.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)
    mask = jnp.asarray([[True, False, False], [False, True, False]])
    value = masked_meta_bootstrap_value(q_values, mask)
    assert jnp.allclose(value, jnp.asarray([-5.0, 2.0]))
