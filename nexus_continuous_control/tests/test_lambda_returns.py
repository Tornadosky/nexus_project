import jax.numpy as jnp

from nexus_continuous.returns import q_lambda_returns


def test_lambda_returns_shape_and_terminal():
    rewards = jnp.ones((3, 2, 4))
    dones = jnp.array([[False, False], [False, True], [False, False]])
    values = jnp.zeros((3, 2, 4))
    last = jnp.zeros((2, 4))
    targets = q_lambda_returns(rewards, dones, values, last, gamma=0.99, lambda_=0.65)
    assert targets.shape == rewards.shape
    assert jnp.allclose(targets[1, 1], 1.0)
