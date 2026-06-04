"""Return utilities for the no-replay-buffer PQN update."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def smooth_l1_loss(pred: jnp.ndarray, target: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """Huber / smooth-L1 loss used by AC-PQN critics."""

    diff = pred - target
    abs_diff = jnp.abs(diff)
    return jnp.where(abs_diff < beta, 0.5 * (diff**2) / beta, abs_diff - 0.5 * beta)


def q_lambda_returns(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    values: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    lambda_: float,
) -> jnp.ndarray:
    """Compute Q(lambda) targets by scanning backwards along time.

    Args:
      rewards: array [T, E, ...].
      dones: array [T, E] or [T, E, 1].
      values: greedy/bootstrap Q-values [T, E, ...]. Target at timestep t uses
        values[t + 1] as the one-step bootstrap and `last_value` at the final
        timestep, matching AC-PQN/PQN.
      last_value: bootstrap value [E, ...] for the final state.
      gamma: discount factor.
      lambda_: trace mixing coefficient.

    Returns:
      Lambda targets with shape [T, E, ...].
    """

    if dones.ndim < rewards.ndim:
        dones_expanded = jnp.expand_dims(dones, axis=-1)
    else:
        dones_expanded = dones

    next_values = jnp.concatenate([values[1:], last_value[jnp.newaxis]], axis=0)

    def _step(lambda_return, transition):
        reward_t, done_t, next_q = transition
        target_bootstrap = reward_t + gamma * (1.0 - done_t) * next_q
        delta = lambda_return - next_q
        lambda_return = target_bootstrap + gamma * lambda_ * (1.0 - done_t) * delta
        lambda_return = jnp.where(done_t.astype(bool), reward_t, lambda_return)
        return lambda_return, lambda_return

    _, targets = jax.lax.scan(
        _step,
        last_value,
        (rewards, dones_expanded, next_values),
        reverse=True,
    )
    return targets
