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
      values: behavior-policy Q-values [T, E, ...]. The next_q used for each
        previous step is the following transition's value, matching AC-PQN/PQN.
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

    def _step(carry, transition):
        lambda_return, next_q = carry
        reward_t, done_t, value_t = transition
        target_bootstrap = reward_t + gamma * (1.0 - done_t) * next_q
        delta = lambda_return - next_q
        lambda_return = target_bootstrap + gamma * lambda_ * (1.0 - done_t) * delta
        lambda_return = jnp.where(done_t.astype(bool), reward_t, lambda_return)
        next_q = value_t
        return (lambda_return, next_q), lambda_return

    final_target = rewards[-1] + gamma * (1.0 - dones_expanded[-1]) * last_value
    (_, _), targets_without_final = jax.lax.scan(
        _step,
        (final_target, last_value),
        (rewards[:-1], dones_expanded[:-1], values[:-1]),
        reverse=True,
    )
    return jnp.concatenate([targets_without_final, final_target[jnp.newaxis]], axis=0)
