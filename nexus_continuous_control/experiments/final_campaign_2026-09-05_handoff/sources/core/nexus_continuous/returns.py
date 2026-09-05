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
    terminals: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Q(lambda) targets by scanning backwards along time.

    Args:
      rewards: array [T, E, ...].
      dones: array [T, E] or [T, E, 1]. Marks an episode BOUNDARY (either a real
        termination or a time-limit truncation) and cuts the lambda trace there.
      values: greedy/bootstrap Q-values [T, E, ...]. Target at timestep t uses
        values[t + 1] as the one-step bootstrap and `last_value` at the final
        timestep, matching AC-PQN/PQN.
      last_value: bootstrap value [E, ...] for the final state.
      gamma: discount factor.
      lambda_: trace mixing coefficient.
      terminals: optional [T, E] mask of TRUE terminations only (excludes
        time-limit truncations). The one-step bootstrap `gamma*V(s')` is zeroed
        only on true terminals; on a truncation we still bootstrap (r + gamma*V),
        which is the correct time-limit handling (Pardo et al. 2018). Defaults to
        ``dones`` (i.e. treat every boundary as terminal — the previous behavior).

    Returns:
      Lambda targets with shape [T, E, ...].
    """

    if terminals is None:
        terminals = dones

    if dones.ndim < rewards.ndim:
        dones_expanded = jnp.expand_dims(dones, axis=-1)
        terminals_expanded = jnp.expand_dims(terminals, axis=-1)
    else:
        dones_expanded = dones
        terminals_expanded = terminals

    next_values = jnp.concatenate([values[1:], last_value[jnp.newaxis]], axis=0)

    def _step(lambda_return, transition):
        reward_t, done_t, terminal_t, next_q = transition
        # Bootstrap unless this is a TRUE terminal (truncation still bootstraps).
        target_bootstrap = reward_t + gamma * (1.0 - terminal_t) * next_q
        delta = lambda_return - next_q
        # Cut the lambda trace at ANY episode boundary (terminal or truncation).
        lambda_return = target_bootstrap + gamma * lambda_ * (1.0 - done_t) * delta
        lambda_return = jnp.where(terminal_t.astype(bool), reward_t, lambda_return)
        return lambda_return, lambda_return

    _, targets = jax.lax.scan(
        _step,
        last_value,
        (rewards, dones_expanded, terminals_expanded, next_values),
        reverse=True,
    )
    return targets
