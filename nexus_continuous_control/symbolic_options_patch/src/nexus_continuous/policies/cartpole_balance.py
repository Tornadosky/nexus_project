"""Hand-written NEXUS skills for MuJoCo Playground CartpoleBalance.

Skills:
  0 recover_balance: reduce pole angle and angular velocity.
  1 center_cart: keep cart near the center of the track.
  2 damp_motion: keep velocities small once balanced.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import actor_obs, decrease, safe_index

SKILL_NAMES = ("recover_balance", "center_cart", "damp_motion")
NUM_SKILLS = len(SKILL_NAMES)


def _features(obs: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = actor_obs(obs)
    cart_pos = safe_index(x, 0)
    pole_angle = safe_index(x, 1)
    cart_vel = safe_index(x, 2)
    pole_ang_vel = safe_index(x, 3)
    return cart_pos, pole_angle, cart_vel, pole_ang_vel


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del env_reward, info
    prev_cart, prev_angle, prev_cart_vel, prev_ang_vel = _features(prev_obs)
    cart, angle, cart_vel, ang_vel = _features(obs)
    act_penalty = 1e-3 * jnp.sum(jnp.square(action), axis=-1)

    angle_error = jnp.abs(angle)
    prev_angle_error = jnp.abs(prev_angle)
    cart_error = jnp.abs(cart)
    prev_cart_error = jnp.abs(prev_cart)
    vel_error = jnp.abs(cart_vel) + 0.25 * jnp.abs(ang_vel)
    prev_vel_error = jnp.abs(prev_cart_vel) + 0.25 * jnp.abs(prev_ang_vel)

    recover = 1.0 - angle_error + 0.5 * decrease(prev_angle_error, angle_error) - act_penalty
    center = 1.0 - cart_error + 0.5 * decrease(prev_cart_error, cart_error) - act_penalty
    damp = 1.0 - vel_error + 0.25 * decrease(prev_vel_error, vel_error) - act_penalty
    rewards = jnp.stack([recover, center, damp], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    del info
    cart, angle, cart_vel, ang_vel = _features(obs)
    urgent_angle = (jnp.abs(angle) > 0.20) | (jnp.abs(ang_vel) > 1.5)
    off_center = jnp.abs(cart) > 0.35
    high_velocity = (jnp.abs(cart_vel) + jnp.abs(ang_vel)) > 1.0
    return jnp.where(urgent_angle, 0, jnp.where(off_center, 1, jnp.where(high_velocity, 2, 0))).astype(
        jnp.int32
    )


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    del info
    cart, angle, cart_vel, ang_vel = _features(obs)
    recover = (jnp.abs(angle) > 0.08) | (jnp.abs(ang_vel) > 0.7)
    center = jnp.abs(cart) > 0.12
    damp = jnp.ones_like(recover, dtype=bool)
    return jnp.stack([recover, center, damp], axis=-1)


def explain_policy() -> str:
    return (
        "CartpoleBalance: recover_balance when pole angle/angular velocity is large; "
        "center_cart when the cart is away from the center; damp_motion otherwise."
    )
