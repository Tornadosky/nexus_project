"""Hand-written NEXUS skills for MuJoCo Playground WalkerWalk.

Skills:
  0 stand_recover: keep the torso high/upright.
  1 walk_forward: track a modest forward walking speed.
  2 stabilize_gait: reduce pitch and velocity spikes.
  3 energy_efficient: reduce unnecessary torques once walking.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import actor_obs, action_cost, info_value, safe_index

SKILL_NAMES = ("stand_recover", "walk_forward", "stabilize_gait", "energy_efficient")
NUM_SKILLS = len(SKILL_NAMES)
TARGET_SPEED = 1.2


def _features(obs: Any, info: Any | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = actor_obs(obs)
    height = info_value(info, ("torso_height", "height", "metrics/height"), safe_index(x, 0, 1.2))
    pitch = info_value(info, ("torso_pitch", "pitch", "metrics/pitch"), safe_index(x, 1))
    x_velocity = info_value(
        info,
        ("x_velocity", "forward_velocity", "metrics/forward_velocity", "reward/forward"),
        safe_index(x, -1),
    )
    joint_speed = jnp.mean(jnp.abs(x[..., x.shape[-1] // 2 :]), axis=-1)
    return height, pitch, x_velocity, joint_speed


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, env_reward
    height, pitch, x_velocity, joint_speed = _features(obs, info)
    ctrl = action_cost(action, coeff=1e-3)
    upright = 1.0 - jnp.abs(pitch)
    height_reward = 1.0 - jnp.abs(height - 1.2)
    speed_track = 1.0 - jnp.abs(x_velocity - TARGET_SPEED)

    stand = height_reward + upright - ctrl
    walk = x_velocity + 0.5 * speed_track - ctrl
    stabilize = upright - 0.05 * joint_speed - ctrl
    efficient = 0.5 * x_velocity - 5.0 * ctrl
    rewards = jnp.stack([stand, walk, stabilize, efficient], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, joint_speed = _features(obs, info)
    fallen_or_tilted = (height < 0.85) | (jnp.abs(pitch) > 0.45)
    unstable = (jnp.abs(pitch) > 0.25) | (joint_speed > 8.0)
    slow = x_velocity < TARGET_SPEED
    return jnp.where(fallen_or_tilted, 0, jnp.where(slow, 1, jnp.where(unstable, 2, 3))).astype(
        jnp.int32
    )


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, joint_speed = _features(obs, info)
    stand = (height < 1.05) | (jnp.abs(pitch) > 0.25)
    walk = x_velocity < 2.0
    stabilize = (jnp.abs(pitch) > 0.10) | (joint_speed > 4.0)
    efficient = jnp.ones_like(stand, dtype=bool)
    return jnp.stack([stand, walk, stabilize, efficient], axis=-1)


def explain_policy() -> str:
    return (
        "WalkerWalk: stand_recover for low/tilted torso; walk_forward below target speed; "
        "stabilize_gait for unstable pitch/joints; energy_efficient otherwise."
    )
