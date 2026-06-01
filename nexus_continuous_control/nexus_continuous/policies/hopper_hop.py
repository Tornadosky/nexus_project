"""Hand-written NEXUS skills for MuJoCo Playground HopperHop.

Skills:
  0 stand_recover: recover height/uprightness.
  1 hop_forward: generate forward hopping velocity.
  2 stabilize_landing: reduce pitch and vertical/joint velocity spikes.
  3 energy_efficient: preserve speed while using smaller torques.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import actor_obs, action_cost, info_value, safe_index

SKILL_NAMES = ("stand_recover", "hop_forward", "stabilize_landing", "energy_efficient")
NUM_SKILLS = len(SKILL_NAMES)
TARGET_HOP_SPEED = 1.5


def _features(obs: Any, info: Any | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = actor_obs(obs)
    height = info_value(info, ("torso_height", "height", "metrics/height"), safe_index(x, 0, 1.2))
    pitch = info_value(info, ("torso_pitch", "pitch", "metrics/pitch"), safe_index(x, 1))
    x_velocity = info_value(
        info,
        ("x_velocity", "forward_velocity", "metrics/forward_velocity", "reward/forward"),
        safe_index(x, -1),
    )
    vertical_or_joint_speed = jnp.mean(jnp.abs(x[..., x.shape[-1] // 2 :]), axis=-1)
    return height, pitch, x_velocity, vertical_or_joint_speed


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, env_reward
    height, pitch, x_velocity, speed = _features(obs, info)
    ctrl = action_cost(action, coeff=1e-3)
    upright = 1.0 - jnp.abs(pitch)
    height_reward = 1.0 - jnp.abs(height - 1.2)
    hop_track = 1.0 - jnp.abs(x_velocity - TARGET_HOP_SPEED)

    stand = height_reward + upright - ctrl
    hop = x_velocity + 0.5 * hop_track - ctrl
    stabilize = upright - 0.03 * speed - ctrl
    efficient = 0.4 * x_velocity - 5.0 * ctrl
    rewards = jnp.stack([stand, hop, stabilize, efficient], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, speed = _features(obs, info)
    recover = (height < 0.9) | (jnp.abs(pitch) > 0.45)
    hop = x_velocity < TARGET_HOP_SPEED
    stabilize = (jnp.abs(pitch) > 0.25) | (speed > 10.0)
    return jnp.where(recover, 0, jnp.where(hop, 1, jnp.where(stabilize, 2, 3))).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, speed = _features(obs, info)
    stand = (height < 1.05) | (jnp.abs(pitch) > 0.25)
    hop = x_velocity < 2.5
    stabilize = (jnp.abs(pitch) > 0.10) | (speed > 5.0)
    efficient = jnp.ones_like(stand, dtype=bool)
    return jnp.stack([stand, hop, stabilize, efficient], axis=-1)


def explain_policy() -> str:
    return (
        "HopperHop: stand_recover for low or tilted torso; hop_forward below target speed; "
        "stabilize_landing for high pitch/joint velocity; energy_efficient otherwise."
    )
