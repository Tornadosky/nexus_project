"""Hand-written NEXUS skills for MuJoCo Playground CheetahRun.

Skills:
  0 accelerate_forward: maximize forward x velocity.
  1 stabilize_posture: avoid extreme torso pitch and joint excursions.
  2 energy_efficient_run: move forward with smaller torques.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import actor_obs, action_cost, info_value, safe_index

SKILL_NAMES = ("accelerate_forward", "stabilize_posture", "energy_efficient_run")
NUM_SKILLS = len(SKILL_NAMES)


def _features(obs: Any, info: Any | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = actor_obs(obs)
    torso_pitch = safe_index(x, 1)
    joint_speed = jnp.mean(jnp.abs(x[..., x.shape[-1] // 2 :]), axis=-1)
    default_vel = safe_index(x, -1)
    x_velocity = info_value(
        info,
        ("x_velocity", "forward_velocity", "reward/forward", "metrics/forward_velocity"),
        default_vel,
    )
    return x_velocity, torso_pitch, joint_speed


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, env_reward
    x_velocity, torso_pitch, joint_speed = _features(obs, info)
    ctrl = action_cost(action, coeff=1e-3)
    pitch_error = jnp.abs(torso_pitch)

    accelerate = x_velocity - ctrl
    stabilize = 1.0 - pitch_error - 0.05 * joint_speed - ctrl
    efficient = 0.5 * x_velocity - 5.0 * ctrl
    rewards = jnp.stack([accelerate, stabilize, efficient], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    x_velocity, torso_pitch, joint_speed = _features(obs, info)
    bad_posture = (jnp.abs(torso_pitch) > 0.55) | (joint_speed > 8.0)
    fast_enough = x_velocity > 6.0
    return jnp.where(bad_posture, 1, jnp.where(fast_enough, 2, 0)).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    x_velocity, torso_pitch, joint_speed = _features(obs, info)
    accelerate = x_velocity < 10.0
    stabilize = (jnp.abs(torso_pitch) > 0.20) | (joint_speed > 4.0)
    efficient = jnp.ones_like(accelerate, dtype=bool)
    return jnp.stack([accelerate, stabilize, efficient], axis=-1)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, env_reward, done
    x_velocity, torso_pitch, joint_speed = _features(obs, info)
    return {
        "cheetah/forward_velocity": x_velocity,
        "cheetah/torso_pitch": torso_pitch,
        "cheetah/joint_speed": joint_speed,
        "cheetah/action_norm": jnp.linalg.norm(action, axis=-1),
    }


def explain_policy() -> str:
    return (
        "CheetahRun: accelerate_forward while below target speed; stabilize_posture "
        "when pitch/joint speeds are large; energy_efficient_run once speed is adequate."
    )
