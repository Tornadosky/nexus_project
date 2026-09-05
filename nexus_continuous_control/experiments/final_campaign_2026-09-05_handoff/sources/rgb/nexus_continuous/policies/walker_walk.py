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

from nexus_continuous.policies.common import actor_obs, action_cost, feature_info, info_value, safe_index

SKILL_NAMES = ("stand_recover", "walk_forward", "stabilize_gait", "energy_efficient")
NUM_SKILLS = len(SKILL_NAMES)
TARGET_SPEED = 1.2


def _features(obs: Any, info: Any | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = actor_obs(obs)
    semantic = feature_info(obs, info)
    height = info_value(semantic, ("torso_height", "height", "metrics/height"), safe_index(x, 0, 1.2))
    pitch = info_value(semantic, ("torso_pitch", "pitch", "metrics/pitch"), safe_index(x, 1))
    x_velocity = info_value(
        semantic,
        ("x_velocity", "forward_velocity", "metrics/forward_velocity", "reward/forward"),
        safe_index(x, -1),
    )
    joint_speed = info_value(
        semantic,
        ("joint_speed", "metrics/joint_speed"),
        jnp.mean(jnp.abs(x[..., x.shape[-1] // 2 :]), axis=-1),
    )
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
    # Walk carries its own posture terms: without them the walk actor can
    # maximize signed velocity by lunging/falling forward, the meta learns to
    # avoid the skill, and the policy settles into the stand+sway exploit of
    # the env's asymmetric move tolerance (runs/loop_fix/LOOP_NOTES.md,
    # batch A-C: budget, K10 commitment, and sustained exploration all failed
    # without this). Additive rather than multiplicative so the walk actor
    # keeps a recovery gradient after a stumble.
    walk = 0.5 * (height_reward + upright) + x_velocity + 0.5 * speed_track - ctrl
    stabilize = upright - 0.05 * joint_speed - ctrl
    efficient = 0.5 * x_velocity - 5.0 * ctrl
    rewards = jnp.stack([stand, walk, stabilize, efficient], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, joint_speed = _features(obs, info)
    # Wrap-safe uprightness (pitch is an unwrapped hinge angle from a uniform
    # random reset orientation): cos thresholds match the old +/-0.45 and
    # +/-0.25 rad bands without mis-scoring torsos that rotated past +/-pi.
    upright_cos = jnp.cos(pitch)
    fallen_or_tilted = (height < 0.85) | (upright_cos < 0.9)
    unstable = (upright_cos < 0.97) | (joint_speed > 8.0)
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


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, action, env_reward, done
    height, pitch, x_velocity, joint_speed = _features(obs, info)
    return {
        "walker/height": height,
        "walker/pitch": pitch,
        "walker/forward_velocity": x_velocity,
        "walker/joint_speed": joint_speed,
    }


def task_metrics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, action, env_reward, done
    torso_height, torso_pitch, forward_velocity, _joint_speed = _features(obs, info)
    # Wrap-safe uprightness: pitch (qpos[2]) is an unwrapped hinge angle in
    # [-pi, pi] at reset and can accumulate past it, so |pitch| mis-scores a
    # torso that righted itself via rotation. cos(pitch) matches the env's own
    # uprightness term (xmat[2,2], the torso "up" z-component).
    stand_success = (torso_height > 0.85) & (jnp.cos(torso_pitch) > 0.7)
    walk_success = stand_success & (forward_velocity > 0.5)
    return {
        "walker/stand_success_rate": stand_success.astype(jnp.float32),
        "walker/walk_success_rate": walk_success.astype(jnp.float32),
        "walker/forward_velocity_mean": forward_velocity,
        "primary_goal_metric": forward_velocity,
        "primary_success_rate": walk_success.astype(jnp.float32),
    }


def explain_policy() -> str:
    return (
        "WalkerWalk: stand_recover for low/tilted torso; walk_forward below target speed; "
        "stabilize_gait for unstable pitch/joints; energy_efficient otherwise."
    )
