"""Hand-written NEXUS skills for MuJoCo Ant locomotion.

Skills:
0 stand: stabilize upright posture
1 walk_forward: move forward
2 run_forward: fast aggressive locomotion
3 recover: recover from fall / instability
"""

from __future__ import annotations 

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import (
    actor_obs,
    action_cost,
    feature_info,
    info_value,
    safe_index,
)

SKILL_NAMES = ("stand", "walk_forward", "run_forward", "recover")
NUM_SKILLS = len(SKILL_NAMES)


def _features(obs: Any, info: Any | None = None):
    x = actor_obs(obs)
    semantic = feature_info(obs, info)
    height = info_value(semantic,("height", "base_height", "torso_height"),safe_index(x, 0, 1.4))
    
    roll = info_value(semantic, ("roll", "base_roll"), safe_index(x, 3))
    pitch = info_value(semantic, ("pitch", "base_pitch"), safe_index(x, 4))
    
    joint_speed = info_value(semantic, ("joint_speed", "metrics/joint_speed"), jnp.mean(jnp.abs(x[..., x.shape[-1] // 2 :]), axis = -1))
    forward_velocity = info_value(semantic, ("x_velocity", "forward_velocity", "metrics/forward_velocity"), safe_index(x, -1))
    
    return height, roll, pitch, joint_speed, forward_velocity


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, env_reward
    height, roll, pitch, _joint_speed, forward_velocity = _features(obs, info)
    
    ctrl = action_cost(action, coeff = 1e-3)
    posture_error = jnp.abs(roll) + jnp.abs(pitch)
    height_reward = 1.0 - jnp.abs(height - 1.4)

    stand = height + (1.0 - posture_error) - ctrl
    walk_forward = jnp.exp(-jnp.abs(forward_velocity - 1.5)) + 0.5 * height_reward - ctrl
    run_forward = forward_velocity + 0.25 * height_reward - ctrl
    recover = 2.0 * height + 2.0 * (1.0 - posture_error) - ctrl
    rewards = jnp.stack([stand, walk_forward, run_forward, recover], axis=-1)
    
    return jnp.where(done[..., None].astype(bool), rewards - 2.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _joint_speed, forward_velocity = _features(obs, info)
    fallen = (height < 0.9) | (jnp.abs(roll) > 0.6) | (jnp.abs(pitch) > 0.6)
    slow = forward_velocity < 1.0
    fast = forward_velocity > 3.0
    return jnp.where(fallen, 3, jnp.where(slow, 1, jnp.where(fast, 2, 0))).astype(jnp.int32)

def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _joint_speed, _forward_velocity = _features(obs, info)
    unstable = (height < 1.1) | (jnp.abs(roll) > 0.3) | (jnp.abs(pitch) > 0.3)
    stand = jnp.ones_like(unstable, dtype=bool)
    walk_forward = height > 1.0
    run_forward = (height > 1.1) & (~unstable)
    recover = unstable
    return jnp.stack([stand, walk_forward, run_forward, recover], axis=-1)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    
    del prev_obs, env_reward, done
    height, roll, pitch, joint_speed, forward_velocity = _features(obs, info)
    
    return {
        "humanoid/height": height,
        "humanoid/roll": roll,
        "humanoid/pitch": pitch,
        "humanoid/joint_speed": joint_speed,
        "humanoid/forward_velocity": forward_velocity,
        "humanoid/action_norm": jnp.linalg.norm(action, axis = -1)
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
    height, roll, pitch, joint_speed, forward_velocity = _features(obs, info)
    not_fallen = (height > 0.9) & (jnp.abs(roll) < 0.6) & (jnp.abs(pitch) < 0.6)
    speed_success = forward_velocity > 1.0
    success = (not_fallen & speed_success)
    
    return {
        "humanoid/no_fall_rate": not_fallen.astype(jnp.float32),
        "humanoid/forward_velocity_mean": forward_velocity,
        "humanoid/height_mean": height,
        "humanois/speed_success_rate": speed_success.astype(jnp.float32),
        "primary_goal_metric": forward_velocity,
        "primary_success_rate": success.astype(jnp.float32),
    }


def explain_policy() -> str:
    return (
        "Humanoid Walk: recover when unstable or fallen, "
        "walk forward at low speed, run forward once stable "
        "and moving quickly, stand on balance otherwise."
    )


