"""Hand-written NEXUS skills for MuJoCo Ant locomotion.

Skills:
0 stand: stabilize upright posture
1 walk_forward: move forward
2 bound_forward: fast aggressive locomotion
3 turn: rotate / yaw control
4 recover: recover from fall / instability
"""

from __future__ import annotations 

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import (
    actor_obs,
    feature_info,
    info_value,
    l2_norm,
    safe_index,
)

SKILL_NAMES = ("stand", "walk_forward", "bound_forward", "turn", "recover")
NUM_SKILLS = len(SKILL_NAMES)


def _features(obs: Any, info: Any | None = None):
    x = actor_obs(obs)
    semantic = feature_info(obs, info)
    height = info_value(semantic,("height", "torso_height"),safe_index(x, 0, 0.5))
    
    roll = info_value(semantic, ("roll",), safe_index(x, 1))
    pitch = info_value(semantic, ("pitch",), safe_index(x, 2))
    vx = info_value(semantic,("x_velocity", "forward_velocity"),safe_index(x, 3))
    vy = info_value(semantic,("y_velocity", "lateral_velocity"),safe_index(x, 4))
    yaw_rate = info_value(semantic, ("yaw_rate"), safe_index(x, 5))
    
    cmd = x[..., 6:9]
    cmd_x = cmd[..., 0]
    cmd_y = cmd[..., 1]
    cmd_yaw = cmd[..., 2]
    speed = jnp.sqrt(vx**2 + vy**2)
    
    return height, roll, pitch, vx, vy, yaw_rate, cmd_x, cmd_y, cmd_yaw, speed


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, env_reward
    height, roll, pitch, vx, vy, yaw_rate, cmd_x, cmd_y, cmd_yaw, speed = _features(obs, info)
    
    posture_error = jnp.abs(roll) + jnp.abs(pitch)
    vel_error = l2_norm(jnp.stack([vx - cmd_x, vy - cmd_y], axis=-1))
    yaw_error = jnp.abs(yaw_rate - cmd_yaw)
    ctrl = jnp.mean(jnp.square(action), axis = -1) * 1e-3

    stand = height + (1.0 - posture_error) - ctrl
    walk_forward = jnp.exp(-1.5 * vel_error) + 0.2*vx - ctrl
    bound_forward = 0.9 * speed + 0.2 * height - 1.5 * ctrl
    turn = jnp.exp(-2.0 * yaw_error) - ctrl
    recover = height + 2.0 * (1.0 - posture_error) - ctrl
    rewards = jnp.stack([stand, walk_forward, bound_forward, turn, recover], axis=-1)
    
    return jnp.where(done[..., None].astype(bool), rewards - 2.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw, speed = _features(obs, info)
    posture_error = jnp.abs(roll) + jnp.abs(pitch)
    fallen = (height < 0.25) | (posture_error > 0.8)
    need_turn = jnp.abs(cmd_yaw) > 0.15
    high_speed = speed > 1.5
    low_speed = speed < 0.3
    _need_move = l2_norm(jnp.stack([cmd_x, cmd_y], axis=-1)) > 0.10
    return jnp.where(fallen, 4, jnp.where(need_turn, 3, jnp.where(high_speed, 2, jnp.where(low_speed, 0, 1)))).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw, _speed = _features(obs, info)
    posture_error = jnp.abs(roll) + jnp.abs(pitch)
    unstable = (height < 0.3) | (posture_error > 0.4)
    stand = jnp.ones_like(unstable, dtype=bool)
    walk = ~unstable
    bound = ~unstable
    turn = jnp.abs(cmd_yaw) > 0.05
    recover = unstable
    return jnp.stack([stand, walk, bound, turn, recover], axis=-1)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    
    del prev_obs, action, env_reward, done
    height, roll, pitch, vx, vy, yaw_rate, cmd_x, cmd_y, cmd_yaw, speed = _features(obs, info)
    vel_error = l2_norm(jnp.stack([vx - cmd_x, vy - cmd_y], axis=-1))
    yaw_error = jnp.abs(yaw_rate - cmd_yaw)
    not_fallen = (height > 0.25) & (jnp.abs(roll) < 0.8) & (jnp.abs(pitch) < 0.8)
    
    return {
        "ant/height": height,
        "ant/roll": roll,
        "ant/pitch": pitch,
        "ant/speed": speed,
        "ant/vel_error": vel_error,
        "ant/yaw_error": yaw_error,
        "ant/not_fallen": not_fallen.astype(jnp.float32)
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
    height, roll, pitch, vx, vy, yaw_rate, cmd_x, cmd_y, cmd_yaw, speed = _features(obs, info)
    posture_error = jnp.abs(roll) + jnp.abs(pitch)
    not_fallen = (height > 0.25) & (posture_error < 0.8)
    velocity_tracking_error = jnp.sqrt((vx - cmd_x)**2 + (vy - cmd_y)**2)
    yaw_tracking_error = jnp.abs(yaw_rate - cmd_yaw)
    tracking_success = not_fallen & (velocity_tracking_error < 0.6) & (yaw_tracking_error < 0.6)
    primary_goal = -(velocity_tracking_error + 0.5 * yaw_tracking_error)
    
    return {
        "ant/no_fall_rate": not_fallen.astype(jnp.float32),
        "ant/tracking_success_rate": tracking_success.astype(jnp.float32),
        "ant/velocity_tracking_error_mean": velocity_tracking_error,
        "ant/yaw_tracking_error_mean": yaw_tracking_error,
        "ant/speed_mean": speed,
        "primary_goal_metric": primary_goal,
        "primary_success_rate": tracking_success.astype(jnp.float32),
    }


def explain_policy() -> str:
    return (
        "Ant system: stand for stability, "
        "walk forward for normal walking, bound forward for fast motion, "
        "turn for yaw control, recover for failures."
    )


