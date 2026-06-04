"""Hand-written NEXUS skills for MuJoCo Playground Go1JoystickFlatTerrain.

Skills:
  0 stand: hold a stable quadruped stance.
  1 track_velocity: follow commanded planar velocity.
  2 turn: follow yaw-rate command.
  3 recover: recover after low body height or large roll/pitch.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.common import actor_obs, action_cost, info_value, l2_norm, safe_index, safe_slice

SKILL_NAMES = ("stand", "track_velocity", "turn", "recover")
NUM_SKILLS = len(SKILL_NAMES)


def _features(obs: Any, info: Any | None = None):
    x = actor_obs(obs)
    base_height = info_value(info, ("base_height", "height", "metrics/base_height"), safe_index(x, 0, 0.32))
    roll = info_value(info, ("roll", "base_roll", "metrics/roll"), safe_index(x, 1))
    pitch = info_value(info, ("pitch", "base_pitch", "metrics/pitch"), safe_index(x, 2))
    lin_vel = safe_slice(x, 3, 5)
    lin_vel_x = info_value(info, ("lin_vel_x", "x_velocity", "forward_velocity"), lin_vel[..., 0])
    lin_vel_y = info_value(info, ("lin_vel_y", "y_velocity", "lateral_velocity"), lin_vel[..., 1])
    yaw_rate = info_value(info, ("yaw_rate", "ang_vel_yaw", "metrics/yaw_rate"), safe_index(x, 5))
    command = safe_slice(x, 6, 9)
    cmd_x = info_value(info, ("command_x", "cmd_x", "commands/x"), command[..., 0])
    cmd_y = info_value(info, ("command_y", "cmd_y", "commands/y"), command[..., 1])
    cmd_yaw = info_value(info, ("command_yaw", "cmd_yaw", "commands/yaw"), command[..., 2])
    return base_height, roll, pitch, lin_vel_x, lin_vel_y, yaw_rate, cmd_x, cmd_y, cmd_yaw


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del prev_obs, env_reward
    height, roll, pitch, vx, vy, yaw_rate, cmd_x, cmd_y, cmd_yaw = _features(obs, info)
    ctrl = action_cost(action, coeff=1e-3)
    posture_error = jnp.abs(roll) + jnp.abs(pitch)
    vel_error = l2_norm(jnp.stack([vx - cmd_x, vy - cmd_y], axis=-1))
    yaw_error = jnp.abs(yaw_rate - cmd_yaw)
    height_reward = 1.0 - jnp.abs(height - 0.32)

    stand = height_reward + 1.0 - posture_error - ctrl
    track = jnp.exp(-2.0 * vel_error) + 0.1 * (vx * cmd_x + vy * cmd_y) - ctrl
    turn = jnp.exp(-2.0 * yaw_error) - ctrl
    recover = height_reward + 2.0 * (1.0 - posture_error) - ctrl
    rewards = jnp.stack([stand, track, turn, recover], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 2.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw = _features(obs, info)
    fallen = (height < 0.22) | (jnp.abs(roll) > 0.6) | (jnp.abs(pitch) > 0.6)
    need_turn = jnp.abs(cmd_yaw) > 0.15
    need_track = l2_norm(jnp.stack([cmd_x, cmd_y], axis=-1)) > 0.10
    return jnp.where(fallen, 3, jnp.where(need_turn, 2, jnp.where(need_track, 1, 0))).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw = _features(obs, info)
    unstable = (height < 0.28) | (jnp.abs(roll) > 0.25) | (jnp.abs(pitch) > 0.25)
    stand = jnp.ones_like(unstable, dtype=bool)
    track = l2_norm(jnp.stack([cmd_x, cmd_y], axis=-1)) > 0.05
    turn = jnp.abs(cmd_yaw) > 0.05
    recover = unstable
    return jnp.stack([stand, track, turn, recover], axis=-1)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, action, env_reward, done
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw = _features(obs, info)
    return {
        "go1/base_height": height,
        "go1/roll": roll,
        "go1/pitch": pitch,
        "go1/command_xy_norm": l2_norm(jnp.stack([cmd_x, cmd_y], axis=-1)),
        "go1/command_yaw": cmd_yaw,
    }


def explain_policy() -> str:
    return (
        "Go1JoystickFlatTerrain: recover if body height/orientation is unsafe; turn for yaw "
        "commands; track_velocity for planar command; stand when no command is active."
    )
