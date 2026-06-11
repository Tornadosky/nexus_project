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

from nexus_continuous.policies.common import actor_obs, feature_info, info_value, safe_index

SKILL_NAMES = ("stand_recover", "hop_forward", "stabilize_landing", "energy_efficient")
NUM_SKILLS = len(SKILL_NAMES)
TARGET_HOP_SPEED = 1.5

# Success-metric calibration aligned to the underlying Playground HopperHop env,
# whose task reward is ``standing * hopping``: ``standing`` saturates once the
# torso-over-foot height reaches ``_STAND_HEIGHT = 0.6`` and ``hopping`` earns
# partial credit from ~1.0 m/s up to ``_HOP_SPEED = 2.0``. ``pitch`` is an
# unwrapped hinge angle, so uprightness uses ``cos(pitch)`` (wrap-safe) instead
# of ``|pitch|``.
ENV_STAND_HEIGHT = 0.6
UPRIGHT_COS = 0.7  # cos(pitch) above this counts as roughly upright (~45 deg).
HOP_SUCCESS_SPEED = 1.0  # env grants partial hop credit from ~1.0 m/s upward.


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
    vertical_or_joint_speed = info_value(
        semantic,
        ("joint_speed", "metrics/joint_speed"),
        jnp.mean(jnp.abs(x[..., x.shape[-1] // 2 :]), axis=-1),
    )
    return height, pitch, x_velocity, vertical_or_joint_speed


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    prev_height, _prev_pitch, _prev_x_velocity, _prev_speed = _features(prev_obs)
    height, pitch, x_velocity, speed = _features(obs, info)
    semantic = feature_info(obs, info)
    action_norm = jnp.linalg.norm(action, axis=-1)
    upright = 1.0 - jnp.clip(jnp.abs(pitch) / 1.2, 0.0, 1.0)
    healthy_height = jnp.clip(height / 0.6, -1.0, 1.0)
    height_progress = height - prev_height
    speed_track = jnp.clip(x_velocity / TARGET_HOP_SPEED, -1.0, 1.0)
    standing_metric = info_value(
        semantic,
        ("reward/standing", "metrics/reward/standing"),
        jnp.clip(healthy_height, 0.0, 1.0) * upright,
    )
    hopping_metric = info_value(
        semantic,
        ("reward/hopping", "metrics/reward/hopping"),
        jnp.clip(speed_track, 0.0, 1.0),
    )

    stand = (
        1.25 * standing_metric
        + 0.5 * healthy_height
        + 0.5 * height_progress
        - 0.2 * jnp.abs(pitch)
        - 0.01 * action_norm
    )
    hop = (
        env_reward
        + 0.75 * hopping_metric
        + 0.25 * standing_metric
        + 0.25 * speed_track
        - 0.01 * action_norm
    )
    stabilize = 0.75 * standing_metric + 0.25 * upright - 0.05 * speed - 0.1 * jnp.abs(pitch)
    efficient = env_reward + 0.25 * standing_metric + 0.1 * speed_track - 0.02 * action_norm
    rewards = jnp.stack([stand, hop, stabilize, efficient], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, speed = _features(obs, info)
    upright_cos = jnp.cos(pitch)
    # Env-aligned and wrap-safe: the env counts standing from height 0.6, and
    # hop cycles dip below 0.9 every bounce, so the old `height < 0.9` plus
    # wrap-unsafe `|pitch| > 0.45` kept the rule in stand_recover on nearly
    # every step (one-skill degeneracy).
    recover = (height < ENV_STAND_HEIGHT) | (upright_cos < UPRIGHT_COS)
    hop = x_velocity < TARGET_HOP_SPEED
    stabilize = (upright_cos < 0.97) | (speed > 10.0)
    return jnp.where(recover, 0, jnp.where(hop, 1, jnp.where(stabilize, 2, 3))).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, pitch, x_velocity, speed = _features(obs, info)
    stand = (height < 1.05) | (jnp.abs(pitch) > 0.25)
    hop = x_velocity < 2.5
    stabilize = (jnp.abs(pitch) > 0.10) | (speed > 5.0)
    efficient = jnp.ones_like(stand, dtype=bool)
    return jnp.stack([stand, hop, stabilize, efficient], axis=-1)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, action, env_reward
    height, pitch, x_velocity, speed = _features(obs, info)
    return {
        "hopper/height": height,
        "hopper/pitch": pitch,
        "hopper/upright_cos": jnp.cos(pitch),
        "hopper/forward_velocity": x_velocity,
        "hopper/joint_speed": speed,
        "hopper/done_fraction": done.astype(jnp.float32),
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
    height, pitch, x_velocity, _speed = _features(obs, info)
    # Aligned to the env's own standing/hopping definition (wrap-safe uprightness).
    upright = (height > ENV_STAND_HEIGHT) & (jnp.cos(pitch) > UPRIGHT_COS)
    hopping = upright & (x_velocity > HOP_SUCCESS_SPEED)
    return {
        "hopper/upright_rate": upright.astype(jnp.float32),
        "hopper/hop_success_rate": hopping.astype(jnp.float32),
        "hopper/forward_velocity_mean": x_velocity,
        "primary_goal_metric": x_velocity,
        "primary_success_rate": hopping.astype(jnp.float32),
    }


def explain_policy() -> str:
    return (
        "HopperHop: stand_recover for low or tilted torso; hop_forward below target speed; "
        "stabilize_landing for high pitch/joint velocity; energy_efficient otherwise."
    )
