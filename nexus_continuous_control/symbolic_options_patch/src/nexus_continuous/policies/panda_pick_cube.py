"""Hand-written NEXUS skills for MuJoCo Playground PandaPickCube.

Skills:
  0 reach_cube: move end-effector/TCP close to the cube.
  1 grasp_cube: close and secure the cube once close.
  2 lift_cube: raise the cube above the table.
  3 place_or_stabilize: move lifted cube toward the target or hold it stable.

The module prefers semantic keys from `info` when available. Otherwise it falls
back to the first nine observation features as tcp/cube/target xyz coordinates.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from nexus_continuous.policies.common import (
    actor_obs,
    decrease,
    feature_info,
    info_value,
    l2_norm,
    safe_index,
    safe_slice,
)

SKILL_NAMES = ("reach_cube", "grasp_cube", "lift_cube", "place_or_stabilize")
NUM_SKILLS = len(SKILL_NAMES)
REACH_RADIUS = 0.06
LIFT_HEIGHT = 0.12


def _vec_info(info: Any | None, keys: tuple[str, ...], default: jnp.ndarray) -> jnp.ndarray:
    value = info_value(info, keys, default)
    if value.ndim == default.ndim - 1:
        value = value[..., None]
    return value


def _features(obs: Any, info: Any | None = None):
    x = actor_obs(obs)
    semantic = feature_info(obs, info)
    tcp = _vec_info(semantic, ("tcp_pos", "eef_pos", "gripper_pos", "hand_pos"), safe_slice(x, 0, 3))
    cube = _vec_info(
        semantic,
        ("cube_pos", "object_pos", "obj_pos", "block_pos"),
        safe_slice(x, 3, 6),
    )
    target = _vec_info(
        semantic,
        ("target_pos", "goal_pos", "mocap_target_pos"),
        safe_slice(x, 6, 9),
    )
    gripper = info_value(
        semantic,
        ("gripper_open", "finger_pos", "gripper_width"),
        safe_index(x, 9, 1.0),
    )
    cube_height = cube[..., 2]
    dist_tcp_cube = l2_norm(tcp - cube)
    dist_cube_target = l2_norm(cube - target)
    grasped_info = info_value(
        semantic,
        ("grasped", "is_grasped", "object_grasped"),
        jnp.zeros_like(dist_tcp_cube),
    )
    inferred_grasped = (dist_tcp_cube < REACH_RADIUS) & (gripper < 0.35)
    grasped = (grasped_info > 0.5) | inferred_grasped
    return tcp, cube, target, gripper, cube_height, dist_tcp_cube, dist_cube_target, grasped


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del env_reward
    *_, prev_height, prev_dist_tcp_cube, prev_dist_cube_target, _prev_grasped = _features(prev_obs)
    _tcp, _cube, _target, gripper, height, dist_tcp_cube, dist_cube_target, grasped = _features(obs, info)
    ctrl = 1e-4 * jnp.sum(jnp.square(action), axis=-1)

    reach = -dist_tcp_cube + 0.5 * decrease(prev_dist_tcp_cube, dist_tcp_cube) - ctrl
    grasp_bonus = jnp.where(grasped, 1.0, 0.0)
    close_gripper_near_cube = jnp.where(dist_tcp_cube < REACH_RADIUS, 1.0 - gripper, 0.0)
    grasp = -dist_tcp_cube + grasp_bonus + 0.1 * close_gripper_near_cube - ctrl
    lift = height + 2.0 * jnp.maximum(0.0, height - prev_height) + grasp_bonus - ctrl
    place = -dist_cube_target + 0.5 * decrease(prev_dist_cube_target, dist_cube_target) + grasp_bonus - ctrl

    rewards = jnp.stack([reach, grasp, lift, place], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    _tcp, _cube, _target, _gripper, height, dist_tcp_cube, dist_cube_target, grasped = _features(obs, info)
    del dist_cube_target
    far_from_cube = (dist_tcp_cube > REACH_RADIUS) & (~grasped)
    need_grasp = ~grasped
    need_lift = height < LIFT_HEIGHT
    return jnp.where(
        far_from_cube,
        0,
        jnp.where(need_grasp, 1, jnp.where(need_lift, 2, 3)),
    ).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    _tcp, _cube, _target, _gripper, height, dist_tcp_cube, _dist_cube_target, grasped = _features(
        obs,
        info,
    )
    reach = (~grasped) & (dist_tcp_cube > 0.03)
    grasp = (dist_tcp_cube < 0.12) & (~grasped)
    lift = grasped & (height < LIFT_HEIGHT + 0.05)
    place = grasped & (height >= LIFT_HEIGHT * 0.7)
    mask = jnp.stack([reach, grasp, lift, place], axis=-1)
    fallback = jax.nn.one_hot(
        jnp.zeros(mask.shape[:-1], dtype=jnp.int32),
        NUM_SKILLS,
    ).astype(bool)
    return jnp.where(jnp.any(mask, axis=-1, keepdims=True), mask, fallback)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, action, env_reward, done
    _tcp, _cube, _target, gripper, height, dist_tcp_cube, dist_cube_target, grasped = _features(
        obs,
        info,
    )
    return {
        "panda/dist_tcp_cube": dist_tcp_cube,
        "panda/dist_cube_target": dist_cube_target,
        "panda/cube_height": height,
        "panda/gripper": gripper,
        "panda/grasped": grasped.astype(jnp.float32),
    }


def explain_policy() -> str:
    return (
        "PandaPickCube: reach_cube until the gripper is close to the cube; grasp_cube "
        "until the object is secured; lift_cube until above the table; place_or_stabilize afterwards."
    )
