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

import jax.numpy as jnp

from nexus_continuous.policies.common import (
    actor_obs,
    decrease,
    ensure_mask_has_skill,
    feature_info,
    info_value,
    l2_norm,
    safe_index,
    safe_slice,
)

SKILL_NAMES = ("reach_cube", "grasp_cube", "lift_cube", "place_or_stabilize")
NUM_SKILLS = len(SKILL_NAMES)
REACH_RADIUS = 0.06
GRASP_RADIUS = 0.15
LIFT_RADIUS = 0.20
LIFT_GRIPPER_OPEN_MAX = 0.65
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
    grasp_proxy = (grasped_info > 0.5) | inferred_grasped
    return tcp, cube, target, gripper, cube_height, dist_tcp_cube, dist_cube_target, grasp_proxy


def skill_rewards(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> jnp.ndarray:
    del env_reward
    *_, prev_height, prev_dist_tcp_cube, prev_dist_cube_target, _prev_grasp_proxy = _features(
        prev_obs
    )
    _tcp, _cube, _target, gripper, height, dist_tcp_cube, dist_cube_target, grasp_proxy = _features(
        obs,
        info,
    )
    ctrl = 1e-4 * jnp.sum(jnp.square(action), axis=-1)

    reach = -dist_tcp_cube + 0.5 * decrease(prev_dist_tcp_cube, dist_tcp_cube) - ctrl
    grasp_bonus = jnp.where(grasp_proxy, 1.0, 0.0)
    close_gripper_near_cube = jnp.where(dist_tcp_cube < REACH_RADIUS, 1.0 - gripper, 0.0)
    grasp = -dist_tcp_cube + grasp_bonus + 0.1 * close_gripper_near_cube - ctrl
    height_delta = jnp.maximum(0.0, height - prev_height)
    height_above_table = jnp.maximum(0.0, height - 0.03)
    lift = 4.0 * height_delta + 2.0 * height_above_table - 0.25 * dist_tcp_cube - ctrl
    place = (
        -dist_cube_target
        + 0.5 * decrease(prev_dist_cube_target, dist_cube_target)
        + grasp_bonus
        - ctrl
    )

    rewards = jnp.stack([reach, grasp, lift, place], axis=-1)
    return jnp.where(done[..., None].astype(bool), rewards - 1.0, rewards)


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    (
        _tcp,
        _cube,
        _target,
        gripper,
        height,
        dist_tcp_cube,
        _dist_cube_target,
        grasp_proxy,
    ) = _features(obs, info)
    reach = (~grasp_proxy) & (dist_tcp_cube > GRASP_RADIUS)
    grasp = (dist_tcp_cube <= GRASP_RADIUS) & (gripper > 0.15) & (height < LIFT_HEIGHT)
    lift = (
        (dist_tcp_cube <= LIFT_RADIUS)
        & (gripper <= LIFT_GRIPPER_OPEN_MAX)
        & (height < LIFT_HEIGHT + 0.03)
    )
    place = height >= LIFT_HEIGHT
    skill = jnp.where(reach, 0, jnp.where(grasp, 1, jnp.where(lift, 2, jnp.where(place, 3, 0))))
    return skill.astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    _tcp, _cube, _target, gripper, height, dist_tcp_cube, _dist_cube_target, grasp_proxy = _features(
        obs,
        info,
    )
    reach = (~grasp_proxy) & (dist_tcp_cube > GRASP_RADIUS)
    grasp = (dist_tcp_cube <= GRASP_RADIUS) & (gripper > 0.15) & (height < LIFT_HEIGHT)
    lift = (
        (dist_tcp_cube <= LIFT_RADIUS)
        & (gripper <= LIFT_GRIPPER_OPEN_MAX)
        & (height < LIFT_HEIGHT + 0.03)
    )
    place = height >= LIFT_HEIGHT
    mask = jnp.stack([reach, grasp, lift, place], axis=-1)
    return ensure_mask_has_skill(mask, default_skill=0)


def diagnostics(
    prev_obs: Any,
    obs: Any,
    action: jnp.ndarray,
    env_reward: jnp.ndarray,
    done: jnp.ndarray,
    info: Any | None = None,
) -> dict[str, jnp.ndarray]:
    del prev_obs, action, env_reward, done
    _tcp, _cube, _target, gripper, height, dist_tcp_cube, dist_cube_target, grasp_proxy = _features(
        obs,
        info,
    )
    reach_success = dist_tcp_cube < 0.06
    closed_near_cube = reach_success & (gripper < 0.35)
    lift_success = height > LIFT_HEIGHT
    place_success = lift_success & (dist_cube_target < 0.12)
    return {
        "panda/dist_tcp_cube": dist_tcp_cube,
        "panda/dist_cube_target": dist_cube_target,
        "panda/cube_height": height,
        "panda/gripper": gripper,
        "panda/grasp_proxy": grasp_proxy.astype(jnp.float32),
        "panda/reach_success": reach_success.astype(jnp.float32),
        "panda/closed_near_cube": closed_near_cube.astype(jnp.float32),
        "panda/lift_success": lift_success.astype(jnp.float32),
        "panda/place_success": place_success.astype(jnp.float32),
        "panda/cube_height_delta_from_table": height - 0.03,
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
    _tcp, _cube, _target, gripper, height, dist_tcp_cube, dist_cube_target, _grasp_proxy = _features(
        obs,
        info,
    )
    reach_success = dist_tcp_cube < 0.06
    closed_near_cube = reach_success & (gripper < 0.35)
    lift_success = height > 0.08
    place_success = lift_success & (dist_cube_target < 0.12)
    height_delta_from_table = height - 0.03
    return {
        "panda/reach_success_rate": reach_success.astype(jnp.float32),
        "panda/closed_near_cube_rate": closed_near_cube.astype(jnp.float32),
        "panda/lift_success_rate": lift_success.astype(jnp.float32),
        "panda/place_success_rate": place_success.astype(jnp.float32),
        "panda/cube_height_max_mean": height,
        "panda/cube_height_delta_max_mean": height_delta_from_table,
        "primary_goal_metric": height_delta_from_table,
        "primary_success_rate": lift_success.astype(jnp.float32),
    }


def explain_policy() -> str:
    return (
        "PandaPickCube: reach_cube until the gripper is close to the cube; grasp_cube "
        "until the object is secured; lift_cube until above the table; place_or_stabilize afterwards."
    )
