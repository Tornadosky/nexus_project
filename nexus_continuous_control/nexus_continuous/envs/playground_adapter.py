"""MuJoCo Playground adapter for the AC-PQN training loop."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from flax import struct


@dataclass(frozen=True)
class PlaygroundEnvBundle:
    env: Any
    env_params: Any
    action_low: Any
    action_high: Any
    action_dim: int
    episode_length: int


@dataclass(frozen=True)
class _Box:
    low: Any
    high: Any
    shape: tuple[int, ...]


class _EnvWrapper:
    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _PlaygroundVecWrapper(_EnvWrapper):
    def __init__(self, env: Any, env_config: Any, env_name: str):
        super().__init__(env)
        self.env_config = env_config
        self.env_name = env_name
        self.action_scale = 1.0
        self.episode_length = env_config.episode_length
        self.action_repeat = env_config.action_repeat
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.privileged_state = isinstance(env.observation_size, dict)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any = None) -> tuple[Any, Any]:
        del params
        state = self._env.reset(key)
        return self._get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: Any, action: jax.Array, params: Any = None
    ) -> tuple[Any, Any, jax.Array, jax.Array, dict[str, Any]]:
        del key, params
        next_state = self._env.step(state, action)
        return (
            self._get_obs(next_state),
            next_state,
            next_state.reward,
            next_state.done > 0.5,
            self._state_info(next_state),
        )

    def action_space(self, params: Any = None) -> _Box:
        del params
        return _Box(
            low=-self.action_scale,
            high=self.action_scale,
            shape=(self.action_size,),
        )

    def _get_obs(self, state: Any) -> dict[str, Any]:
        obs = state.obs
        if self.privileged_state:
            actor_obs = obs["state"]
            critic_obs = obs["privileged_state"]
        else:
            actor_obs = obs
            critic_obs = obs
        return {
            "actor": actor_obs,
            "critic": critic_obs,
            "raw_actor": actor_obs,
            "raw_critic": critic_obs,
            "policy_info": self._state_info(state),
        }

    @staticmethod
    def _is_small_array(value: Any) -> bool:
        return hasattr(value, "shape") and hasattr(value, "dtype") and len(value.shape) <= 2

    @staticmethod
    def _tail_abs_mean(value: Any, start: int) -> Any:
        if value is None:
            return None
        if value.shape[-1] <= start:
            return jnp.zeros(value.shape[:-1], dtype=value.dtype)
        return jnp.mean(jnp.abs(value[..., start:]), axis=-1)

    @staticmethod
    def _safe_attr(env: Any, name: str, default: Any = None) -> Any:
        try:
            return getattr(env, name)
        except AttributeError:
            return default

    @staticmethod
    def _static_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except TypeError:
            pass
        if hasattr(value, "item"):
            try:
                return int(value.item())
            except ValueError:
                pass
        return int(value[0])

    def _state_info(self, state: Any) -> dict[str, Any]:
        info: dict[str, Any] = {}
        metrics = getattr(state, "metrics", None)
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                if self._is_small_array(value):
                    info[str(key)] = value
        raw_info = getattr(state, "info", None)
        if isinstance(raw_info, Mapping):
            for key, value in raw_info.items():
                key = str(key)
                if key.startswith("AutoResetWrapper") or key == "rng":
                    continue
                if self._is_small_array(value):
                    info[key] = value
        info.update(self._semantic_state_info(state))
        return info

    def _semantic_state_info(self, state: Any) -> dict[str, Any]:
        data = getattr(state, "data", None)
        if data is None:
            return {}
        qpos = getattr(data, "qpos", None)
        qvel = getattr(data, "qvel", None)
        xpos = getattr(data, "xpos", None)
        xipos = getattr(data, "xipos", None)
        site_xmat = getattr(data, "site_xmat", None)
        site_xpos = getattr(data, "site_xpos", None)
        mocap_pos = getattr(data, "mocap_pos", None)
        env_name = self.env_name.lower()
        info: dict[str, Any] = {}
        raw_info = getattr(state, "info", None)

        if "cartpole" in env_name and qpos is not None and qvel is not None:
            slider = self._static_int(self._safe_attr(self._env, "_slider_qposadr", 0))
            hinge = self._static_int(self._safe_attr(self._env, "_hinge_1_qposadr", 1))
            info.update(
                {
                    "cart_position": qpos[..., slider],
                    "pole_angle": qpos[..., hinge],
                    "cart_velocity": qvel[..., slider],
                    "pole_angular_velocity": qvel[..., hinge],
                }
            )

        if "cheetah" in env_name and qpos is not None and qvel is not None:
            info.update(
                {
                    "torso_pitch": qpos[..., 2],
                    "pitch": qpos[..., 2],
                    "x_velocity": qvel[..., 0],
                    "forward_velocity": qvel[..., 0],
                    "joint_speed": self._tail_abs_mean(qvel, 3),
                }
            )

        if "walker" in env_name and qpos is not None and qvel is not None and xpos is not None:
            torso = self._safe_attr(self._env, "_torso_id")
            torso_idx = self._static_int(torso)
            if torso_idx is not None:
                info.update(
                    {
                        "torso_height": xpos[..., torso_idx, 2],
                        "height": xpos[..., torso_idx, 2],
                    }
                )
            info.update(
                {
                    "torso_pitch": qpos[..., 2],
                    "pitch": qpos[..., 2],
                    "x_velocity": qvel[..., 0],
                    "forward_velocity": qvel[..., 0],
                    "joint_speed": self._tail_abs_mean(qvel, 3),
                }
            )

        if "hopper" in env_name and qpos is not None and qvel is not None:
            torso = self._safe_attr(self._env, "_torso_id")
            foot = self._safe_attr(self._env, "_foot_id")
            torso_idx = self._static_int(torso)
            foot_idx = self._static_int(foot)
            if torso_idx is not None and foot_idx is not None and xipos is not None:
                height = xipos[..., torso_idx, 2] - xipos[..., foot_idx, 2]
                info.update({"torso_height": height, "height": height})
            info.update(
                {
                    "torso_pitch": qpos[..., 2],
                    "pitch": qpos[..., 2],
                    "x_velocity": qvel[..., 0],
                    "forward_velocity": qvel[..., 0],
                    "joint_speed": self._tail_abs_mean(qvel, 3),
                }
            )

        if "panda" in env_name and qpos is not None:
            gripper_site = self._safe_attr(self._env, "_gripper_site")
            obj_body = self._safe_attr(self._env, "_obj_body")
            mocap_target = self._safe_attr(self._env, "_mocap_target")
            gripper_site_idx = self._static_int(gripper_site)
            obj_body_idx = self._static_int(obj_body)
            mocap_target_idx = self._static_int(mocap_target)
            if gripper_site_idx is not None and site_xpos is not None:
                tcp_pos = site_xpos[..., gripper_site_idx, :]
                info.update({"tcp_pos": tcp_pos, "gripper_pos": tcp_pos, "eef_pos": tcp_pos})
            if obj_body_idx is not None and xpos is not None:
                cube_pos = xpos[..., obj_body_idx, :]
                info.update({"cube_pos": cube_pos, "object_pos": cube_pos, "obj_pos": cube_pos})
            if mocap_target_idx is not None and mocap_pos is not None:
                target_pos = mocap_pos[..., mocap_target_idx, :]
                info.update({"target_pos": target_pos, "goal_pos": target_pos})
            robot_qposadr = self._safe_attr(self._env, "_robot_qposadr")
            if robot_qposadr is not None and len(robot_qposadr) >= 2:
                finger_idx = tuple(int(idx) for idx in robot_qposadr[-2:])
                gripper_width = jnp.sum(qpos[..., list(finger_idx)], axis=-1)
                info["gripper_width"] = gripper_width
                info["gripper_open"] = jnp.clip(gripper_width / 0.08, 0.0, 1.0)

        if "go1" in env_name and qpos is not None and qvel is not None:
            info.update(
                {
                    "base_height": qpos[..., 2],
                    "height": qpos[..., 2],
                    "lin_vel_x": qvel[..., 0],
                    "lin_vel_y": qvel[..., 1],
                    "x_velocity": qvel[..., 0],
                    "y_velocity": qvel[..., 1],
                    "yaw_rate": qvel[..., 5],
                    "ang_vel_yaw": qvel[..., 5],
                }
            )
            imu_site = self._static_int(self._safe_attr(self._env, "_imu_site_id"))
            if imu_site is not None and site_xmat is not None:
                up = site_xmat[..., imu_site, :, 2]
                roll = jnp.arctan2(up[..., 1], up[..., 2])
                pitch = jnp.arctan2(
                    -up[..., 0],
                    jnp.sqrt(jnp.square(up[..., 1]) + jnp.square(up[..., 2])),
                )
                info.update({"roll": roll, "base_roll": roll, "pitch": pitch, "base_pitch": pitch})
            if isinstance(raw_info, Mapping) and "command" in raw_info:
                command = raw_info["command"]
                info.update(
                    {
                        "command_x": command[..., 0],
                        "cmd_x": command[..., 0],
                        "command_y": command[..., 1],
                        "cmd_y": command[..., 1],
                        "command_yaw": command[..., 2],
                        "cmd_yaw": command[..., 2],
                    }
                )
        
        if "ant" in env_name and qpos is not None and qvel is not None:
            info.update({
                "base_height": qpos[..., 2],
                "height": qpos[..., 2],
                "x_velocity": qvel[..., 0],
                "y_velocity": qvel[..., 1],
                "roll": 0.0,
                "pitch": 0.0,
                "joint_speed": self._tail_abs_mean(qvel, 3),
                "qpos": qpos,
                "qvel": qvel
            })
            
        if "humanoid" in env_name and qpos is not None and qvel is not None:
            info.update(
                {
                    "base_height": qpos[..., 2],
                    "height": qpos[..., 2],
                    "x_velocity": qvel[..., 0],
                    "y_velocity": qvel[..., 1],
                    "forward_velocity": qvel[..., 0],
                    "joint_speed": self._tail_abs_mean(qvel, 3),
                }
            )
            if qpos.shape[-1] > 6:
                info.update(
                    {
                        "roll": qpos[..., 3],
                        "pitch": qpos[..., 4],
                        "yaw": qpos[..., 5]
                    }
                )

        return info


@struct.dataclass
class _LogVecEnvState:
    env_state: Any
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array
    timestep: jax.Array


class _LogVecWrapper(_EnvWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any = None) -> tuple[Any, _LogVecEnvState]:
        obs, env_state = self._env.reset(key, params)
        batch_size = obs["actor"].shape[0]
        zeros = jnp.zeros(batch_size)
        return obs, _LogVecEnvState(env_state, zeros, zeros, zeros, zeros, zeros)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: _LogVecEnvState, action: jax.Array, params: Any = None
    ) -> tuple[Any, _LogVecEnvState, jax.Array, jax.Array, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        finite_reward = jnp.isfinite(reward)
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        episode_returns = state.episode_returns + reward
        episode_lengths = state.episode_lengths + 1
        next_state = _LogVecEnvState(
            env_state=env_state,
            episode_returns=episode_returns * (1 - done),
            episode_lengths=episode_lengths * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + episode_returns * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + episode_lengths * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = next_state.returned_episode_returns
        info["returned_episode_lengths"] = next_state.returned_episode_lengths
        info["returned_episode"] = done
        info["timestep"] = next_state.timestep
        info["original_reward"] = reward
        info["nonfinite_reward"] = (~finite_reward).astype(jnp.float32)
        return obs, next_state, reward, done, info


class _ClipAction(_EnvWrapper):
    def __init__(self, env: Any, low: Any, high: Any):
        super().__init__(env)
        self.low = low
        self.high = high

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: Any, action: jax.Array, params: Any = None
    ) -> tuple[Any, Any, jax.Array, jax.Array, dict[str, Any]]:
        return self._env.step(key, state, jnp.clip(action, self.low, self.high), params)


def _updated_stats(
    mean: jax.Array, var: jax.Array, count: jax.Array, batch: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - mean
    total = count + batch_count
    new_mean = mean + delta * batch_count / total
    m2 = (
        var * count
        + batch_var * batch_count
        + jnp.square(delta) * count * batch_count / total
    )
    return new_mean, m2 / total, total


@struct.dataclass
class _NormalizeObsState:
    actor_mean: jax.Array
    actor_var: jax.Array
    actor_count: jax.Array
    critic_mean: jax.Array
    critic_var: jax.Array
    critic_count: jax.Array
    env_state: Any


class _NormalizeVecObservation(_EnvWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any = None) -> tuple[Any, _NormalizeObsState]:
        obs, env_state = self._env.reset(key, params)
        actor = obs["actor"]
        critic = obs["critic"]
        actor_mean, actor_var, actor_count = _updated_stats(
            jnp.zeros_like(actor[0]), jnp.ones_like(actor[0]), jnp.asarray(1e-4), actor
        )
        critic_mean, critic_var, critic_count = _updated_stats(
            jnp.zeros_like(critic[0]), jnp.ones_like(critic[0]), jnp.asarray(1e-4), critic
        )
        state = _NormalizeObsState(
            actor_mean,
            actor_var,
            actor_count,
            critic_mean,
            critic_var,
            critic_count,
            env_state,
        )
        return self._normalized(obs, state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: _NormalizeObsState, action: jax.Array, params: Any = None
    ) -> tuple[Any, _NormalizeObsState, jax.Array, jax.Array, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        actor_mean, actor_var, actor_count = _updated_stats(
            state.actor_mean, state.actor_var, state.actor_count, obs["actor"]
        )
        critic_mean, critic_var, critic_count = _updated_stats(
            state.critic_mean, state.critic_var, state.critic_count, obs["critic"]
        )
        next_state = _NormalizeObsState(
            actor_mean,
            actor_var,
            actor_count,
            critic_mean,
            critic_var,
            critic_count,
            env_state,
        )
        return self._normalized(obs, next_state), next_state, reward, done, info

    @staticmethod
    def _normalized(obs: dict[str, Any], state: _NormalizeObsState) -> dict[str, Any]:
        return {
            "actor": (obs["actor"] - state.actor_mean) / jnp.sqrt(state.actor_var + 1e-8),
            "critic": (obs["critic"] - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-8),
            "raw_actor": obs.get("raw_actor", obs["actor"]),
            "raw_critic": obs.get("raw_critic", obs["critic"]),
            "policy_info": obs.get("policy_info", {}),
        }


@struct.dataclass
class _NormalizeRewardState:
    mean: jax.Array
    var: jax.Array
    count: jax.Array
    return_value: jax.Array
    env_state: Any


class _NormalizeVecReward(_EnvWrapper):
    def __init__(self, env: Any, gamma: float):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any = None) -> tuple[Any, _NormalizeRewardState]:
        obs, env_state = self._env.reset(key, params)
        batch_size = obs["actor"].shape[0]
        return obs, _NormalizeRewardState(
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            jnp.asarray(1e-4),
            jnp.zeros(batch_size),
            env_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: _NormalizeRewardState, action: jax.Array, params: Any = None
    ) -> tuple[Any, _NormalizeRewardState, jax.Array, jax.Array, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_value = state.return_value * self.gamma * (1 - done) + reward
        mean, var, count = _updated_stats(
            state.mean, state.var, state.count, return_value[:, None]
        )
        next_state = _NormalizeRewardState(
            jnp.squeeze(mean), jnp.squeeze(var), count, return_value, env_state
        )
        return obs, next_state, reward / jnp.sqrt(next_state.var + 1e-8), done, info


def build_playground_env(config: dict[str, Any]) -> PlaygroundEnvBundle:
    """Create a vectorized MuJoCo Playground environment."""

    try:
        from mujoco_playground import registry  # type: ignore
        from mujoco_playground._src.wrapper import wrap_for_brax_training  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-specific message
        raise ImportError(
            "Install MuJoCo Playground from the source install recommended by "
            "google-deepmind/mujoco_playground."
        ) from exc

    env_config = registry.get_default_config(config["ENV_NAME"])
    env_config.impl = config.get("PLAYGROUND_IMPL", "jax")
    env = registry.load(config["ENV_NAME"], env_config)
    env = wrap_for_brax_training(
        env,
        episode_length=env_config.episode_length,
        action_repeat=env_config.action_repeat,
    )
    env = _PlaygroundVecWrapper(env, env_config, config["ENV_NAME"])
    env_params = None
    action_space = env.action_space(env_params)
    env = _LogVecWrapper(env)
    env = _ClipAction(env, low=action_space.low, high=action_space.high)
    if config.get("NORMALIZE_REWARD", False):
        env = _NormalizeVecReward(env, config.get("GAMMA", 0.99))
    if config.get("NORMALIZE_OBS", True):
        env = _NormalizeVecObservation(env)

    return PlaygroundEnvBundle(
        env=env,
        env_params=env_params,
        action_low=action_space.low,
        action_high=action_space.high,
        action_dim=int(action_space.shape[0]),
        episode_length=int(getattr(env, "episode_length", config.get("EPISODE_LENGTH", 1000))),
    )


def get_actor_obs(obs: Any) -> Any:
    """Return network actor observations. May be normalized."""

    if isinstance(obs, dict):
        return obs.get("actor", obs.get("raw_actor", obs.get("state", obs.get("obs"))))
    return obs


def get_critic_obs(obs: Any) -> Any:
    """Return network critic observations. May be normalized."""

    if isinstance(obs, dict):
        return obs.get(
            "critic",
            obs.get("actor", obs.get("raw_critic", obs.get("raw_actor", obs.get("state", obs.get("obs"))))),
        )
    return obs


def get_policy_obs(obs: Any) -> dict[str, Any]:
    """Return raw observations for symbolic rules, masks, rewards, and diagnostics."""

    if not isinstance(obs, dict):
        return {"actor": obs, "critic": obs, "raw_actor": obs, "raw_critic": obs}
    raw_actor = obs.get("raw_actor", obs.get("actor"))
    raw_critic = obs.get("raw_critic", obs.get("critic", raw_actor))
    policy_obs = {
        "actor": raw_actor,
        "critic": raw_critic,
        "raw_actor": raw_actor,
        "raw_critic": raw_critic,
    }
    if "policy_info" in obs:
        policy_obs["policy_info"] = obs["policy_info"]
    return policy_obs
