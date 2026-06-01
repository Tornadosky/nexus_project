"""MuJoCo Playground adapter for the AC-PQN training loop."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

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
    def __init__(self, env: Any, env_config: Any):
        super().__init__(env)
        self.env_config = env_config
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
        return self._get_obs(state.obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: Any, action: jax.Array, params: Any = None
    ) -> tuple[Any, Any, jax.Array, jax.Array, dict[str, Any]]:
        del key, params
        next_state = self._env.step(state, action)
        return (
            self._get_obs(next_state.obs),
            next_state,
            next_state.reward,
            next_state.done > 0.5,
            {},
        )

    def action_space(self, params: Any = None) -> _Box:
        del params
        return _Box(
            low=-self.action_scale,
            high=self.action_scale,
            shape=(self.action_size,),
        )

    def _get_obs(self, obs: Any) -> dict[str, Any]:
        if self.privileged_state:
            return {"actor": obs["state"], "critic": obs["privileged_state"]}
        return {"actor": obs, "critic": obs}


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
    env = _PlaygroundVecWrapper(env, env_config)
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
    """Return actor observation from a Playground observation pytree."""

    if isinstance(obs, dict):
        if "actor" in obs:
            return obs["actor"]
        if "state" in obs:
            return obs["state"]
        if "obs" in obs:
            return obs["obs"]
    return obs


def get_critic_obs(obs: Any) -> Any:
    """Return critic observation from a Playground observation pytree."""

    if isinstance(obs, dict):
        if "critic" in obs:
            return obs["critic"]
        if "state" in obs:
            return obs["state"]
        if "actor" in obs:
            return obs["actor"]
        if "obs" in obs:
            return obs["obs"]
    return obs
