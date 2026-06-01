"""MuJoCo Playground adapter.

The fastest and most reliable route is to use the wrappers already provided by
`purejaxql`, because they expose Playground environments in the Gymnax-like API
used by the AC-PQN script. This adapter centralizes that dependency and provides
clear errors when it is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlaygroundEnvBundle:
    env: Any
    env_params: Any
    action_low: Any
    action_high: Any
    action_dim: int
    episode_length: int


def build_playground_env(config: dict[str, Any]) -> PlaygroundEnvBundle:
    """Create a vectorized MuJoCo Playground environment.

    Required config keys:
      ENV_NAME: Playground registry name, e.g. CartpoleBalance.
      NORMALIZE_OBS, NORMALIZE_REWARD, GAMMA are optional.
    """

    try:
        from purejaxql.utils.brax_wrappers import (  # type: ignore
            ClipAction,
            LogVecWrapper,
            NormalizeVecObservation,
            NormalizeVecReward,
            PlaygroundVecGymnaxWrapper,
        )
    except Exception as exc:  # pragma: no cover - dependency-specific message
        raise ImportError(
            "The Playground adapter expects purejaxql to be installed. Install with:\n"
            "  pip install git+https://github.com/mttga/purejaxql.git\n"
            "and install MuJoCo Playground with either `pip install playground` or the "
            "source install recommended by google-deepmind/mujoco_playground."
        ) from exc

    env_params = None
    env = PlaygroundVecGymnaxWrapper(config["ENV_NAME"])
    action_space = env.action_space(env_params)
    env = LogVecWrapper(env)
    env = ClipAction(env, low=action_space.low, high=action_space.high)
    if config.get("NORMALIZE_REWARD", False):
        env = NormalizeVecReward(env, config.get("GAMMA", 0.99))
    if config.get("NORMALIZE_OBS", True):
        env = NormalizeVecObservation(env)

    return PlaygroundEnvBundle(
        env=env,
        env_params=env_params,
        action_low=action_space.low,
        action_high=action_space.high,
        action_dim=int(action_space.shape[0]),
        episode_length=int(getattr(env, "episode_length", config.get("EPISODE_LENGTH", 1000))),
    )


def get_actor_obs(obs: Any) -> Any:
    """Return actor observation from a Playground/PureJAXQL observation pytree."""

    if isinstance(obs, dict):
        if "actor" in obs:
            return obs["actor"]
        if "state" in obs:
            return obs["state"]
        if "obs" in obs:
            return obs["obs"]
    return obs


def get_critic_obs(obs: Any) -> Any:
    """Return critic observation from a Playground/PureJAXQL observation pytree."""

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
