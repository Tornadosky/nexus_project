"""Neural modules for continuous-control NEXUS.

The modules intentionally avoid BatchNorm state so that they are easy to vmap over
skills and seeds. LayerNorm is supported and is the default because it is stable
in the no-replay-buffer / no-target-network PQN setting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


def _activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == "relu":
        return nn.relu
    if name == "tanh":
        return nn.tanh
    if name == "gelu":
        return nn.gelu
    if name == "silu" or name == "swish":
        return nn.silu
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Small configurable MLP with optional LayerNorm."""

    hidden_sizes: Sequence[int]
    output_size: int
    activation: str = "relu"
    norm_type: str = "layer_norm"
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = _activation(self.activation)
        for width in self.hidden_sizes:
            x = nn.Dense(width, kernel_init=nn.initializers.orthogonal())(x)
            if self.norm_type == "layer_norm":
                x = nn.LayerNorm(epsilon=1e-6)(x)
            elif self.norm_type in ("none", None):
                pass
            else:
                raise ValueError(f"Unsupported norm_type: {self.norm_type}")
            x = act(x)
        return nn.Dense(
            self.output_size,
            kernel_init=nn.initializers.orthogonal(self.output_scale),
            bias_init=nn.initializers.zeros,
        )(x)


class SkillActor(nn.Module):
    """Deterministic continuous actor for one skill.

    The tanh output is mapped to the environment action range with
    action_scale/action_bias, matching the DDPG-style actor used by AC-PQN.
    """

    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    norm_type: str = "layer_norm"
    init_scale: float = 0.01

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = MLP(
            self.hidden_sizes,
            self.action_dim,
            activation=self.activation,
            norm_type=self.norm_type,
            output_scale=self.init_scale,
        )(obs)
        x = jnp.tanh(x)
        return x * self.action_scale + self.action_bias


class SkillCritic(nn.Module):
    """Q(s, a) critic for one skill reward."""

    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    norm_type: str = "layer_norm"
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(
            self.hidden_sizes,
            1,
            activation=self.activation,
            norm_type=self.norm_type,
            output_scale=self.init_scale,
        )(x)
        return jnp.squeeze(x, axis=-1)


class MetaQ(nn.Module):
    """Discrete Q-function over interpretable skills/options."""

    num_skills: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    norm_type: str = "layer_norm"
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            self.hidden_sizes,
            self.num_skills,
            activation=self.activation,
            norm_type=self.norm_type,
            output_scale=self.init_scale,
        )(obs)
