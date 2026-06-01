"""Optional RGB actor components for the skill-agent extension.

These modules are intentionally separate from the default state-based algorithm.
Use them when the Playground observation wrapper supplies `pixels` and
`proprioception` fields. The recommended setup is a privileged critic:
actor_i(rgb, proprioception) -> action, critic_i(state, action) -> Q_i.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from nexus_continuous.networks import MLP


class RGBEncoder(nn.Module):
    embedding_dim: int = 128

    @nn.compact
    def __call__(self, pixels: jnp.ndarray) -> jnp.ndarray:
        # Accept uint8 [B,H,W,C] or float. Normalize only when values look like 0..255.
        x = pixels.astype(jnp.float32)
        x = jnp.where(jnp.max(x) > 2.0, x / 255.0, x)
        for channels, stride in [(32, 2), (64, 2), (64, 2)]:
            x = nn.Conv(channels, kernel_size=(3, 3), strides=(stride, stride), padding="SAME")(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.embedding_dim)(x)
        x = nn.LayerNorm()(x)
        return nn.relu(x)


class VisionSkillActor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    hidden_sizes: Sequence[int] = (256, 256)
    embedding_dim: int = 128

    @nn.compact
    def __call__(self, pixels: jnp.ndarray, proprioception: jnp.ndarray) -> jnp.ndarray:
        z = RGBEncoder(self.embedding_dim)(pixels)
        x = jnp.concatenate([z, proprioception], axis=-1)
        x = MLP(self.hidden_sizes, self.action_dim, output_scale=0.01)(x)
        return jnp.tanh(x) * self.action_scale + self.action_bias
