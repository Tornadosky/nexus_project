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
        # Deterministic, dtype-driven normalization (DrQ-v2 convention,
        # arXiv:2107.09645): integer frames in [0, 255] are mapped to [-0.5, 0.5];
        # float frames are assumed already normalized (MuJoCo Playground emits
        # float grayscale ~[-0.5, 0.5]) and pass through unchanged. The branch is
        # on the *static* dtype, never on the data, so every minibatch — augmented
        # or not — is scaled identically. (The previous `jnp.max(x) > 2` heuristic
        # was a per-batch global reduction: one stray pixel rescaled the whole
        # batch, silently injecting a moving input scale.)
        if jnp.issubdtype(pixels.dtype, jnp.integer):
            x = pixels.astype(jnp.float32) / 255.0 - 0.5
        else:
            x = pixels.astype(jnp.float32)
        # Orthogonal init with sqrt(2) gain is the ReLU-matched default used by
        # DrQ-v2 / CleanRL pixel encoders.
        conv_init = nn.initializers.orthogonal(jnp.sqrt(2.0))
        for channels, stride in [(32, 2), (64, 2), (64, 2)]:
            x = nn.Conv(
                channels,
                kernel_size=(3, 3),
                strides=(stride, stride),
                padding="SAME",
                kernel_init=conv_init,
            )(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        # Bounded LayerNorm->tanh trunk (DrQ-v2 / SAC+AE convention). In this
        # asymmetric design the CNN is trained ONLY by the deterministic policy
        # gradient through the privileged state-critic's Q (Pinto et al. 2018,
        # arXiv:1710.06542); the tanh-bounded [-1, 1] features keep that lone,
        # comparatively brittle gradient well-conditioned.
        x = nn.Dense(self.embedding_dim, kernel_init=conv_init)(x)
        x = nn.LayerNorm()(x)
        return nn.tanh(x)


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
