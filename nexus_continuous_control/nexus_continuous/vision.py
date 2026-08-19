"""Optional RGB actor components for the skill-agent extension.

These modules are intentionally separate from the default state-based algorithm.
Use them when the Playground observation wrapper supplies `pixels` and
`proprioception` fields. The recommended setup is a privileged critic:
actor_i(rgb, proprioception) -> action, critic_i(state, action) -> Q_i.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, NamedTuple

import flax.linen as nn
import jax
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
    """Pixel skill actor, optionally with an auxiliary pixel->state head.

    ``aux_state_dim > 0`` adds a linear head that regresses the privileged state
    (qpos+qvel) from the CNN latent. This exists because the in-loop ablation of
    2026-08-17 showed the encoder receives too weak a signal to learn perception:
    trained only by the deterministic policy gradient through the privileged
    critic, the actor's output varied by 0.005-0.151% of the action range across
    completely different frames -- effectively blind, with the state-based meta
    doing the control. The aux head gives the encoder the same DENSE supervised
    signal that makes the distillation arm reach r ~ 0.99 on identical inputs,
    but inside the RL loop. Predicting state (not reconstructing pixels) is the
    cheaper, control-relevant choice.
    """

    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    hidden_sizes: Sequence[int] = (256, 256)
    embedding_dim: int = 128
    aux_state_dim: int = 0

    @nn.compact
    def __call__(
        self,
        pixels: jnp.ndarray,
        proprioception: jnp.ndarray,
        return_aux: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray | None]:
        z = RGBEncoder(self.embedding_dim)(pixels)
        x = jnp.concatenate([z, proprioception], axis=-1)
        x = MLP(self.hidden_sizes, self.action_dim, output_scale=0.01)(x)
        action = jnp.tanh(x) * self.action_scale + self.action_bias
        # Built whenever enabled -- NOT only when return_aux is set -- so the
        # parameter tree is identical between init and every apply call.
        state_pred = (
            nn.Dense(self.aux_state_dim, name="aux_state")(z)
            if self.aux_state_dim > 0
            else None
        )
        if return_aux:
            return action, state_pred
        return action


# --------------------------------------------------------------------------- #
# Shared-encoder variant (config RGB_SHARED_ENCODER, opt-in).                 #
# --------------------------------------------------------------------------- #
#
# In the default RGB path every one of the N skill actors carries its OWN CNN
# (the whole VisionSkillActor is `jax.vmap`-ed over N init keys). Two measured
# problems follow:
#
#   1. Signal starvation. The encoder's only gradient is the deterministic
#      policy gradient through the privileged critic, and each of the N copies
#      receives only its own 1/N share of it. The 2026-08-17 ablation found the
#      actors effectively BLIND (output moved 0.005-0.151% of the action range
#      across completely different frames) and needed an auxiliary pixel->state
#      loss to see at all.
#   2. Wasted compute. The trainer closes over `obs_pixels` WITHOUT vmapping it,
#      so today the SAME image is pushed through N independent CNNs every step.
#
# Splitting VisionSkillActor into (one shared trunk) + (N small heads) fixes
# both: the trunk gets the summed gradient of all N heads and runs once.
#
# The two modules below are ADDITIVE. `VisionSkillActor` is deliberately left
# untouched so that runs with RGB_SHARED_ENCODER off keep a byte-identical
# parameter tree, and old checkpoints keep loading.


class SharedRGBTrunk(nn.Module):
    """The CNN encoder (+ optional aux pixel->state head) shared by all skills.

    Its parameters live OUTSIDE the skill axis: one copy, N times the gradient.
    The aux head belongs here rather than on the per-skill heads because its
    regression target (the privileged state) is skill-INDEPENDENT -- N per-skill
    aux heads are N redundant solutions to the same problem. Keeping the
    existing ``jnp.mean`` reduction over the (now absent) skill axis means the
    loss magnitude is roughly unchanged, so an ``RGB_AUX_STATE_COEF`` tuned for
    the unshared path ports over without retuning.
    """

    embedding_dim: int = 128
    aux_state_dim: int = 0

    @nn.compact
    def __call__(
        self, pixels: jnp.ndarray, return_aux: bool = False
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray | None]:
        z = RGBEncoder(self.embedding_dim)(pixels)
        # Built whenever enabled -- NOT only when return_aux is set -- so the
        # parameter tree is identical between init and every apply call.
        state_pred = (
            nn.Dense(self.aux_state_dim, name="aux_state")(z)
            if self.aux_state_dim > 0
            else None
        )
        if return_aux:
            return z, state_pred
        return z


class VisionSkillHead(nn.Module):
    """Per-skill action head on top of a SharedRGBTrunk latent.

    Its parameters STAY stacked ``[num_skills, ...]`` and are vmapped exactly
    like the unshared actor's, so every downstream consumer that slices skill
    ``k`` out of the actor tree keeps working.
    """

    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    hidden_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, latent: jnp.ndarray, proprioception: jnp.ndarray) -> jnp.ndarray:
        # These three lines are DUPLICATED from VisionSkillActor on purpose.
        # Refactoring VisionSkillActor to delegate to this module would rename
        # its `MLP_0` submodule to `VisionSkillHead_0/MLP_0`, changing the
        # parameter tree of the EXISTING (unshared) path and invalidating every
        # checkpoint plus the byte-identity guarantee. Do not "clean this up".
        x = jnp.concatenate([latent, proprioception], axis=-1)
        x = MLP(self.hidden_sizes, self.action_dim, output_scale=0.01)(x)
        return jnp.tanh(x) * self.action_scale + self.action_bias


class RGBActorFns(NamedTuple):
    """Everything a caller needs to init/run the RGB skill actors.

    One place that knows about the two possible parameter layouts, so the
    trainer and the offline analysis scripts (`rgb_pixel_ablation`,
    `rgb_pixel_sensitivity`, `rgb_inloop_visualize`) cannot drift apart again.

    Parameter layouts:
      * shared_encoder=False -> the historical layout: the whole
        ``VisionSkillActor`` tree stacked ``[num_skills, ...]``.
      * shared_encoder=True  -> ``{"encoder": <one trunk tree, no skill axis>,
        "heads": <VisionSkillHead tree stacked [num_skills, ...]>}``.

    The two are mutually incompatible; checkpoints must record which one they
    hold (see `rgb_pixel_ablation.py`).
    """

    shared_encoder: bool
    #: ``VisionSkillActor`` in the unshared layout, else ``None``.
    actor: nn.Module | None
    #: ``SharedRGBTrunk`` / ``VisionSkillHead`` in the shared layout, else ``None``.
    trunk: nn.Module | None
    head: nn.Module | None
    #: ``(rng, num_skills, dummy_pixels, dummy_proprio) -> params``
    init: Callable[..., Any]
    #: ``(params, pixels, proprio) -> [num_skills, batch, action_dim]``
    apply: Callable[..., jnp.ndarray]
    #: ``(params, pixels, proprio) -> (actions, state_pred)``; ``state_pred`` is
    #: ``[num_skills, batch, D]`` unshared and ``[batch, D]`` shared (see
    #: :attr:`aux_has_skill_axis`), or ``None`` when the aux head is disabled.
    apply_aux: Callable[..., tuple[jnp.ndarray, jnp.ndarray | None]]
    #: ``(params, pixels) -> latent``; ``[batch, E]`` shared, ``[N, batch, E]``
    #: unshared.
    encode: Callable[..., jnp.ndarray]
    #: Module apply used as the flax ``TrainState.apply_fn`` (static metadata).
    train_state_apply: Callable[..., Any]
    #: Whether ``apply_aux``'s ``state_pred`` carries a leading skill axis.
    aux_has_skill_axis: bool


def build_rgb_actor_fns(
    *,
    action_dim: int,
    action_scale: jnp.ndarray,
    action_bias: jnp.ndarray,
    hidden_sizes: Sequence[int] = (256, 256),
    embedding_dim: int = 128,
    aux_state_dim: int = 0,
    shared_encoder: bool = False,
) -> RGBActorFns:
    """Build the RGB skill-actor modules and their init/apply closures.

    ``shared_encoder=False`` reproduces the pre-existing code path operation for
    operation (same module, same ``jax.random.split(rng, num_skills)`` key
    consumption, same vmapped apply), so seeds reproduce exactly.
    """

    if not shared_encoder:
        actor = VisionSkillActor(
            action_dim=action_dim,
            action_scale=action_scale,
            action_bias=action_bias,
            hidden_sizes=tuple(hidden_sizes),
            embedding_dim=embedding_dim,
            aux_state_dim=aux_state_dim,
        )

        def _init(rng, num_skills, dummy_pixels, dummy_proprio):
            actor_rngs = jax.random.split(rng, num_skills)
            return jax.vmap(lambda k: actor.init(k, dummy_pixels, dummy_proprio)["params"])(
                actor_rngs
            )

        def _apply(params, pixels, proprio):
            return jax.vmap(lambda p: actor.apply({"params": p}, pixels, proprio))(params)

        def _apply_aux(params, pixels, proprio):
            return jax.vmap(
                lambda p: actor.apply({"params": p}, pixels, proprio, return_aux=True)
            )(params)

        def _encode(params, pixels):
            encoder = RGBEncoder(embedding_dim)
            return jax.vmap(lambda p: encoder.apply({"params": p}, pixels))(
                params["RGBEncoder_0"]
            )

        return RGBActorFns(
            shared_encoder=False,
            actor=actor,
            trunk=None,
            head=None,
            init=_init,
            apply=_apply,
            apply_aux=_apply_aux,
            encode=_encode,
            train_state_apply=actor.apply,
            aux_has_skill_axis=True,
        )

    trunk = SharedRGBTrunk(embedding_dim=embedding_dim, aux_state_dim=aux_state_dim)
    head = VisionSkillHead(
        action_dim=action_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        hidden_sizes=tuple(hidden_sizes),
    )

    def _init(rng, num_skills, dummy_pixels, dummy_proprio):
        enc_rng, head_seed = jax.random.split(rng)
        head_rngs = jax.random.split(head_seed, num_skills)
        dummy_latent = jnp.zeros((dummy_pixels.shape[0], embedding_dim), dtype=jnp.float32)
        return {
            "encoder": trunk.init(enc_rng, dummy_pixels)["params"],
            "heads": jax.vmap(lambda k: head.init(k, dummy_latent, dummy_proprio)["params"])(
                head_rngs
            ),
        }

    def _heads(params, latent, proprio):
        return jax.vmap(lambda p: head.apply({"params": p}, latent, proprio))(params["heads"])

    def _encode(params, pixels):
        return trunk.apply({"params": params["encoder"]}, pixels)

    def _apply(params, pixels, proprio):
        # ONE CNN pass for all N skills (the unshared path does N identical ones).
        return _heads(params, _encode(params, pixels), proprio)

    def _apply_aux(params, pixels, proprio):
        latent, state_pred = trunk.apply(
            {"params": params["encoder"]}, pixels, return_aux=True
        )
        return _heads(params, latent, proprio), state_pred

    return RGBActorFns(
        shared_encoder=True,
        actor=None,
        trunk=trunk,
        head=head,
        init=_init,
        apply=_apply,
        apply_aux=_apply_aux,
        encode=_encode,
        train_state_apply=head.apply,
        aux_has_skill_axis=False,
    )
