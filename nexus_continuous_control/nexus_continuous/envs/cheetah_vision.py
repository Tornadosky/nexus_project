"""In-loop RGB vision for CheetahRun -- ports CartpoleBalance's MJWarp pipeline.

mujoco_playground implements the in-loop MJWarp render path only for
CartpoleBalance; ``dm_control_suite/cheetah.py`` raises ``NotImplementedError`` for
``vision=True``. This subclass adds the *same* pipeline cartpole uses: build a
render context from the model, and on reset/step render a 64x64 grayscale 3-frame
stack into ``obs["pixels/view_0"]`` (inline, single-env -- so the adapter's vec
wrapper + render context batch it exactly as they batch cartpole).

The forward-speed task reward is unchanged -- only the actor's observation becomes
pixels (the asymmetric / privileged-critic design: critic + meta stay on state).

Requires ``mujoco-warp`` and the render shims in ``playground_adapter``
(``ensure_mjwarp_graphmode`` / ``ensure_mjx_render_compat``). Cheetah's cameras are
tracking (trackcom); this relies on the MJWarp render context following the body,
which is verified by inspecting a rendered rollout.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import cheetah as _cheetah


def _arrayify(data):
    """Coerce any non-array (e.g. scalar ``TypedFloat``) pytree leaf to a JAX array.

    cheetah's state reset sets ``time=0.0`` as a Python float, which lands as a
    scalar ``TypedFloat`` with no ``.shape``; MJWarp's ``refit_bvh`` broadcast to
    ``nworld`` then fails. Making every leaf a real JAX array fixes that.
    """

    return jax.tree_util.tree_map(
        lambda x: x if hasattr(x, "shape") else jp.asarray(x, jp.float32), data
    )

# Mirrors cartpole.default_vision_config(); cam_active selects cheetah's 'side'
# (index 0, trackcom) camera and disables 'back' (index 1).
_VISION_KWARGS: dict[str, Any] = dict(
    cam_res=(64, 64),
    use_textures=False,
    use_shadows=False,
    render_rgb=(True,),
    render_depth=(False,),
    enabled_geom_groups=[0, 1, 2],
    cam_active=(True, False),
)


class CheetahRunVision(_cheetah.Run):
    """CheetahRun with an in-loop MJWarp pixel observation (64x64 gray, 3-stack)."""

    def __init__(self, nworld: int, episode_length: int = 1000, impl: str = "warp"):
        cfg = _cheetah.default_config()
        cfg.vision = False  # parent raises for vision=True; we attach the context below
        cfg.impl = impl
        cfg.episode_length = int(episode_length)
        super().__init__(config=cfg)
        kw = dict(_VISION_KWARGS)
        kw["nworld"] = int(nworld)
        self._rc = mjx.create_render_context(mjm=self._mj_model, **kw)
        self._rc_pytree = self._rc.pytree()

    def _render_gray(self, data):
        """refit + render -> (data, [H,W,1] centered gray). Mirrors cartpole."""
        data = _arrayify(data)
        data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        out = mjx.render(self.mjx_model, data, self._rc_pytree)  # shim -> (rgb, depth, data)
        rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
        gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5
        return data, gray

    def reset(self, rng):
        state = super().reset(rng)
        data, gray = self._render_gray(state.data)
        frame_stack = jp.repeat(gray, 3, axis=-1)
        info = dict(state.info)
        info["frame_stack"] = frame_stack
        obs = {"pixels/view_0": frame_stack}
        return mjx_env.State(data, obs, state.reward, state.done, state.metrics, info)

    def step(self, state, action):
        s2 = super().step(state, action)
        data, gray = self._render_gray(s2.data)
        prev = state.info["frame_stack"]
        frame_stack = jp.concatenate([prev[..., 1:], gray], axis=-1)
        info = dict(s2.info)
        info["frame_stack"] = frame_stack
        obs = {"pixels/view_0": frame_stack}
        return mjx_env.State(data, obs, s2.reward, s2.done, s2.metrics, info)
