"""In-loop RGB vision for the dm_control_suite locomotion envs.

The framework implements the in-loop MJWarp render path only for CartpoleBalance;
CheetahRun / WalkerWalk / HopperHop all `raise NotImplementedError` for
`vision=True`. These subclasses add cartpole's pipeline (a render context built
from the model; `refit_bvh` -> `render` -> `get_rgb` -> 64x64 grayscale 3-frame
stack into `obs["pixels/view_0"]`, inline in reset/step so the adapter's vec
wrapper batches them exactly as it batches cartpole). The task reward is unchanged
-- only the actor's observation becomes pixels (asymmetric / privileged-critic).

Requires `mujoco-warp` + the render shims in `playground_adapter`
(`ensure_mjwarp_graphmode` / `ensure_mjx_render_compat`). The locomotion cameras
track the body (trackcom); verified in-frame.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import cheetah as _cheetah
from mujoco_playground._src.dm_control_suite import hopper as _hopper
from mujoco_playground._src.dm_control_suite import walker as _walker

# Mirrors cartpole.default_vision_config(); cam_active selects the first ('side' /
# 'cam0', trackcom) camera and disables 'back' for all three envs.
_VISION_KWARGS = dict(
    cam_res=(64, 64),
    use_textures=False,
    use_shadows=False,
    render_rgb=(True,),
    render_depth=(False,),
    enabled_geom_groups=[0, 1, 2],
    cam_active=(True, False),
)


def _arrayify(data):
    """Coerce non-array (scalar ``TypedFloat``, e.g. ``time=0.0``) leaves to JAX
    arrays so MJWarp's ``refit_bvh`` broadcast-to-nworld works."""
    return jax.tree_util.tree_map(
        lambda x: x if hasattr(x, "shape") else jp.asarray(x, jp.float32), data
    )


def _mk_config(default_config_fn, episode_length, impl):
    cfg = default_config_fn()
    cfg.vision = False  # parents raise for vision=True; we attach the context ourselves
    cfg.impl = impl
    cfg.episode_length = int(episode_length)
    return cfg


def _setup_rc(env, nworld):
    kw = dict(_VISION_KWARGS)
    kw["nworld"] = int(nworld)
    env._rc = mjx.create_render_context(mjm=env._mj_model, **kw)
    env._rc_pytree = env._rc.pytree()


def _render_gray(env, data):
    data = _arrayify(data)
    data = mjx.refit_bvh(env.mjx_model, data, env._rc_pytree)
    out = mjx.render(env.mjx_model, data, env._rc_pytree)  # shim -> (rgb, depth, data)
    rgb = mjx.get_rgb(env._rc_pytree, 0, out[0])
    gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5
    return data, gray


def _vision_reset(env, base_cls, rng):
    state = base_cls.reset(env, rng)
    data, gray = _render_gray(env, state.data)
    fs = jp.repeat(gray, 3, axis=-1)
    info = dict(state.info)
    info["frame_stack"] = fs
    return mjx_env.State(data, {"pixels/view_0": fs}, state.reward, state.done, state.metrics, info)


def _vision_step(env, base_cls, state, action):
    s2 = base_cls.step(env, state, action)
    data, gray = _render_gray(env, s2.data)
    prev = state.info["frame_stack"]
    fs = jp.concatenate([prev[..., 1:], gray], axis=-1)
    info = dict(s2.info)
    info["frame_stack"] = fs
    return mjx_env.State(data, {"pixels/view_0": fs}, s2.reward, s2.done, s2.metrics, info)


class CheetahRunVision(_cheetah.Run):
    def __init__(self, nworld: int, episode_length: int = 1000, impl: str = "warp"):
        super().__init__(config=_mk_config(_cheetah.default_config, episode_length, impl))
        _setup_rc(self, nworld)

    def reset(self, rng):
        return _vision_reset(self, _cheetah.Run, rng)

    def step(self, state, action):
        return _vision_step(self, _cheetah.Run, state, action)


class WalkerWalkVision(_walker.PlanarWalker):
    def __init__(self, nworld: int, episode_length: int = 1000, impl: str = "warp"):
        super().__init__(move_speed=_walker.WALK_SPEED,
                         config=_mk_config(_walker.default_config, episode_length, impl))
        _setup_rc(self, nworld)

    def reset(self, rng):
        return _vision_reset(self, _walker.PlanarWalker, rng)

    def step(self, state, action):
        return _vision_step(self, _walker.PlanarWalker, state, action)


class HopperHopVision(_hopper.Hopper):
    def __init__(self, nworld: int, episode_length: int = 1000, impl: str = "warp"):
        super().__init__(hopping=True,
                         config=_mk_config(_hopper.default_config, episode_length, impl))
        _setup_rc(self, nworld)

    def reset(self, rng):
        return _vision_reset(self, _hopper.Hopper, rng)

    def step(self, state, action):
        return _vision_step(self, _hopper.Hopper, state, action)


VISION_ENVS = {
    "CheetahRun": CheetahRunVision,
    "WalkerWalk": WalkerWalkVision,
    "HopperHop": HopperHopVision,
}
