"""Is Go1's `not_fallen` predicate actually using roll and pitch, or only height?

`go1_joystick._features` reads roll/pitch from the adapter's semantic info dict and falls back to
`safe_index(x, 1)` / `safe_index(x, 2)` — raw observation components — when those keys are absent.
The adapter only populates them if `self._env._imu_site_id` resolves
(`playground_adapter.py:309`). If that guard fails on Go1, then `not_fallen`'s
`|roll| < 0.6 & |pitch| < 0.6` terms are computed from arbitrary observation entries, and the
tracking-success metric behind every Go1 number in this campaign would be height-only.

A video review flagged the possibility: `raw_feature_diagnostics.csv` reports max |roll|, |pitch|
of 0.012 rad while an independent on-policy scan of the same policy family reported |roll| q95 =
0.27. That is a 20x discrepancy, and batch-mean sign cancellation is only one explanation.

This settles it by comparing what `_features` returns against roll/pitch recomputed from the
MuJoCo IMU site orientation for the same states.

    JAX_PLATFORMS=cpu python tools/check_go1_roll_pitch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus_continuous.envs.playground_adapter import build_playground_env, get_policy_obs
from nexus_continuous.policies import go1_joystick


def main() -> int:
    cfg = {"ENV_NAME": "Go1JoystickFlatTerrain", "NUM_ENVS": 8,
           "NORMALIZE_OBS": False, "NORMALIZE_REWARD": False}
    bundle = build_playground_env(cfg)
    env, params = bundle.env, bundle.env_params

    rng = jax.random.PRNGKey(0)
    raw, state = env.reset(jax.random.split(rng, 8), params)

    # Step a few times with random actions so the bodies leave the reset pose and roll/pitch
    # become non-trivial; a check run only at reset would pass on a broken pipeline.
    for i in range(25):
        rng, k = jax.random.split(rng)
        act = jax.random.uniform(k, (8, bundle.action_dim), minval=-1.0, maxval=1.0)
        raw, state, _r, _d, info = env.step(jax.random.split(k, 8), state, act, params)

    pobs = get_policy_obs(raw)
    sem = pobs.get("policy_info", {}) if isinstance(pobs, dict) else {}
    print("semantic info keys present:", sorted(sem.keys())[:40] or "NONE")
    for key in ("roll", "base_roll", "pitch", "base_pitch", "base_height"):
        print(f"  {key:12s} present={key in sem}")

    feats = go1_joystick._features(pobs)
    height, roll, pitch = feats[0], feats[1], feats[2]
    print("\n_features() returned:")
    print("  height", np.asarray(height))
    print("  roll  ", np.asarray(roll))
    print("  pitch ", np.asarray(pitch))

    # Ground truth from the IMU site orientation, computed here independently.
    inner = env
    for _ in range(8):
        nxt = getattr(inner, "_env", None)
        if nxt is None:
            break
        inner = nxt
    imu_id = getattr(inner, "_imu_site_id", None)
    print("\n_imu_site_id on the unwrapped env:", imu_id)

    data = getattr(getattr(state, "env_state", state), "data", None)
    depth = 0
    st = state
    while data is None and depth < 8:
        st = getattr(st, "env_state", None)
        if st is None:
            break
        data = getattr(st, "data", None)
        depth += 1
    if data is None or imu_id is None:
        print("could not reach mjx data or imu site; inconclusive")
        return 2

    rot = data.site_xmat[..., int(imu_id), :, :]
    up = rot[..., :, 2]
    gt_roll = np.asarray(jnp.arctan2(up[..., 1], up[..., 2]))
    gt_pitch = np.asarray(
        jnp.arctan2(-up[..., 0], jnp.sqrt(jnp.square(up[..., 1]) + jnp.square(up[..., 2]))))
    print("  ground-truth roll ", gt_roll)
    print("  ground-truth pitch", gt_pitch)

    ok_roll = np.allclose(np.asarray(roll), gt_roll, atol=1e-4)
    ok_pitch = np.allclose(np.asarray(pitch), gt_pitch, atol=1e-4)
    print(f"\nroll matches ground truth : {ok_roll}")
    print(f"pitch matches ground truth: {ok_pitch}")
    if ok_roll and ok_pitch:
        print("\nVERDICT: PASS — not_fallen uses real roll/pitch; the metric is not height-only.")
        return 0
    print("\nVERDICT: FAIL — _features is NOT returning true roll/pitch. Every Go1 "
          "tracking-success number is affected; not_fallen degenerates toward height-only.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
