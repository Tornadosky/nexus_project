"""go1_joystick with REVISED rule thresholds (runbook §5 stage 1). Experiment tag: `rules2`.

Everything except the `recover` admissibility thresholds is imported unchanged from
`go1_joystick`: same skills, same skill rewards, same task metrics, same command thresholds.
This is a threshold revision and nothing else, so a difference between `nesy` and `nesy·rules2`
is attributable to the thresholds alone.

Why
---
`tools/rule_threshold_scan.py` measured the state distribution actually visited by a working
policy (`go1_joystick_neural_v2_s2`, `no_fall_rate` 0.944, 19200 samples):

    height   mean 0.2840   q05 0.2278  q10 0.2422  q50 0.2870
    |roll|   mean 0.0995   q90 0.2266  q95 0.2749
    |pitch|  mean 0.0932   q90 0.2055  q95 0.2582

The shipped mask is `height < 0.28 | |roll| > 0.25 | |pitch| > 0.25` and admits `recover` on
**44.4%** of visited states. The height term alone accounts for 40.2% of that — because 0.28
sits essentially at the *median* height of a healthy policy (0.287). A predicate meant to catch
"unsafe low body height" is firing on nearly half of normal operation. The roll and pitch terms
are fine (7.5% and 5.6%, i.e. roughly their q93/q94).

So only the height threshold is genuinely miscalibrated, but §5 says to set each threshold at a
quantile of the successful policy's visitation, and the q05 choice below lands the combined
predicate at **11.8%** admissibility — inside the "top 10-20% of states" target.

    height  0.28 -> 0.2278  (q05)
    |roll|  0.25 -> 0.2749  (q95)
    |pitch| 0.25 -> 0.2582  (q95)

Known tension, recorded rather than resolved: 0.2278 is only 0.008 above the `fallen` predicate
(`height < 0.22`), so `recover` becomes admissible barely before the robot counts as fallen.
If `rules2` fails because recovery is triggered too late, that is informative and is exactly
the attribution §5 asks for — it would point at the skill, not the threshold, which is where
§E.1 already suggests the go1 problem lives.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.go1_joystick import (  # noqa: F401  (re-exported unchanged)
    NUM_SKILLS,
    SKILL_NAMES,
    _features,
    diagnostics,
    l2_norm,
    skill_rewards,
    task_metrics,
)

# Revised recover-admissibility thresholds; see module docstring for provenance.
RECOVER_HEIGHT = 0.2278
RECOVER_ROLL = 0.2749
RECOVER_PITCH = 0.2582


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    """Unchanged from go1_joystick.

    The symbolic cascade keys on `fallen` (height < 0.22), a different predicate from the mask,
    and the measurements say it fires correctly — the arm really is on the ground
    (`no_fall_rate` 0.126). Revising it would be a different hypothesis; `rules2` is only about
    the mask thresholds.
    """
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw = _features(obs, info)
    fallen = (height < 0.22) | (jnp.abs(roll) > 0.6) | (jnp.abs(pitch) > 0.6)
    need_turn = jnp.abs(cmd_yaw) > 0.15
    need_track = l2_norm(jnp.stack([cmd_x, cmd_y], axis=-1)) > 0.10
    return jnp.where(fallen, 3, jnp.where(need_turn, 2, jnp.where(need_track, 1, 0))).astype(
        jnp.int32
    )


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    height, roll, pitch, _vx, _vy, _yaw_rate, cmd_x, cmd_y, cmd_yaw = _features(obs, info)
    unstable = (
        (height < RECOVER_HEIGHT)
        | (jnp.abs(roll) > RECOVER_ROLL)
        | (jnp.abs(pitch) > RECOVER_PITCH)
    )
    stand = jnp.ones_like(unstable, dtype=bool)
    track = l2_norm(jnp.stack([cmd_x, cmd_y], axis=-1)) > 0.05
    turn = jnp.abs(cmd_yaw) > 0.05
    recover = unstable
    return jnp.stack([stand, track, turn, recover], axis=-1)


def explain_policy() -> str:
    return (
        "Go1Joystick rules2: identical to go1_joystick except the recover mask thresholds, "
        f"which are set at quantiles of a working policy's visitation "
        f"(height < {RECOVER_HEIGHT}, |roll| > {RECOVER_ROLL}, |pitch| > {RECOVER_PITCH}) "
        "instead of the shipped 0.28/0.25/0.25 that admitted recover on 44% of normal states."
    )
