"""cartpole_balance with REVISED rule thresholds (runbook §5 stage 1). Experiment tag: `rules2`.

Everything except `skill_mask` and `symbolic_meta_policy` is imported unchanged from
`cartpole_balance`: same skills, same skill rewards, same task metrics. This is a threshold
revision and nothing else, so a difference between `nesy` and `nesy·rules2` is attributable to
the thresholds alone.

What the measurement actually showed (and how it contradicts the board)
----------------------------------------------------------------------
The hypothesis on the board came from `tools/rule_coverage.py`, which samples the state box
UNIFORMLY and found the mask admits `recover_balance` on 92.6% of it. §5 stage 1 therefore
predicted an over-firing rule, to be tightened.

`tools/rule_threshold_scan.py` measures the distribution a *successful* policy actually visits
(`cartpole_balance_neural_explore_s0`, success 0.894, 12800 samples) and the result is the
opposite:

    shipped skill_mask admissibility, ON-POLICY:
      recover_balance   0.0%      (needs |angle|>0.08 or |ang_vel|>0.7)
      center_cart      39.8%      (needs |cart|>0.12)
      damp_motion     100.0%
    shipped symbolic_meta_policy:  damp 98.9%, center 1.1%, recover 0.0%

Both numbers are true and they are not in conflict: uniformly over the box the pole is usually
far from upright, so `recover` is admissible almost everywhere; along a working trajectory the
pole is never far from upright (|angle| q99 = 0.0314 rad ≈ 1.8°), so `recover` is admissible
nowhere. The operative defect is the second one. On-policy the mask leaves the nesy meta with
`damp_motion` always, `center_cart` sometimes, and `recover_balance` never — it collapses a
3-way choice to roughly a 1-way one, which is a mechanism for nesy·explore (0.437) losing to
neural·explore (0.894) that needs no over-firing at all.

So the revision here LOOSENS the mask instead of tightening it, while still doing what §5 asks:
each threshold sits at a quantile of the successful policy's own visitation.

    recover  |angle|  > 0.0259  (q90)     was 0.08   ->  0.0% admissible
             |ang_vel|> 0.0509  (q90)     was 0.7
    center   |cart|   > 0.2098  (q80)     was 0.12   -> 39.8% admissible
    damp     always                       unchanged

Known tension, recorded rather than resolved
--------------------------------------------
The quantiles are strongly seed-dependent, because a better policy visits a narrower band.
Seed 1 of the same cell is ~5x tighter than seed 0:

                       s0 (used here)   s1
    |angle|   q90         0.0259       0.0049
    |ang_vel| q90         0.0509       0.0193
    |cart|    q80         0.2098       0.0893

s0 is used deliberately: it is the WIDER of the two, and thresholds taken from s1 would declare
a pole balanced to 0.3° to be in the "worst 10% of states", which is a quantile in the
arithmetic sense and nothing in the physical sense. This is a real limitation of the
quantile method on an env whose successful policies are near-perfect, and it is the reason
cartpole's outcome here should be read as evidence about attribution, not as a tuned result.
If `rules2` does not move cartpole, §5's stage 2 (the decomposition itself — `recover`/`center`/
`damp` are competing corrections rather than stages) is the live hypothesis, and it gets its
own tag `rules3`.

Pre-registered success criterion (runbook §5, unchanged): cartpole `nesy·rules2` closes >= half
the gap from `nesy·explore` (0.437) to `neural·explore` (0.894), i.e. success >= 0.666.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nexus_continuous.policies.cartpole_balance import (  # noqa: F401  (re-exported unchanged)
    NUM_SKILLS,
    SKILL_NAMES,
    _features,
    diagnostics,
    skill_rewards,
    task_metrics,
)

# Revised thresholds; see module docstring for provenance.
RECOVER_ANGLE = 0.0259      # q90 of |pole_angle| under a working policy
RECOVER_ANG_VEL = 0.0509    # q90 of |pole_angular_velocity|
CENTER_CART = 0.2098        # q80 of |cart_position|


def symbolic_meta_policy(obs: Any, info: Any | None = None) -> jnp.ndarray:
    """Rule-only selection at the revised thresholds.

    The shipped rule's urgent band (0.20 rad / 1.5 rad/s) is never entered on-policy, so the
    symbolic arm sits in `damp_motion` on 98.9% of decisions and the rule is decorative. At the
    revised band it hands off to `recover_balance` on roughly the worst 10% of states, which is
    what a rule-based controller is supposed to do.
    """
    cart, angle, _cart_vel, ang_vel = _features(obs, info)
    urgent_angle = (jnp.abs(angle) > RECOVER_ANGLE) | (jnp.abs(ang_vel) > RECOVER_ANG_VEL)
    off_center = jnp.abs(cart) > CENTER_CART
    return jnp.where(urgent_angle, 0, jnp.where(off_center, 1, 2)).astype(jnp.int32)


def skill_mask(obs: Any, info: Any | None = None) -> jnp.ndarray:
    cart, angle, _cart_vel, ang_vel = _features(obs, info)
    recover = (jnp.abs(angle) > RECOVER_ANGLE) | (jnp.abs(ang_vel) > RECOVER_ANG_VEL)
    center = jnp.abs(cart) > CENTER_CART
    damp = jnp.ones_like(recover, dtype=bool)
    return jnp.stack([recover, center, damp], axis=-1)


def explain_policy() -> str:
    return (
        "CartpoleBalance (rules2): recover_balance when |pole angle| > 0.026 rad or "
        "|angular velocity| > 0.051 rad/s (the worst ~10% of states a working policy visits); "
        "center_cart when |cart position| > 0.21; damp_motion otherwise."
    )
