"""Measure, on-policy, how often each hand-written rule admits each skill — env-agnostic.

`tools/rule_coverage.py` samples the state *box* uniformly, which is how the cartpole rule was
found to fire `recover_balance` on 92.6% of it. That is a statement about the box, not about
the states a trained policy visits, and the two can differ by a lot. This measures the same
thing along an actual greedy rollout, and additionally reports quantiles of each driver feature
so a revised threshold can be placed at a quantile of the *successful* policy's visitation
(runbook §5 stage 1).

Unlike `tools/rule_threshold_scan.py` (go1-specific, hand-written predicates) this calls the
policy module's own `skill_mask` / `symbolic_meta_policy`, so it cannot drift from the rule it
is describing and works for any registered policy.

    JAX_PLATFORMS=cpu python tools/rule_coverage_measured.py \
        --checkpoint runs/verify/cartpole_balance_neural_v2_s0.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus_continuous.envs.playground_adapter import (  # noqa: E402
    build_playground_env,
    get_actor_obs,
    get_policy_obs,
)
from nexus_continuous.policies.registry import load_policy_module  # noqa: E402
from tools.robustness_eval import (  # noqa: E402
    _build_networks,
    _load_checkpoint,
    _make_normalizer,
    _make_selector,
    _restore_params,
    init_hold_state,
)

QUANTILES = (0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=31000)
    args = ap.parse_args()

    ck = _load_checkpoint(args.checkpoint)
    cfg = dict(ck["config"])
    pm = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    num_skills = int(pm.NUM_SKILLS)
    names = list(getattr(pm, "SKILL_NAMES", [str(i) for i in range(num_skills)]))

    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NORMALIZE_REWARD"] = False
    bundle = build_playground_env(eval_cfg)
    env, params = bundle.env, bundle.env_params
    lo, hi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
    actor, meta_q = _build_networks(cfg, num_skills, bundle.action_dim, (hi - lo) / 2.0,
                                    (hi + lo) / 2.0)
    raw0, _ = env.reset(jax.random.split(jax.random.PRNGKey(0), args.num_envs), params)
    ar = jax.random.split(jax.random.PRNGKey(0), num_skills)
    fresh = jax.vmap(lambda k: actor.init(k, get_actor_obs(raw0))["params"])(ar)
    actor_params = _restore_params(fresh, ck["runner_state"]["0"]["actor"]["params"])
    meta_type = str(cfg.get("META_POLICY_TYPE", "nesy")).lower()
    meta_params = None
    if meta_type in ("neural", "nesy"):
        fm = meta_q.init(jax.random.PRNGKey(1), get_actor_obs(raw0))["params"]
        meta_params = _restore_params(fm, ck["runner_state"]["0"]["meta"]["params"])
    normalize = _make_normalizer(ck.get("normalization_stats"), cfg.get("NORMALIZE_OBS", True))
    di = int(cfg.get("META_DECISION_INTERVAL", 1))
    select = _make_selector(actor, meta_q, actor_params, meta_params, pm, meta_type, lo, hi, di)

    rng = jax.random.PRNGKey(args.seed)
    raw, state = env.reset(jax.random.split(rng, args.num_envs), params)
    obs = normalize(raw)
    hold = init_hold_state(args.num_envs, di)
    masks: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    rule_pick: list[np.ndarray] = []
    for _ in range(args.steps):
        action, _skill, hold = select(obs, hold)
        pobs = get_policy_obs(obs)
        if hasattr(pm, "skill_mask"):
            masks.append(np.asarray(pm.skill_mask(pobs), dtype=bool))
        if hasattr(pm, "symbolic_meta_policy"):
            rule_pick.append(np.asarray(pm.symbolic_meta_policy(pobs)))
        f = pm._features(pobs)
        feats.append(np.stack([np.asarray(v) for v in f[: min(len(f), 6)]], axis=-1))
        rng, sr = jax.random.split(rng)
        raw, state, _r, done, _i = env.step(jax.random.split(sr, args.num_envs), state, action,
                                            params)
        obs = normalize(raw)
        if hold is not None:
            hold = (hold[0], hold[1], done.astype(bool))

    print(f"\n{args.checkpoint}  ({meta_type}, {args.num_envs * args.steps} states)\n")
    if masks:
        m = np.concatenate(masks, axis=0)
        print("  mask admissibility, ON-POLICY (vs rule_coverage.py's uniform state box):")
        for i, nm in enumerate(names):
            print(f"    {i}_{nm:18s} admitted on {100 * m[:, i].mean():5.1f}% of visited states")
    if rule_pick:
        r = np.concatenate(rule_pick, axis=0)
        print("\n  symbolic rule's own choice (the `symbolic` arm's policy):")
        for i, nm in enumerate(names):
            print(f"    {i}_{nm:18s} chosen on   {100 * (r == i).mean():5.1f}%")
    arr = np.concatenate(feats, axis=0)
    print("\n  driver quantiles (feature order as returned by the module's _features):")
    for j in range(arr.shape[1]):
        v = np.abs(arr[:, j])
        qs = " ".join(f"q{int(q * 100):02d}={np.quantile(v, q):.4f}" for q in QUANTILES)
        print(f"    |f{j}| mean={v.mean():.4f}  {qs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
