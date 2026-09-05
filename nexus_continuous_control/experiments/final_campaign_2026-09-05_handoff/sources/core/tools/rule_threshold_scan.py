"""Measure the state distribution a SUCCESSFUL policy actually visits, and cost the rule against it.

Runbook §5 stage 1: hand-written rule thresholds should sit at a quantile of a successful
policy's visitation, not at a number chosen a priori. On go1 the shipped rule is

    recover admissible iff  height < 0.28 | |roll| > 0.25 | |pitch| > 0.25

which admits `recover` on ~27% of steps, and the `symbolic` arm (rule-only selection) sits in
`recover` on 88.7% of decisions in-distribution and 99.2% under zero commands — where it scores
0.005 and cannot stand still on flat ground. That is a broken rule, and this script produces
the evidence needed to revise it: the empirical quantiles of the drivers under a policy that
works, plus the admissibility each candidate threshold would produce.

It reports, it does not edit. Writing the revised rule is a new policy module and a new
experiment tag (`rules2`), per §5.

    JAX_PLATFORMS=cpu python tools/rule_threshold_scan.py \
        --checkpoint runs/verify/go1_joystick_neural_v2_s2.pkl --episodes 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Run as `python tools/rule_threshold_scan.py` from the repo root, where `tools` is not on the
# path but the repo root is not either when the script's own dir shadows it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus_continuous.envs.playground_adapter import build_playground_env, get_policy_obs
from nexus_continuous.policies.registry import load_policy_module
from tools.robustness_eval import (
    _build_networks,
    _load_checkpoint,
    _make_normalizer,
    _make_selector,
    _restore_params,
    init_hold_state,
)

QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--rule-module", default=None,
                    help="Cost a DIFFERENT policy module's rule against this checkpoint's "
                         "visitation (e.g. cartpole_balance_rules2). The checkpoint still "
                         "drives the rollout; only skill_mask/symbolic_meta_policy come from "
                         "this module. Lets a revised rule be checked before it is trained.")
    args = ap.parse_args()

    ck = _load_checkpoint(args.checkpoint)
    cfg = dict(ck["config"])
    policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    num_skills = int(policy_module.NUM_SKILLS)
    # The rollout is always driven by the checkpoint's own policy; only the rule being COSTED
    # may come from elsewhere, so a revised rule is scored against the same visitation.
    rule_module = load_policy_module(args.rule_module) if args.rule_module else policy_module
    if args.rule_module:
        print(f"costing rule from {args.rule_module} against {cfg.get('POLICY')}'s visitation")

    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NORMALIZE_REWARD"] = False
    bundle = build_playground_env(eval_cfg)
    env, params = bundle.env, bundle.env_params
    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    action_scale = (action_high - action_low) / 2.0

    actor, meta_q = _build_networks(cfg, num_skills, bundle.action_dim, action_scale,
                                    (action_high + action_low) / 2.0)
    raw0, _ = env.reset(jax.random.split(jax.random.PRNGKey(0), args.num_envs), params)
    from nexus_continuous.envs.playground_adapter import get_actor_obs

    ar = jax.random.split(jax.random.PRNGKey(0), num_skills)
    fresh_actor = jax.vmap(lambda k: actor.init(k, get_actor_obs(raw0))["params"])(ar)
    actor_params = _restore_params(fresh_actor, ck["runner_state"]["0"]["actor"]["params"])
    meta_type = str(cfg.get("META_POLICY_TYPE", "nesy")).lower()
    meta_params = None
    if meta_type in ("neural", "nesy"):
        fresh_meta = meta_q.init(jax.random.PRNGKey(1), get_actor_obs(raw0))["params"]
        meta_params = _restore_params(fresh_meta, ck["runner_state"]["0"]["meta"]["params"])

    normalize_obs = _make_normalizer(ck.get("normalization_stats"), cfg.get("NORMALIZE_OBS", True))
    select = _make_selector(actor, meta_q, actor_params, meta_params, policy_module,
                            meta_type, action_low, action_high,
                            int(cfg.get("META_DECISION_INTERVAL", 1)))

    rng = jax.random.PRNGKey(args.seed)
    raw, state = env.reset(jax.random.split(rng, args.num_envs), params)
    obs = normalize_obs(raw)
    hold = init_hold_state(args.num_envs, int(cfg.get("META_DECISION_INTERVAL", 1)))
    # Drivers come from the module's own `diagnostics()` rather than a hardcoded slice of
    # `_features`: every policy module implements it with named keys, so the same scan works on
    # cartpole (§5's stage-1 target) without a second copy of this rollout. Admissibility is
    # MEASURED by calling `skill_mask` on the visited states, not re-derived from thresholds —
    # re-deriving it is how a rule and its audit drift apart.
    feats: dict[str, list[np.ndarray]] = {}
    masks: list[np.ndarray] = []
    sym: list[np.ndarray] = []
    for _ in range(args.steps):
        action, _skill, hold = select(obs, hold)
        rng, step_rng = jax.random.split(rng)
        raw, state, reward, done, info = env.step(
            jax.random.split(step_rng, args.num_envs), state, action, params)
        pobs = get_policy_obs(obs)
        for k, v in policy_module.diagnostics(pobs, pobs, action, reward, done, info).items():
            feats.setdefault(k, []).append(np.asarray(v))
        if hasattr(rule_module, "skill_mask"):
            masks.append(np.asarray(rule_module.skill_mask(pobs)))
        if hasattr(rule_module, "symbolic_meta_policy"):
            sym.append(np.asarray(rule_module.symbolic_meta_policy(pobs)))
        obs = normalize_obs(raw)
        if hold is not None:
            hold = (hold[0], hold[1], done.astype(bool))

    drivers = {k: np.concatenate(v, axis=0) for k, v in feats.items()}
    n = next(iter(drivers.values())).size
    skill_names = tuple(getattr(policy_module, "SKILL_NAMES", ()))
    print(f"\n{args.checkpoint}  ({meta_type}, {n} state samples)\n")
    print("  driver quantiles (on the successful policy's own visitation)")
    print("  signed:")
    for name, v in sorted(drivers.items()):
        qs = " ".join(f"q{int(q * 100):02d}={np.quantile(v, q):.4f}" for q in QUANTILES)
        print(f"    {name:32s} mean={v.mean():+.4f}  {qs}")
    # Absolute values, because that is the form the rules are written in: every cartpole
    # threshold and go1's roll/pitch terms compare |driver| against a bound. Reading a
    # threshold off a signed quantile would put it at the wrong place by construction.
    print("  absolute:")
    for name, v in sorted(drivers.items()):
        a = np.abs(v)
        qs = " ".join(f"q{int(q * 100):02d}={np.quantile(a, q):.4f}" for q in QUANTILES)
        print(f"    |{name:31s} mean={a.mean():.4f}  {qs}")

    if masks:
        adm = np.concatenate(masks, axis=0)  # [T*E, NUM_SKILLS]
        print("\n  shipped `skill_mask` admissibility, per skill (share of visited states)")
        for i in range(adm.shape[-1]):
            label = skill_names[i] if i < len(skill_names) else str(i)
            print(f"    {i} {label:20s} {100 * adm[:, i].mean():5.1f}%")
    if sym:
        choice = np.concatenate(sym, axis=0)
        print("\n  shipped `symbolic_meta_policy` firing distribution")
        for i in range(len(skill_names) or int(choice.max()) + 1):
            label = skill_names[i] if i < len(skill_names) else str(i)
            print(f"    {i} {label:20s} {100 * (choice == i).mean():5.1f}%")

    if "go1/height" not in drivers and not any(k.endswith("height") for k in drivers):
        # The block below is go1's shipped-rule arithmetic; other envs stop at the general
        # report above, which is what §5 stage 1 needs to pick quantiles.
        return 0
    hk = next(k for k in drivers if k.endswith("height"))
    height = drivers[hk]
    roll = np.abs(drivers[next(k for k in drivers if k.endswith("roll"))])
    pitch = np.abs(drivers[next(k for k in drivers if k.endswith("pitch"))])

    shipped = (height < 0.28) | (roll > 0.25) | (pitch > 0.25)
    print(f"\n  shipped rule (h<0.28 | |roll|>0.25 | |pitch|>0.25) admits recover on "
          f"{100 * shipped.mean():.1f}% of visited states")
    print(f"    height term alone: {100 * (height < 0.28).mean():.1f}%   "
          f"roll term: {100 * (roll > 0.25).mean():.1f}%   "
          f"pitch term: {100 * (pitch > 0.25).mean():.1f}%")

    print("\n  candidate revisions (target: recover admissible on the worst ~10% of states)")
    for q in (0.05, 0.10, 0.15, 0.20):
        h_t = float(np.quantile(height, q))
        r_t = float(np.quantile(roll, 1 - q))
        p_t = float(np.quantile(pitch, 1 - q))
        adm = (height < h_t) | (roll > r_t) | (pitch > p_t)
        print(f"    q={q:.2f}: h<{h_t:.4f} | |roll|>{r_t:.4f} | |pitch|>{p_t:.4f}"
              f"  -> admits {100 * adm.mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
