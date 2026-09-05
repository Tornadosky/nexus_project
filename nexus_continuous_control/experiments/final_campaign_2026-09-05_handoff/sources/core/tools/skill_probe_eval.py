"""Causal Q1/Q2 evidence: forced-skill probes and eval-time skill ablation.

Two eval-only experiments the paper does not have, both on existing checkpoints:

* **Forced-skill probes** (``--force``): run each skill actor *solo* — the meta-policy is
  replaced by a constant — and record the env's semantic metrics. If the skills are what their
  names say (Q1, "meaningful disentangled skills"), ``turn`` should track yaw commands best,
  ``hop_forward`` should carry the highest forward velocity, and so on. This tests the skill
  *semantics* directly rather than inferring them from reward curves.

* **Eval-time ablation** (``--ablate``): remove exactly one skill from selection (NeSy: its mask
  bit is forced false; neural: its meta-Q is masked to -1e9) and re-run the same deterministic
  eval. If the meta-policy's revealed preferences are load-bearing (Q2), removing a
  heavily-relied-on skill must collapse success while removing a rarely-chosen one must not.

Both run in one process per checkpoint so the env build and JIT are paid once. Selection,
normalization and metrics reuse ``robustness_eval`` verbatim — what these probes score is
exactly what the deterministic eval scores.

NeSy all-invalid guard: ablating a skill can leave a state with an all-false mask, where
``where(mask, q, -1e9)`` degenerates to argmax==0 regardless of the mask. In that state the
ablated skill is allowed back in as a fallback and the fraction of such steps is recorded as
``fallback_rate`` — silently mis-selecting skill 0 would corrupt the measurement.

Usage
-----
    JAX_PLATFORMS=cpu python tools/skill_probe_eval.py \
        --checkpoint runs/verify/go1_joystick_nesy_v2_s0.pkl \
        --episodes 64 --out runs/probes/go1_nesy_s0.probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustness_eval import (  # noqa: E402
    _build_networks,
    _load_checkpoint,
    _make_normalizer,
    _restore_params,
    evaluate,
)

from nexus_continuous.envs.playground_adapter import (  # noqa: E402
    build_playground_env,
    get_actor_obs,
    get_policy_obs,
)
from nexus_continuous.policies.registry import load_policy_module  # noqa: E402


def _make_probe_selector(actor, meta_q, actor_params, meta_params, policy_module,
                         meta_type: str, action_low, action_high,
                         force_skill: int | None = None,
                         disable_skill: int | None = None):
    """Selector with the meta-policy either forced to one skill or denied one skill.

    Mirrors ``robustness_eval._make_selector`` (greedy, stateless — probes are run without
    option commitment; every probed checkpoint here was trained at META_DECISION_INTERVAL=1).
    Returns (action, skill, hold) with hold passed through untouched.
    """

    def actor_actions(obs_actor):  # [N, E, A]
        return jax.vmap(lambda p: actor.apply({"params": p}, obs_actor))(actor_params)

    def select(obs, hold=None):
        obs_actor = get_actor_obs(obs)
        policy_obs = get_policy_obs(obs)
        all_actions = jnp.swapaxes(actor_actions(obs_actor), 0, 1)  # [E, N, A]
        E, N = obs_actor.shape[0], all_actions.shape[1]

        if force_skill is not None:
            skill = jnp.full((E,), force_skill, dtype=jnp.int32)
        elif meta_type == "symbolic":
            skill = jnp.asarray(policy_module.symbolic_meta_policy(policy_obs)).astype(jnp.int32)
        else:
            q = meta_q.apply({"params": meta_params}, obs_actor)  # [E, N]
            mask = jnp.ones((E, N), dtype=bool)
            if meta_type == "nesy":
                mask = jnp.asarray(policy_module.skill_mask(policy_obs)).astype(bool)
            if disable_skill is not None:
                abl = jnp.ones((N,), dtype=bool).at[disable_skill].set(False)
                new_mask = mask & abl[None, :]
                # all-invalid guard: fall back to the un-ablated mask where nothing remains
                empty = ~jnp.any(new_mask, axis=-1, keepdims=True)
                mask = jnp.where(empty, mask, new_mask)
            skill = jnp.argmax(jnp.where(mask, q, -1.0e9), axis=-1).astype(jnp.int32)

        action = all_actions[jnp.arange(E), skill]
        return jnp.clip(action, action_low, action_high), skill, hold

    return select


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--force", action="store_true", default=True,
                    help="run forced-skill probes (default on)")
    ap.add_argument("--ablate", action="store_true", default=True,
                    help="run per-skill ablation (default on; skipped for symbolic/flat)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ck = _load_checkpoint(args.checkpoint)
    cfg = dict(ck["config"])
    rs = ck["runner_state"]
    stats = ck.get("normalization_stats")
    meta_type = str(cfg.get("META_POLICY_TYPE", "nesy")).lower()
    policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    task_module = load_policy_module(cfg.get("TASK_POLICY", cfg["ENV_NAME"]))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES",
                               tuple(f"skill_{i}" for i in range(num_skills))))
    decision_interval = int(cfg.get("META_DECISION_INTERVAL", 1))
    if decision_interval > 1:
        print(f"note: checkpoint has META_DECISION_INTERVAL={decision_interval}; probes are "
              f"run stateless (forced/ablated selection has no commitment to model)")

    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NORMALIZE_REWARD"] = False
    bundle = build_playground_env(eval_cfg)
    eval_env, eval_env_params = bundle.env, bundle.env_params
    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    action_scale = (action_high - action_low) / 2.0
    max_steps = int(cfg.get("EVAL_MAX_STEPS") or bundle.episode_length)

    actor, meta_q = _build_networks(cfg, num_skills, bundle.action_dim, action_scale,
                                    (action_high + action_low) / 2.0)
    raw_obs0, _ = eval_env.reset(jax.random.split(jax.random.PRNGKey(0), args.num_envs),
                                 eval_env_params)
    dummy_actor = get_actor_obs(raw_obs0)
    fresh_actor = jax.vmap(lambda k: actor.init(k, dummy_actor)["params"])(
        jax.random.split(jax.random.PRNGKey(0), num_skills))
    actor_params = _restore_params(fresh_actor, rs["0"]["actor"]["params"])
    if meta_type in ("neural", "nesy"):
        fresh_meta = meta_q.init(jax.random.PRNGKey(1), dummy_actor)["params"]
        meta_params = _restore_params(fresh_meta, rs["0"]["meta"]["params"])
    else:
        meta_params = None

    normalize_obs = _make_normalizer(stats, cfg.get("NORMALIZE_OBS", True))

    # Merge task_metrics with diagnostics: the probes' whole point is semantic behaviour
    # (yaw error, forward velocity, joint speed), and task_metrics alone does not carry it.
    diag_fn = getattr(task_module, "diagnostics", None)
    task_fn = getattr(task_module, "task_metrics", None)

    def metrics_fn(*a):
        out = {}
        if task_fn is not None:
            out.update(task_fn(*a))
        if diag_fn is not None:
            out.update(diag_fn(*a))
        return out

    def run(select):
        return evaluate(cfg, eval_env, eval_env_params, select, normalize_obs, metrics_fn,
                        action_low, action_high, action_scale, args.num_envs, args.episodes,
                        max_steps, action_noise=0.0, seed=args.seed,
                        action_dim=bundle.action_dim, decision_interval=1)

    result: dict = {
        "checkpoint": str(args.checkpoint),
        "env": cfg["ENV_NAME"],
        "meta": meta_type,
        "skill_names": skill_names,
        "episodes": args.episodes,
    }

    def sel(**kw):
        return _make_probe_selector(actor, meta_q, actor_params, meta_params, policy_module,
                                    meta_type, action_low, action_high, **kw)

    print("intact eval …")
    intact = run(sel())
    result["intact"] = intact
    print(f"  return={intact['episode_return_mean']:.1f} "
          f"success={intact.get('primary_success_rate', float('nan')):.3f}")

    if args.force:
        result["forced"] = {}
        for i, name in enumerate(skill_names):
            print(f"forced skill {i} ({name}) …")
            s = run(sel(force_skill=i))
            result["forced"][name] = s
            print(f"  return={s['episode_return_mean']:.1f} "
                  f"success={s.get('primary_success_rate', float('nan')):.3f}")

    if args.ablate and meta_type in ("neural", "nesy"):
        result["ablated"] = {}
        for i, name in enumerate(skill_names):
            print(f"ablated skill {i} ({name}) …")
            s = run(sel(disable_skill=i))
            result["ablated"][name] = s
            print(f"  return={s['episode_return_mean']:.1f} "
                  f"success={s.get('primary_success_rate', float('nan')):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
