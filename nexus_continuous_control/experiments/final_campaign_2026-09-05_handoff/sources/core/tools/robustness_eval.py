"""Robustness / distribution-shift evaluation for trained NEXUS checkpoints.

Paper analogue: the NEXUS paper (Q4) tests trained agents on *modified* /
simplified environments and reports how much performance degrades. This tool
does the continuous-control equivalent **without retraining**: it loads a saved
checkpoint and re-runs the same deterministic evaluation as the trainer (same
greedy skill selection, same normalization, same task-success metrics), but
under a swept perturbation:

  * ``--mode action_noise`` (default): add Gaussian noise (scaled by the action
    range) to the deterministic action before stepping the env. A clean,
    env-agnostic actuation-robustness test.
  * ``--mode dynamics``: rebuild the Playground env with one or more config
    overrides (e.g. a gravity/mass/timestep change) and evaluate the unperturbed
    policy on it. The paper's "modified environment" angle. Override fields with
    ``--env-override KEY=VALUE`` (dotted keys allowed).

It reuses the exact success metrics the trainer uses (including the panda lift
and walker net-locomotion honest overrides), so numbers are comparable to the
``eval_metrics`` saved at training time.

Outputs a CSV (one row per perturbation level) and a degradation plot next to it.
Run once per checkpoint/variant; overlay the CSVs to compare variants.

Example
-------
    python -m nexus_continuous.scripts.train_nexus_playground ...  # produce a .pkl
    python tools/robustness_eval.py \
        --checkpoint runs/cartpole_balance_nesy_seed0.pkl \
        --mode action_noise --levels 0.0,0.05,0.1,0.2,0.3 \
        --episodes 128 --out runs/robustness/cartpole_nesy.csv

NOTE: written to match the in-trainer eval. Skill-holding (META_DECISION_INTERVAL>1) IS
modelled: the interval is read from the checkpoint's own config and the re-decision rule is
transcribed from the trainer, so a committed checkpoint is evaluated the way it was trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from nexus_continuous.envs.playground_adapter import (
    build_playground_env,
    get_actor_obs,
    get_critic_obs,
    get_policy_obs,
)
from nexus_continuous.networks import MetaQ, SkillActor
from nexus_continuous.policies.registry import load_policy_module


# --------------------------------------------------------------------------- #
# checkpoint + network reconstruction
# --------------------------------------------------------------------------- #
def _load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _build_networks(cfg: dict[str, Any], num_skills: int, action_dim: int,
                    action_scale: jnp.ndarray, action_bias: jnp.ndarray):
    actor = SkillActor(
        action_dim=action_dim,
        action_scale=action_scale,
        action_bias=action_bias,
        hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
        activation=cfg.get("ACTIVATION", "relu"),
        norm_type=cfg.get("NORM_TYPE", "layer_norm"),
        init_scale=float(cfg.get("ACTOR_INIT_SCALE", 0.01)),
    )
    meta_q = MetaQ(
        num_skills=num_skills,
        hidden_sizes=tuple(cfg.get("META_HIDDEN_SIZES", (256, 256))),
        activation=cfg.get("ACTIVATION", "relu"),
        norm_type=cfg.get("NORM_TYPE", "layer_norm"),
        init_scale=float(cfg.get("META_INIT_SCALE", 1.0)),
    )
    return actor, meta_q


def _restore_params(net_init_params, saved_params):
    """from_state_dict against a fresh init tree => exact structure match."""
    return serialization.from_state_dict(net_init_params, saved_params)


# --------------------------------------------------------------------------- #
# normalization + deterministic selection (mirrors the in-trainer eval)
# --------------------------------------------------------------------------- #
def _make_normalizer(stats: dict[str, Any] | None, normalize: bool):
    if not normalize or not stats or "actor_mean" not in stats:
        return lambda raw_obs: raw_obs
    am = jnp.asarray(stats["actor_mean"]); av = jnp.asarray(stats["actor_var"])
    cm = jnp.asarray(stats["critic_mean"]); cv = jnp.asarray(stats["critic_var"])

    def normalize_obs(raw_obs):
        raw_actor = raw_obs["raw_actor"] if isinstance(raw_obs, dict) else raw_obs
        raw_critic = raw_obs["raw_critic"] if isinstance(raw_obs, dict) else raw_obs
        obs = {
            "actor": (raw_actor - am) / jnp.sqrt(av + 1e-8),
            "critic": (raw_critic - cm) / jnp.sqrt(cv + 1e-8),
            "raw_actor": raw_actor,
            "raw_critic": raw_critic,
        }
        if isinstance(raw_obs, dict) and "policy_info" in raw_obs:
            obs["policy_info"] = raw_obs["policy_info"]
        return obs

    return normalize_obs


def _make_selector(actor, meta_q, actor_params, meta_params, policy_module,
                   meta_type: str, action_low, action_high, decision_interval: int = 1):
    """Greedy (no-exploration) action selector: (obs, hold) -> (action [E, A], skill, hold).

    `decision_interval` mirrors the trainer's `META_DECISION_INTERVAL`. It must, or a checkpoint
    trained with option commitment gets *evaluated* under per-step re-selection — the right
    policy driven by the wrong selection rule, which silently misrepresents it. The re-decision
    condition is transcribed from `hierarchical_ac_pqn_playground._select_action`: commit expires,
    OR the episode reset, OR (nesy only) the held skill left the mask.

    `hold` is `(held_skill, steps_held, prev_done)` or None for stateless selection.
    """

    def actor_actions(obs_actor):  # [N, E, A]
        return jax.vmap(lambda p: actor.apply({"params": p}, obs_actor))(actor_params)

    def select(obs, hold=None):
        obs_actor = get_actor_obs(obs)
        policy_obs = get_policy_obs(obs)
        all_actions = jnp.swapaxes(actor_actions(obs_actor), 0, 1)  # [E, N, A]
        E = obs_actor.shape[0]
        mask = jnp.ones((E, all_actions.shape[1]), dtype=bool)
        if meta_type == "symbolic":
            skill = jnp.asarray(policy_module.symbolic_meta_policy(policy_obs)).astype(jnp.int32)
        elif meta_type == "flat":
            skill = jnp.zeros((E,), dtype=jnp.int32)
        else:
            q = meta_q.apply({"params": meta_params}, obs_actor)  # [E, N]
            if meta_type == "nesy":
                mask = jnp.asarray(policy_module.skill_mask(policy_obs)).astype(bool)
                q = jnp.where(mask, q, -1.0e9)
            skill = jnp.argmax(q, axis=-1).astype(jnp.int32)

        new_hold = hold
        if decision_interval > 1 and hold is not None and meta_type != "flat":
            held_skill, steps_held, prev_done = hold
            held_valid = jnp.take_along_axis(mask, held_skill[:, None], axis=-1)[:, 0]
            redecide = (steps_held >= decision_interval) | prev_done | ~held_valid
            skill = jnp.where(redecide, skill, held_skill).astype(jnp.int32)
            steps_held = jnp.where(redecide, jnp.ones_like(steps_held), steps_held + 1)
            new_hold = (skill, steps_held, prev_done)

        action = all_actions[jnp.arange(E), skill]
        return jnp.clip(action, action_low, action_high), skill, new_hold

    return select


def init_hold_state(num_envs: int, decision_interval: int):
    """Initial `(held_skill, steps_held, prev_done)`, or None when commitment is off."""
    if decision_interval <= 1:
        return None
    return (
        jnp.zeros((num_envs,), dtype=jnp.int32),
        jnp.full((num_envs,), decision_interval, dtype=jnp.int32),  # forces a decision on step 0
        jnp.zeros((num_envs,), dtype=bool),
    )


# --------------------------------------------------------------------------- #
# honest-metric episode overrides (ported verbatim from the trainer eval)
# --------------------------------------------------------------------------- #
def _panda_overrides(mean_metrics, max_metrics, initial_metrics):
    if "panda/cube_height_max_mean" not in mean_metrics:
        return mean_metrics
    out = dict(mean_metrics)
    max_h = max_metrics["panda/cube_height_max_mean"]
    init_h = initial_metrics["panda/cube_height_max_mean"]
    init_h = jnp.where(init_h > 0.01, init_h, 0.03)
    lift = (max_h > init_h + 0.05) | (max_h > 0.08)
    out["panda/lift_success_rate"] = lift.astype(jnp.float32)
    out["primary_success_rate"] = lift.astype(jnp.float32)
    out["primary_goal_metric"] = max_h - init_h
    return out


def _walker_overrides(mean_metrics):
    if "walker/forward_velocity_mean" not in mean_metrics:
        return mean_metrics
    out = dict(mean_metrics)
    net = (out["walker/forward_velocity_mean"] > 0.5) & (out["walker/stand_success_rate"] > 0.5)
    out["walker/net_walk_success_rate"] = net.astype(jnp.float32)
    out["primary_success_rate"] = net.astype(jnp.float32)
    return out


# --------------------------------------------------------------------------- #
# evaluation under a perturbation
# --------------------------------------------------------------------------- #
def evaluate(cfg, eval_env, eval_env_params, select, normalize_obs, task_metrics_fn,
             action_low, action_high, action_scale, num_envs, num_episodes, max_steps,
             action_noise: float, seed: int, action_dim: int | None = None,
             decision_interval: int = 1, num_skills: int = 0,
             skill_names: Sequence[str] | None = None):
    num_batches = max(1, int(np.ceil(num_episodes / num_envs)))
    # `bundle.action_low` comes from the adapter's _Box, whose `low` is a SCALAR bound, not a
    # per-dimension array — so `action_low.shape[-1]` raises IndexError. Take the width from
    # the bundle's action_dim and fall back to the array only if it really is per-dimension.
    if action_dim is None:
        action_dim = int(jnp.asarray(action_low).shape[-1]) if jnp.asarray(action_low).ndim else 1
    act_dtype = jnp.asarray(action_low).dtype

    def eval_batch(batch_rng):
        reset_rng = jax.random.split(batch_rng, num_envs)
        raw_obs, state = eval_env.reset(reset_rng, eval_env_params)
        obs = normalize_obs(raw_obs)
        zero_a = jnp.zeros((num_envs, action_dim), dtype=act_dtype)
        zero_r = jnp.zeros((num_envs,), dtype=act_dtype)
        zero_d = jnp.zeros((num_envs,), dtype=bool)
        init_metrics = task_metrics_fn(get_policy_obs(obs), get_policy_obs(obs),
                                       zero_a, zero_r, zero_d, None)
        zeros_m = jax.tree_util.tree_map(jnp.zeros_like, init_metrics)
        neginf_m = jax.tree_util.tree_map(lambda x: jnp.full_like(x, -jnp.inf), init_metrics)

        def step(carry, _):
            state, obs, rng, done_seen, returns, lengths, sum_m, max_m, hold, skill_ct = carry
            action, _skill, hold = select(obs, hold)
            rng, rng_noise, rng_step = jax.random.split(rng, 3)
            if action_noise > 0:
                noise = jax.random.normal(rng_noise, action.shape) * action_noise * action_scale
                action = jnp.clip(action + noise, action_low, action_high)
            step_rng = jax.random.split(rng_step, num_envs)
            raw_next, state, reward, done, info = eval_env.step(step_rng, state, action, eval_env_params)
            next_obs = normalize_obs(raw_next)
            metrics = task_metrics_fn(get_policy_obs(obs), get_policy_obs(next_obs),
                                      action, reward, done, info)
            active = (~done_seen).astype(jnp.float32)
            active_b = active.astype(bool)
            returns = returns + reward * active
            lengths = lengths + active
            sum_m = jax.tree_util.tree_map(lambda a, v: a + v * active, sum_m, metrics)
            max_m = jax.tree_util.tree_map(
                lambda a, v: jnp.maximum(a, jnp.where(active_b, v, -jnp.inf)), max_m, metrics)
            # Which skill the meta-policy actually ran, counted only on steps that are still
            # inside the episode (`active`), so a finished env padding out the scan cannot
            # inflate whichever skill it happened to hold. This is the behavioural half of the
            # evaluation: success says how well the hierarchy did, this says what it did.
            if num_skills:
                skill_ct = skill_ct + jnp.sum(
                    jax.nn.one_hot(_skill, num_skills) * active[:, None], axis=0
                )
            done_seen = done_seen | done
            if hold is not None:
                # The trainer refreshes prev_done from the step's `done` so a reset forces a
                # re-decision on the next step; mirror that or a held skill survives an episode
                # boundary.
                hold = (hold[0], hold[1], done.astype(bool))
            return (state, next_obs, rng, done_seen, returns, lengths, sum_m, max_m, hold,
                    skill_ct), None

        init = (state, obs, batch_rng, zero_d,
                jnp.zeros((num_envs,), action_low.dtype), jnp.zeros((num_envs,), action_low.dtype),
                zeros_m, neginf_m, init_hold_state(num_envs, decision_interval),
                jnp.zeros((max(num_skills, 1),), jnp.float32))
        final, _ = jax.lax.scan(step, init, None, max_steps)
        _, _, _, _, returns, lengths, sum_m, max_m, _, skill_ct = final
        denom = jnp.maximum(lengths, 1.0)
        mean_m = jax.tree_util.tree_map(lambda v: v / denom, sum_m)
        mean_m = _panda_overrides(mean_m, max_m, init_metrics)
        mean_m = _walker_overrides(mean_m)
        table = {"episode_return": returns, "episode_length": lengths, **mean_m}
        if num_skills:
            # Broadcast the batch-level share onto every episode row so the outer
            # `jnp.mean` over episodes reproduces it rather than mangling the shape.
            share = skill_ct / jnp.maximum(jnp.sum(skill_ct), 1.0)
            for i in range(num_skills):
                name = skill_names[i] if skill_names else str(i)
                table[f"skill_usage/{i}_{name}"] = jnp.full((num_envs,), share[i])
        return table

    rngs = jax.random.split(jax.random.PRNGKey(seed), num_batches)
    _, batched = jax.lax.scan(lambda _, r: (None, eval_batch(r)), None, rngs)
    flat = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:])[:num_episodes], batched)
    summary = {
        "episode_return_mean": float(jnp.mean(flat["episode_return"])),
        "episode_return_std": float(jnp.std(flat["episode_return"])),
        "episode_length_mean": float(jnp.mean(flat["episode_length"])),
    }
    for k, v in flat.items():
        if k in ("episode_return", "episode_length"):
            continue
        summary[k] = float(jnp.mean(v))
    return summary


def _apply_env_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply KEY=VALUE overrides into a copy of cfg under ENV_CONFIG_OVERRIDES.

    build_playground_env is expected to read ENV_CONFIG_OVERRIDES if present; if it
    does not, dynamics mode is a no-op and a warning is printed (action_noise mode
    still works regardless).
    """
    out = dict(cfg)
    od = dict(out.get("ENV_CONFIG_OVERRIDES", {}))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--env-override expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        # JSON first so vector-valued env keys work: Go1's `command_config.a` is a
        # 3-vector (max |lin_vel_x|, |lin_vel_y|, |ang_vel_yaw|), and the OOD-command
        # conditions scale it. float()-only parsing silently turned such a value into
        # the *string* "[2.25,1.2,1.8]", which setattr accepts without complaint — the
        # eval would then have run at default command ranges and reported the result as
        # if it were the shifted condition.
        try:
            val_parsed: Any = json.loads(val)
        except json.JSONDecodeError:
            try:
                val_parsed = float(val)
            except ValueError:
                val_parsed = val
        od[key.strip()] = val_parsed
    out["ENV_CONFIG_OVERRIDES"] = od
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to a trained .pkl checkpoint.")
    p.add_argument("--mode", choices=["action_noise", "dynamics"], default="action_noise")
    p.add_argument("--levels", default="0.0,0.05,0.1,0.2,0.3",
                   help="Comma list of action-noise stds (action_noise mode).")
    p.add_argument("--env-override", action="append", default=[],
                   help="dynamics mode: KEY=VALUE Playground-config override (repeatable).")
    p.add_argument("--env-name", default=None,
                   help="Evaluate the checkpoint on a DIFFERENT Playground env (zero-shot "
                        "transfer, e.g. a flat-terrain Go1 policy on Go1JoystickRoughTerrain). "
                        "The policy module, networks and normalization stats stay the "
                        "checkpoint's own; if the target env's observation layout differs, "
                        "network restore or the semantic features will fail loudly — that is "
                        "the desired failure mode, not something to silence.")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--out", default=None, help="CSV output path (default next to checkpoint).")
    args = p.parse_args(argv)

    ck = _load_checkpoint(args.checkpoint)
    cfg = dict(ck["config"])
    rs = ck["runner_state"]
    stats = ck.get("normalization_stats")
    meta_type = str(cfg.get("META_POLICY_TYPE", "nesy")).lower()
    policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    task_module = load_policy_module(cfg.get("TASK_POLICY", cfg["ENV_NAME"]))
    num_skills = int(policy_module.NUM_SKILLS)

    # eval env: unnormalized (we apply frozen training stats ourselves), like the trainer.
    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NORMALIZE_REWARD"] = False
    if args.env_name:
        print(f"zero-shot transfer: checkpoint env {cfg['ENV_NAME']} -> eval env {args.env_name}")
        eval_cfg["ENV_NAME"] = args.env_name
    # Apply overrides in EVERY mode. Gating this on `--mode dynamics` meant an
    # `--env-override` passed alongside `--mode action_noise` was silently discarded: the
    # command-shift OOS runs produced success numbers identical to in-distribution, to three
    # decimals, and would have been reported as "the arms are insensitive to OOD commands".
    # A flag the user supplied must never be dropped without saying so.
    if args.env_override:
        eval_cfg = _apply_env_overrides(eval_cfg, args.env_override)
        print(f"env overrides applied: {eval_cfg['ENV_CONFIG_OVERRIDES']}")
    bundle = build_playground_env(eval_cfg)
    eval_env, eval_env_params = bundle.env, bundle.env_params
    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    action_scale = (action_high - action_low) / 2.0
    action_dim = bundle.action_dim
    max_steps = int(cfg.get("EVAL_MAX_STEPS") or bundle.episode_length)

    actor, meta_q = _build_networks(cfg, num_skills, action_dim, action_scale,
                                    (action_high + action_low) / 2.0)

    # restore params via from_state_dict against a fresh init of the same structure
    raw_obs0, _ = eval_env.reset(jax.random.split(jax.random.PRNGKey(0), args.num_envs),
                                 eval_env_params)
    dummy_actor = get_actor_obs(raw_obs0)
    ar = jax.random.split(jax.random.PRNGKey(0), num_skills)
    fresh_actor = jax.vmap(lambda k: actor.init(k, dummy_actor)["params"])(ar)
    actor_params = _restore_params(fresh_actor, rs["0"]["actor"]["params"])
    if meta_type in ("neural", "nesy"):
        fresh_meta = meta_q.init(jax.random.PRNGKey(1), dummy_actor)["params"]
        meta_params = _restore_params(fresh_meta, rs["0"]["meta"]["params"])
    else:
        meta_params = None

    normalize_obs = _make_normalizer(stats, cfg.get("NORMALIZE_OBS", True))
    # Read commitment from the CHECKPOINT's own config, not from a flag: a checkpoint must be
    # evaluated under the selection rule it was trained with, and nobody should have to remember
    # to pass it.
    decision_interval = int(cfg.get("META_DECISION_INTERVAL", 1))
    if decision_interval > 1:
        print(f"note: checkpoint trained with META_DECISION_INTERVAL={decision_interval}; "
              f"evaluating with option commitment")
    select = _make_selector(actor, meta_q, actor_params, meta_params, policy_module,
                            meta_type, action_low, action_high, decision_interval)
    task_metrics_fn = (lambda *a: task_module.task_metrics(*a)) if hasattr(task_module, "task_metrics") else (lambda *a: {})

    if args.mode == "action_noise":
        levels = [float(x) for x in args.levels.split(",") if x.strip() != ""]
    else:
        levels = [0.0]  # dynamics shift baked into env; single eval

    rows = []
    for lvl in levels:
        summary = evaluate(
            cfg, eval_env, eval_env_params, select, normalize_obs, task_metrics_fn,
            action_low, action_high, action_scale, args.num_envs, args.episodes, max_steps,
            action_noise=lvl, seed=args.seed, action_dim=action_dim,
            decision_interval=decision_interval, num_skills=num_skills,
            skill_names=getattr(policy_module, "SKILL_NAMES", None),
        )
        summary["perturbation"] = lvl
        summary["mode"] = args.mode
        summary["env"] = eval_cfg["ENV_NAME"]  # the env actually evaluated on (zero-shot aware)
        summary["train_env"] = cfg["ENV_NAME"]
        summary["meta"] = meta_type
        rows.append(summary)
        succ = summary.get("primary_success_rate", float("nan"))
        print(f"[{args.mode} {lvl:>5}] return={summary['episode_return_mean']:.2f} "
              f"success={succ:.3f}")

    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".robustness.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_path)

    # degradation plot (action_noise mode only)
    if args.mode == "action_noise" and len(rows) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [r["perturbation"] for r in rows]
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(xs, [r["episode_return_mean"] for r in rows], "o-", color="tab:blue", label="return")
            ax1.set_xlabel("action-noise std"); ax1.set_ylabel("episode return", color="tab:blue")
            if "primary_success_rate" in rows[0]:
                ax2 = ax1.twinx()
                ax2.plot(xs, [r.get("primary_success_rate", float("nan")) for r in rows],
                         "s--", color="tab:red", label="success")
                ax2.set_ylabel("primary success", color="tab:red"); ax2.set_ylim(0, 1)
            plt.title(f"Robustness: {cfg['ENV_NAME']} ({meta_type})")
            plt.tight_layout()
            png = out_path.with_suffix(".png")
            plt.savefig(png, dpi=130)
            print("wrote", png)
        except Exception as exc:  # pragma: no cover
            print("plot skipped:", exc)


if __name__ == "__main__":
    main()
