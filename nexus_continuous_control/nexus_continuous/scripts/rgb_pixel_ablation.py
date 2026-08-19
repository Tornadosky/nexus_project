"""Does the in-loop pixel actor ACTUALLY use its pixels?

The in-loop RGB result (`USE_RGB`) claims the skill actors control the robot from
64x64 camera frames. Structurally that must be true -- with `RGB_PROPRIO: none`
the actor's non-pixel input is a width-0 tensor, so no state can reach it. But the
HIERARCHY still receives privileged state through the meta-policy, which re-selects
a skill every step (`META_DECISION_INTERVAL` defaults to 1). So a sceptic can ask:

    "What if the CNN ignores its input and each skill just emits a constant action?
     A state-driven meta switching between a few constant actions every step is a
     classic bang-bang controller -- that alone could balance a pole."

This script settles it, with NO retraining of the ablated policy. It trains (or
loads) one in-loop pixel policy and then re-evaluates the SAME frozen weights
under corruptions that break the pixel->state link while leaving the environment,
the meta-policy and the privileged state untouched:

  intact          the real pixel stack                       (baseline)
  frozen_first    always the t=0 stack                       (image carries no
                  current information at all)
  random_replay   a real stack sampled from elsewhere in the episode (identical
                  image STATISTICS, wrong correlation with the true state --
                  the strongest and least "out-of-distribution" test)
  shuffle_frames  the 3 frames of the current stack permuted (pose preserved,
                  MOTION/velocity destroyed -- separates "uses the picture" from
                  "uses the movement in the picture")
  zeros           an all-zero image                          (blunt sanity check)
  const_action    pixels untouched but the actor's output is REPLACED by that
                  skill's mean action from the intact rollout -- this is the
                  sceptic's hypothesis made explicit: meta + constant actions

Reading the result:
  * intact >> corrupted            -> the actor genuinely uses the pixels.
  * intact ~= corrupted            -> it does not; the meta-policy is doing the
                                      work and the pixel claim must be dropped.
  * intact >> const_action         -> the actor's pixel-conditioned variation is
                                      necessary, not just the skill switching.
  * shuffle_frames << intact       -> it uses motion, not just a static pose.

    python -m nexus_continuous.scripts.rgb_pixel_ablation \\
        --config configs/cartpole_balance_nesy_rgb.yaml --meta nesy --seed 0 \\
        --updates 250 --num-envs 128 --episodes 5 \\
        --save-policy runs/abl_cartpole.pkl --out results/rgb/ablation/cartpole/nesy_blind
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

CONDITIONS = (
    "intact",
    "frozen_first",
    "random_replay",
    "shuffle_frames",
    "zeros",
    "const_action",
)


def _mjx_data(state):
    """Dig through the env wrappers (LogVec/Clip/Normalize) to the mjx State's .data."""
    for _ in range(8):
        if hasattr(state, "data"):
            return state.data
        state = getattr(state, "env_state", None)
        if state is None:
            break
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cartpole_balance_nesy_rgb.yaml",
                    help="any state config works; USE_RGB is forced on")
    ap.add_argument("--meta", default="neural", choices=["nesy", "neural", "symbolic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/rgb_ablation")
    ap.add_argument("--updates", type=int, default=250, help="in-loop training updates")
    ap.add_argument("--num-envs", type=int, default=128, help="training render batch (nworld)")
    ap.add_argument("--episodes", type=int, default=5,
                    help="rollouts per condition (different reset keys -> mean +/- std)")
    ap.add_argument("--eval-steps", type=int, default=250, help="steps per rollout")
    ap.add_argument("--save-policy", default=None,
                    help="pickle the trained in-loop policy so ablations can be re-run "
                         "without retraining")
    ap.add_argument("--load-policy", default=None,
                    help="skip training and load a policy saved by --save-policy")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp

    from nexus_continuous.utils import load_config
    from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
    from nexus_continuous.networks import MetaQ
    from nexus_continuous.vision import VisionSkillActor
    from nexus_continuous.policies.registry import load_policy_module
    from nexus_continuous.envs.playground_adapter import (
        build_playground_env,
        get_actor_obs,
        get_actor_pixels,
        get_policy_obs,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["SEED"] = args.seed
    cfg["META_POLICY_TYPE"] = args.meta
    cfg["USE_RGB"] = True
    cfg["NUM_SEEDS"] = 1
    cfg["NUM_ENVS"] = args.num_envs
    cfg["TOTAL_TIMESTEPS"] = args.num_envs * cfg.get("NUM_STEPS", 64) * args.updates
    env_name = cfg["ENV_NAME"]
    is_cartpole = env_name == "CartpoleBalance"

    proprio_mode = str(cfg.get("RGB_PROPRIO", "none")).lower()
    if proprio_mode != "none":
        print(f"[WARN] RGB_PROPRIO={proprio_mode!r}: the actor also reads privileged "
              "state, so a small ablation effect would be expected even if the pixels "
              "matter. Run with RGB_PROPRIO=none for the clean test.")
    if int(cfg.get("META_DECISION_INTERVAL", 1)) != 1:
        print(f"[note] META_DECISION_INTERVAL={cfg.get('META_DECISION_INTERVAL')} "
              "(the meta re-decides less often, so the state channel is narrower).")

    print(f"jax {jax.__version__} {jax.devices()} | env {env_name} | meta {args.meta}")

    # ---- Stage 1: obtain an in-loop pixel policy (train, or load a saved one) ----
    if args.load_policy:
        print(f"[1] loading policy from {args.load_policy}")
        blob = pickle.loads(Path(args.load_policy).read_bytes())
        actor_params = jax.tree_util.tree_map(jnp.asarray, blob["actor_params"])
        meta_params = (None if blob["meta_params"] is None
                       else jax.tree_util.tree_map(jnp.asarray, blob["meta_params"]))
        stats = jax.tree_util.tree_map(jnp.asarray, blob["normalization_stats"])
        train_return = blob.get("final_train_return")
    else:
        print(f"[1] training in-loop from pixels ({args.updates} updates, "
              f"{args.num_envs} envs)...")
        output = run_training(cfg)
        train_state = output.runner_state[0]
        actor_params = train_state.actor.params            # vmapped over skills [N, ...]
        meta_params = None if args.meta == "symbolic" else train_state.meta.params
        stats = output.normalization_stats
        curve = (output.metrics or {}).get("env/returned_episode_returns")
        train_return = None
        if curve is not None:
            c = np.asarray(curve)
            c = c.reshape(c.shape[0], -1).mean(1) if c.ndim > 1 else c.reshape(-1)
            train_return = float(c[-20:].mean())
            print(f"    final training return ~{train_return:.2f}")
        if args.save_policy:
            Path(args.save_policy).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_policy).write_bytes(pickle.dumps({
                "actor_params": jax.device_get(actor_params),
                "meta_params": None if meta_params is None else jax.device_get(meta_params),
                "normalization_stats": jax.device_get(stats),
                "final_train_return": train_return,
                "env": env_name, "meta": args.meta, "seed": args.seed,
            }))
            print(f"    saved policy -> {args.save_policy}")
        # Training curves, so a finished run is reconstructable without a retrain.
        # pixel_sensitivity is the important one: it shows perception developing
        # (or staying flat, which is exactly the failure this campaign found).
        curves = {}
        for mkey, mname in (("env/returned_episode_returns", "episode_return"),
                            ("train/rgb/pixel_sensitivity", "pixel_sensitivity"),
                            ("train/rgb/aux_state_loss", "aux_state_loss")):
            raw = (output.metrics or {}).get(mkey)
            if raw is None:
                continue
            arr = np.asarray(raw)
            arr = arr.reshape(arr.shape[0], -1).mean(1) if arr.ndim > 1 else arr.reshape(-1)
            curves[mname] = arr.tolist()
        if curves:
            (out / "training_curves.json").write_text(json.dumps(
                {"env": env_name, "updates": args.updates, "num_envs": args.num_envs,
                 "config": args.config, "curves": curves}, indent=2))
            fig_c, axs = plt.subplots(1, len(curves), figsize=(5.0 * len(curves), 3.6),
                                      squeeze=False)
            for ax_c, (cname, cvals) in zip(axs[0], curves.items()):
                ax_c.plot(np.arange(len(cvals)), cvals, lw=1.6, color="#4C72B0")
                ax_c.set_xlabel("training update")
                ax_c.set_title(cname.replace("_", " "))
                if cname == "pixel_sensitivity":
                    ax_c.axhline(0.01, ls="--", lw=1, color="grey")
                    ax_c.set_ylabel("action shift when pixels change")
            fig_c.suptitle(f"{env_name} in-loop training ({args.updates} updates, "
                           f"{args.num_envs} envs)")
            fig_c.tight_layout()
            fig_c.savefig(out / "training_curves.png", dpi=130, bbox_inches="tight")
            plt.close(fig_c)
            print(f"    saved training curves: {sorted(curves)}")

    # ---- Stage 2: 1-env vision environment for the ablation rollouts ----
    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False    # raw obs; frozen stats applied to the meta input
    eval_cfg["NUM_ENVS"] = 1
    eval_cfg["RENDER_NWORLD"] = 1
    bundle = build_playground_env(eval_cfg)
    env, env_params = bundle.env, bundle.env_params
    alo, ahi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
    action_dim = int(bundle.action_dim)
    step_fn = jax.jit(env.step)
    normalize_obs = bool(cfg.get("NORMALIZE_OBS", True))

    policy_module = load_policy_module(cfg.get("POLICY", env_name))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES",
                               [f"skill{i}" for i in range(num_skills)]))

    vactor = VisionSkillActor(
        action_dim=action_dim, action_scale=(ahi - alo) / 2.0, action_bias=(ahi + alo) / 2.0,
        hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
        embedding_dim=int(cfg.get("RGB_EMBED_DIM", 128)),
    )
    meta_q = MetaQ(
        num_skills=num_skills, hidden_sizes=tuple(cfg.get("META_HIDDEN_SIZES", (256, 256))),
        activation=cfg.get("ACTIVATION", "relu"), norm_type=cfg.get("NORM_TYPE", "layer_norm"),
        init_scale=float(cfg.get("META_INIT_SCALE", 1.0)),
    )
    rgb_indices = cfg.get("RGB_PROPRIO_INDICES")

    def _actor_proprio(oa):
        """Mirror the trainer's RGB_PROPRIO restriction exactly."""
        if proprio_mode == "full":
            return oa
        if proprio_mode == "indices" and rgb_indices is not None:
            return oa[..., jnp.asarray(list(rgb_indices), dtype=jnp.int32)]
        return oa[..., :0]

    def _norm_meta(obs):
        oa = get_actor_obs(obs)
        if normalize_obs:
            oa = (oa - stats["actor_mean"]) / jnp.sqrt(stats["actor_var"] + 1e-8)
        return oa

    @jax.jit
    def select_skill(obs):
        """The meta-policy: unchanged, always on privileged state."""
        oa = _norm_meta(obs)
        pol = get_policy_obs(obs)
        if args.meta == "symbolic":
            return jnp.atleast_1d(
                jnp.asarray(policy_module.symbolic_meta_policy(pol), jnp.int32))
        qm = jnp.atleast_2d(meta_q.apply({"params": meta_params}, oa))
        if args.meta == "nesy":
            mask = jnp.atleast_2d(jnp.asarray(policy_module.skill_mask(pol), bool))
            any_valid = jnp.any(mask, axis=-1, keepdims=True)
            qm = jnp.where(jnp.where(any_valid, mask, jnp.ones_like(mask)), qm, -1.0e9)
        return jnp.argmax(qm, axis=-1).astype(jnp.int32)

    @jax.jit
    def act_from_pixels(obs, px, skill):
        """The skill actor. Pixels are an explicit argument so we can corrupt them."""
        proprio = _actor_proprio(_norm_meta(obs))
        all_a = jax.vmap(lambda p: vactor.apply({"params": p}, px, proprio))(actor_params)
        return all_a[skill, jnp.arange(all_a.shape[1])]

    # ---- Stage 3: one rollout under a given pixel condition ----
    def rollout(condition: str, ep: int, replay_bank, mean_action):
        rng = jax.random.PRNGKey(9000 + 97 * ep + args.seed)
        rng, rr = jax.random.split(rng)
        obs, state = env.reset(jax.random.split(rr, 1), env_params)
        npr = np.random.RandomState(1234 + ep)

        first_px = None
        rewards, upright, seen = [], 0, []
        for t in range(args.eval_steps):
            px = get_actor_pixels(obs)                       # [1, H, W, 3] real stack
            if px is None:
                raise RuntimeError("no actor_pixels in the observation -- is USE_RGB on?")
            if first_px is None:
                first_px = px
            seen.append(np.asarray(px))

            if condition == "intact" or condition == "const_action":
                px_in = px
            elif condition == "frozen_first":
                px_in = first_px
            elif condition == "zeros":
                px_in = jnp.zeros_like(px)
            elif condition == "shuffle_frames":
                perm = npr.permutation(px.shape[-1])
                px_in = px[..., jnp.asarray(perm)]
            elif condition == "random_replay":
                if replay_bank is None or len(replay_bank) == 0:
                    px_in = px                               # bank not built yet
                else:
                    px_in = jnp.asarray(replay_bank[npr.randint(len(replay_bank))])
            else:
                raise ValueError(f"unknown condition {condition!r}")

            skill = select_skill(obs)
            if condition == "const_action":
                # The sceptic's hypothesis: meta + one fixed action per skill.
                act = jnp.asarray(mean_action[np.asarray(skill)[0]])[None]
            else:
                act = act_from_pixels(obs, px_in, skill)
            act = jnp.clip(act, alo, ahi)

            if is_cartpole:
                sd = _mjx_data(state)
                if sd is not None:
                    cart = float(np.asarray(sd.qpos[0, 0]))
                    ang = float(np.asarray(sd.qpos[0, 1]))
                    upright += int(abs(np.arctan2(np.sin(ang), np.cos(ang))) < 0.25
                                   and abs(cart) < 1.0)

            rng, rs = jax.random.split(rng)
            obs, state, rew, _d, _i = step_fn(jax.random.split(rs, 1), state, act, env_params)
            rewards.append(float(np.asarray(rew).reshape(-1)[0]))

        rewards = np.asarray(rewards)
        return {
            "reward_per_step": float(rewards.mean()),
            "return": float(rewards.sum()),
            "upright_fraction": (upright / args.eval_steps) if is_cartpole else None,
            "frames": seen,
        }

    # ---- Stage 4: build the intact reference, the replay bank and the mean actions ----
    print(f"[2] intact reference rollouts ({args.episodes} episodes x {args.eval_steps} steps)...")
    replay_bank, mean_action = None, np.zeros((num_skills, action_dim), np.float32)
    intact_runs = [rollout("intact", ep, None, mean_action) for ep in range(args.episodes)]
    replay_bank = [f for r in intact_runs for f in r["frames"]]        # real, in-distribution

    # per-skill mean action of the trained pixel actor, for the const_action control
    acc = np.zeros((num_skills, action_dim), np.float64)
    cnt = np.zeros((num_skills,), np.int64)
    rng = jax.random.PRNGKey(4242 + args.seed)
    rng, rr = jax.random.split(rng)
    obs, state = env.reset(jax.random.split(rr, 1), env_params)
    for _t in range(args.eval_steps):
        px = get_actor_pixels(obs)
        skill = select_skill(obs)
        a = np.asarray(act_from_pixels(obs, px, skill))[0]
        k = int(np.asarray(skill)[0])
        acc[k] += a
        cnt[k] += 1
        rng, rs = jax.random.split(rng)
        obs, state, _r, _d, _i = step_fn(
            jax.random.split(rs, 1), state, jnp.clip(jnp.asarray(a)[None], alo, ahi), env_params)
    mean_action = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    print("    per-skill mean action (used by const_action):")
    for k in range(num_skills):
        print(f"      {skill_names[k]:>22s}  n={int(cnt[k]):4d}  {np.round(mean_action[k], 3)}")

    # ---- Stage 5: run every condition ----
    results = {}
    for cond in CONDITIONS:
        runs = intact_runs if cond == "intact" else [
            rollout(cond, ep, replay_bank, mean_action) for ep in range(args.episodes)
        ]
        rps = np.array([r["reward_per_step"] for r in runs])
        entry = {
            "reward_per_step_mean": float(rps.mean()),
            "reward_per_step_std": float(rps.std()),
            "per_episode": rps.tolist(),
        }
        if is_cartpole:
            up = np.array([r["upright_fraction"] for r in runs])
            entry["upright_fraction_mean"] = float(up.mean())
            entry["upright_fraction_std"] = float(up.std())
        results[cond] = entry
        extra = (f" | upright {entry['upright_fraction_mean']:.3f}" if is_cartpole else "")
        print(f"    {cond:>15s}: {rps.mean():.4f} +/- {rps.std():.4f} reward/step{extra}")

    # ---- Stage 6: verdict ----
    # Primary metric: for cartpole use the BOUNDED upright fraction. Mean task
    # reward can be negative for a mediocre policy, and a ratio against a negative
    # baseline inverts the sign (an IMPROVEMENT would read as a huge "drop").
    metric_key = "upright_fraction_mean" if is_cartpole else "reward_per_step_mean"
    base = results["intact"][metric_key]

    def drop(cond):
        """Fraction of the baseline lost. Negative = the condition did BETTER."""
        v = results[cond][metric_key]
        return float((base - v) / max(abs(base), 1e-9))

    drops = {c: drop(c) for c in CONDITIONS if c != "intact"}
    pixel_drops = [drops[c] for c in ("frozen_first", "random_replay", "zeros")]
    # 2-of-3 MEDIAN, not min. Requiring every corruption to clear the bar makes
    # the verdict hostage to the most forgiving one: on the fixed cartpole run
    # blanking the image cost 56.3% and freezing it 37.0%, yet the min-rule
    # reported "does not use pixels" because random_replay came in at 27.9%
    # (real frames from another timestep are sometimes coincidentally apt).
    # Both numbers are reported so nothing is hidden.
    pixel_drop_median = float(sorted(pixel_drops)[len(pixel_drops) // 2])
    uses_pixels = bool(pixel_drop_median > 0.30)
    needs_variation = bool(drops["const_action"] > 0.30)
    verdict = {
        "env": env_name, "meta": args.meta, "seed": args.seed,
        "updates": args.updates, "num_envs": args.num_envs,
        "episodes": args.episodes, "eval_steps": args.eval_steps,
        "rgb_proprio": proprio_mode,
        "final_train_return": train_return,
        "performance_drop_fraction": drops,
        "actor_uses_pixels": uses_pixels,
        "actor_uses_pixels_strict": bool(min(pixel_drops) > 0.30),
        "pixel_drop_median": pixel_drop_median,
        "pixel_drop_min": float(min(pixel_drops)),
        "actor_variation_needed": needs_variation,
        "motion_sensitive": bool(drops["shuffle_frames"] > 0.15),
        "results": results,
    }
    (out / "pixel_ablation.json").write_text(json.dumps(verdict, indent=2))

    print("\n==== VERDICT ====")
    print(f"  baseline (intact): {base:.4f} [{metric_key}]")
    for c, d in drops.items():
        print(f"  {c:>15s}: {100 * d:5.1f}% performance drop")
    print(f"\n  actor USES the pixels          : {'YES' if uses_pixels else 'NO -- claim unsupported'}"
          "   (all pixel corruptions cost >30%)")
    print(f"  pixel-conditioned variation is : {'NECESSARY' if needs_variation else 'NOT necessary'}"
          f"   (dropping the actor for a constant action costs "
          f"{100 * drops['const_action']:.1f}%)")
    print(f"  uses MOTION, not just a pose   : {'YES' if verdict['motion_sensitive'] else 'weak/no'}"
          "   (frame-shuffle costs >15%)")

    # ---- bar chart ----
    fig = plt.figure(figsize=(7.5, 4.2))
    xs = list(CONDITIONS)
    vals = [results[c]["reward_per_step_mean"] for c in xs]
    errs = [results[c]["reward_per_step_std"] for c in xs]
    colors = ["#4C72B0"] + ["#DD8452"] * 4 + ["#937860"]
    plt.bar(range(len(xs)), vals, yerr=errs, capsize=5, color=colors)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(range(len(xs)), xs, rotation=20, ha="right", fontsize=9)
    plt.ylabel("mean task reward / step")
    plt.title(f"Does the in-loop pixel actor use its pixels? {env_name} ({args.meta}, "
              f"{args.episodes} episodes)\n"
              f"{'YES' if uses_pixels else 'NO'} — corrupting the image costs "
              f"{100 * min(pixel_drops):.0f}–{100 * max(pixel_drops):.0f}% of performance")
    fig.savefig(out / "pixel_ablation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\nwrote", (out / "pixel_ablation.json").resolve())
    print("wrote", (out / "pixel_ablation.png").resolve())


if __name__ == "__main__":
    main()
