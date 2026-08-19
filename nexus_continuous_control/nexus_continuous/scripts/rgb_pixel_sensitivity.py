"""Is a trained in-loop pixel actor sensitive to its image input AT ALL?

The closed-loop ablation (`rgb_pixel_ablation.py`) can be confounded: if the
eval-time renderer produced static frames, "corrupting the pixels changes
nothing" would be an artifact rather than a property of the policy. This script
separates the two, open-loop and cheaply (loads a saved policy, no training):

  A. RENDER CHECK   -- do the frames actually change over a rollout?
                       (temporal std per pixel, frame-to-frame abs diff)
  B. ACTOR SENSITIVITY -- feed each skill actor many genuinely different real
                       frames and measure the spread of its output actions,
                       relative to the action range. A blind network outputs a
                       constant: spread ~ 0.
  C. STATE TRACKING -- correlate each skill's action with the true pole angle /
                       root velocity. A pixel actor that decodes the scene should
                       track the state it is supposed to react to.

    python -m nexus_continuous.scripts.rgb_pixel_sensitivity \\
        --config configs/cartpole_balance_nesy_rgb.yaml --meta neural \\
        --load-policy runs/abl_cartpole.pkl --steps 250
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def _mjx_data(state):
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
    ap.add_argument("--config", default="configs/cartpole_balance_nesy_rgb.yaml")
    ap.add_argument("--meta", default="neural", choices=["nesy", "neural", "symbolic"])
    ap.add_argument("--load-policy", required=True, help="pkl from rgb_pixel_ablation --save-policy")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", default=None, help="optional dir for the json report")
    args = ap.parse_args(argv)

    import numpy as np
    import jax
    import jax.numpy as jnp

    from nexus_continuous.utils import load_config
    from nexus_continuous.networks import MetaQ
    from nexus_continuous.vision import build_rgb_actor_fns
    from nexus_continuous.policies.registry import load_policy_module
    from nexus_continuous.envs.playground_adapter import (
        build_playground_env, get_actor_obs, get_actor_pixels, get_policy_obs,
    )

    blob = pickle.loads(Path(args.load_policy).read_bytes())
    actor_params = jax.tree_util.tree_map(jnp.asarray, blob["actor_params"])
    meta_params = (None if blob["meta_params"] is None
                   else jax.tree_util.tree_map(jnp.asarray, blob["meta_params"]))
    stats = jax.tree_util.tree_map(jnp.asarray, blob["normalization_stats"])
    print(f"loaded {args.load_policy}: env={blob.get('env')} meta={blob.get('meta')} "
          f"train_return={blob.get('final_train_return')}")

    cfg = load_config(args.config)
    cfg["USE_RGB"] = True
    cfg["META_POLICY_TYPE"] = args.meta
    env_name = cfg["ENV_NAME"]
    # A shared CNN trunk stores the actor as {"encoder", "heads"} instead of one
    # stacked VisionSkillActor tree. Cross-check the blob's stamp against the
    # config so a wrong --config fails here rather than silently reporting
    # "effectively blind" from an unreadable parameter tree.
    shared_encoder = bool(cfg.get("RGB_SHARED_ENCODER", False))
    blob_shared = bool(blob.get("rgb_shared_encoder", False))
    if blob_shared != shared_encoder:
        raise ValueError(
            f"{args.load_policy} was saved with rgb_shared_encoder={blob_shared} "
            f"but {args.config} has RGB_SHARED_ENCODER={shared_encoder}. The two "
            "actor parameter layouts are incompatible; pass the config the "
            "policy was trained with."
        )
    # Lever B: a meta-Q trained with RGB_META_SEES_PIXELS has a first Dense kernel
    # of width state_dim + RGB_EMBED_DIM, so the flag must match the blob or the
    # meta cannot be reconstructed.
    meta_sees_pixels = bool(cfg.get("RGB_META_SEES_PIXELS", False))
    blob_meta_px = bool(blob.get("rgb_meta_sees_pixels", False))
    if blob_meta_px != meta_sees_pixels:
        raise ValueError(
            f"{args.load_policy} was saved with rgb_meta_sees_pixels={blob_meta_px} "
            f"but {args.config} has RGB_META_SEES_PIXELS={meta_sees_pixels}; the "
            "meta-Q input widths differ (state vs state+RGB_EMBED_DIM)."
        )
    if meta_sees_pixels and not shared_encoder:
        raise ValueError(
            "RGB_META_SEES_PIXELS requires RGB_SHARED_ENCODER (no single latent "
            "exists with one private encoder per skill)."
        )
    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NUM_ENVS"] = 1
    eval_cfg["RENDER_NWORLD"] = 1
    bundle = build_playground_env(eval_cfg)
    env, env_params = bundle.env, bundle.env_params
    alo, ahi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
    step_fn = jax.jit(env.step)
    normalize_obs = bool(cfg.get("NORMALIZE_OBS", True))

    policy_module = load_policy_module(cfg.get("POLICY", env_name))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES",
                               [f"skill{i}" for i in range(num_skills)]))
    # aux_state_dim=0 on purpose: the auxiliary pixel->state head is a TRAINING
    # loss only and is never evaluated here. Flax ignores the extra `aux_state`
    # entries a trained blob may carry, so this stays correct either way.
    rgb_fns = build_rgb_actor_fns(
        action_dim=int(bundle.action_dim), action_scale=(ahi - alo) / 2.0,
        action_bias=(ahi + alo) / 2.0,
        hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
        embedding_dim=int(cfg.get("RGB_EMBED_DIM", 128)),
        shared_encoder=shared_encoder,
    )
    meta_q = MetaQ(num_skills=num_skills,
                   hidden_sizes=tuple(cfg.get("META_HIDDEN_SIZES", (256, 256))),
                   activation=cfg.get("ACTIVATION", "relu"),
                   norm_type=cfg.get("NORM_TYPE", "layer_norm"),
                   init_scale=float(cfg.get("META_INIT_SCALE", 1.0)))

    def _norm(obs):
        oa = get_actor_obs(obs)
        if normalize_obs:
            oa = (oa - stats["actor_mean"]) / jnp.sqrt(stats["actor_var"] + 1e-8)
        return oa

    @jax.jit
    def all_skill_actions(px):
        """Every skill's action for one pixel stack -> [num_skills, 1, A]."""
        proprio = jnp.zeros((px.shape[0], 0), jnp.float32)
        return rgb_fns.apply(actor_params, px, proprio)

    @jax.jit
    def pick_skill(obs):
        oa = _norm(obs)
        pol = get_policy_obs(obs)
        if args.meta == "symbolic":
            return jnp.atleast_1d(jnp.asarray(policy_module.symbolic_meta_policy(pol), jnp.int32))
        if meta_sees_pixels:
            # Lever B: the meta-Q was trained on [state, latent] (same order as
            # the trainer's `_meta_values`). This probe only ever feeds it the real
            # frames; the corruption experiments live in rgb_pixel_ablation.py.
            oa = jnp.concatenate(
                [oa, rgb_fns.encode(actor_params, get_actor_pixels(obs))], axis=-1
            )
        qm = jnp.atleast_2d(meta_q.apply({"params": meta_params}, oa))
        if args.meta == "nesy":
            mask = jnp.atleast_2d(jnp.asarray(policy_module.skill_mask(pol), bool))
            any_valid = jnp.any(mask, axis=-1, keepdims=True)
            qm = jnp.where(jnp.where(any_valid, mask, jnp.ones_like(mask)), qm, -1.0e9)
        return jnp.argmax(qm, axis=-1).astype(jnp.int32)

    # ---- roll the real policy, collecting frames + the state it should react to ----
    rng = jax.random.PRNGKey(31337)
    rng, rr = jax.random.split(rng)
    obs, state = env.reset(jax.random.split(rr, 1), env_params)
    frames, angles, skills = [], [], []
    for _t in range(args.steps):
        px = get_actor_pixels(obs)
        frames.append(np.asarray(px)[0])
        sd = _mjx_data(state)
        if sd is not None:
            angles.append(float(np.asarray(sd.qpos[0, min(1, sd.qpos.shape[1] - 1)])))
        k = pick_skill(obs)
        skills.append(int(np.asarray(k)[0]))
        act = jnp.clip(all_skill_actions(px)[k, jnp.arange(1)], alo, ahi)
        rng, rs = jax.random.split(rng)
        obs, state, _r, _d, _i = step_fn(jax.random.split(rs, 1), state, act, env_params)

    F = np.stack(frames)                       # [T, H, W, C]
    print("\n== A. RENDER CHECK ==")
    temporal_std = float(F.std(axis=0).mean())
    frame_diff = float(np.abs(np.diff(F[..., -1], axis=0)).mean())
    print(f"  frame value range        : [{F.min():.4f}, {F.max():.4f}]")
    print(f"  per-pixel temporal std   : {temporal_std:.6f}")
    print(f"  mean |frame(t)-frame(t-1)|: {frame_diff:.6f}")
    render_alive = temporal_std > 1e-4
    print(f"  -> renderer is {'PRODUCING CHANGING IMAGES' if render_alive else 'STATIC (BROKEN)'}")

    # ---- B. actor sensitivity to genuinely different frames ----
    print("\n== B. ACTOR SENSITIVITY (spread of action over different real frames) ==")
    idx = np.linspace(0, len(F) - 1, min(128, len(F))).astype(int)
    acts = np.stack([np.asarray(all_skill_actions(jnp.asarray(F[i][None])))[:, 0]
                     for i in idx])                        # [N, num_skills, A]
    rng_span = float(np.asarray(ahi - alo).mean())
    zero_act = np.asarray(all_skill_actions(jnp.zeros_like(jnp.asarray(F[0][None]))))[:, 0]
    sens = {}
    for k in range(num_skills):
        spread = float(acts[:, k].std(axis=0).mean())
        rel = spread / max(rng_span, 1e-9)
        d_zero = float(np.abs(acts[:, k].mean(0) - zero_act[k]).mean())
        sens[skill_names[k]] = {"action_std_over_frames": spread,
                                "relative_to_action_range": rel,
                                "shift_vs_blank_image": d_zero}
        print(f"  {skill_names[k]:>22s}: std {spread:.6f} "
              f"({100 * rel:.3f}% of action range) | |mean - blank-image action| {d_zero:.6f}")
    max_rel = max(v["relative_to_action_range"] for v in sens.values())
    responsive = max_rel > 0.01
    print(f"  -> actor is {'RESPONSIVE to pixels' if responsive else 'EFFECTIVELY BLIND'} "
          f"(largest spread {100 * max_rel:.3f}% of the action range)")

    # ---- C. does the action track the state it should react to? ----
    print("\n== C. STATE TRACKING (corr between action and pole angle) ==")
    corr = {}
    if angles:
        a = np.asarray(angles)[idx]
        for k in range(num_skills):
            ak = acts[:, k, 0]
            c = (float(np.corrcoef(ak, a)[0, 1])
                 if ak.std() > 1e-8 and a.std() > 1e-8 else float("nan"))
            corr[skill_names[k]] = c
            print(f"  {skill_names[k]:>22s}: r = {c:+.3f}")

    report = {
        "env": env_name, "meta": args.meta, "policy": args.load_policy,
        "train_return": blob.get("final_train_return"),
        "render": {"temporal_std": temporal_std, "frame_diff": frame_diff,
                   "alive": bool(render_alive)},
        "actor_sensitivity": sens, "actor_responsive": bool(responsive),
        "action_state_correlation": corr,
        "skill_histogram": np.bincount(np.asarray(skills), minlength=num_skills).tolist(),
    }
    print("\n==== CONCLUSION ====")
    if not render_alive:
        print("  Renderer is static -> the closed-loop ablation is INVALID; fix the render first.")
    elif not responsive:
        print("  Images change, but the actor's output does NOT -> the CNN is effectively")
        print("  BLIND. The hierarchy's performance comes from the state-based meta-policy.")
    else:
        print("  Images change AND the actor responds -> the pixel claim is supported;")
        print("  a flat closed-loop ablation would then mean the task tolerates the error.")

    if args.out:
        o = Path(args.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "pixel_sensitivity.json").write_text(json.dumps(report, indent=2))
        print("wrote", (o / "pixel_sensitivity.json").resolve())


if __name__ == "__main__":
    main()
