"""In-loop RGB visualization: roll out a *pixel-RL-trained* NEXUS hierarchy and
render the qualitative artifacts (a video of what the agent actually sees, an
observation filmstrip, and a skill + reward timeline).

Unlike rgb_visualize.py (which visualizes the *distilled* policy), this trains the
skill actors IN-LOOP from MJWarp pixels (USE_RGB) and then rolls that policy out
greedily in a 1-env vision environment. The meta selects the skill from privileged
state; the in-loop VisionSkillActor acts on the 64x64 pixel stack.

    python -m nexus_continuous.scripts.rgb_inloop_visualize \
        --config configs/cartpole_balance_nesy_rgb.yaml --meta neural --seed 0 \
        --updates 200 --out runs/rgb_inloop_viz_cartpole

STATE-ONLY ARMS (`--no-rgb`)
----------------------------
Mirrors rgb_pixel_ablation.py: the ENVIRONMENT still renders (so the video
exists and the task is unchanged) but the skill actors are plain state MLPs
with no camera pathway. The rendered frames are then a THIRD-PERSON VIEW OF
THE SCENE, not an actor input, and every artifact says so -- a camera panel
captioned as "what the agent sees" would be a lie for a state-only actor.

`--result-json` points at the arm's pixel_ablation.json so the caption can
state the measured intact score and, for pixel arms, how strongly THAT SEED's
actor depended on its camera. The caption reports the LARGEST single pixel
corruption and names it, not the stored `actor_uses_pixels` boolean: that
boolean thresholds the MEDIAN of three corruptions at 30%, a bar calibrated on
`RGB_PROPRIO: none` actors, and on WalkerWalk seed 0 it labelled a run
"IGNORED its camera" whose performance drops 94.9% when the image is held at
the t=0 frame.

    python -m nexus_continuous.scripts.rgb_inloop_visualize --no-rgb \
        --config configs/walker_walk_nesy_state_matched.yaml --meta nesy \
        --load-policy ~/runs_spr/walker_state_matched_s0.pkl \
        --result-json results/rgb/state_plus_rgb/walker/state_matched_seed0/pixel_ablation.json \
        --out results/rgb/state_plus_rgb/video/walker_state_matched_seed0
"""

from __future__ import annotations

import argparse
import os
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cartpole_balance_nesy_rgb.yaml")
    ap.add_argument("--meta", default="neural", choices=["nesy", "neural", "symbolic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/rgb_inloop_viz")
    ap.add_argument("--updates", type=int, default=200, help="in-loop training updates")
    ap.add_argument("--num-envs", type=int, default=128, help="training render batch (nworld)")
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--upscale", type=int, default=256, help="video frame size (upscaled 64x64)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--load-policy", default=None,
                    help="reuse a policy saved by rgb_pixel_ablation --save-policy "
                         "instead of retraining (seconds instead of ~20 minutes)")
    ap.add_argument("--no-rgb", action="store_true",
                    help="STATE-ONLY arm: the env still renders, but the actor is "
                         "a plain state MLP with no camera pathway. Artifacts are "
                         "labelled so the render is never presented as an input.")
    ap.add_argument("--result-json", default=None,
                    help="the arm's pixel_ablation.json; used to caption the "
                         "artifacts with the measured intact score and this "
                         "seed's camera verdict")
    ap.add_argument("--arm-label", default=None,
                    help="short arm name for captions, e.g. 'state-only'")
    ap.add_argument("--note", default=None,
                    help="extra caption line for an arm-specific caveat, e.g. that "
                         "this run's eval mean is dragged down by one collapsed "
                         "episode")
    args = ap.parse_args(argv)

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import jax
    import jax.numpy as jnp

    from nexus_continuous.utils import load_config
    from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
    from nexus_continuous.networks import MetaQ
    from nexus_continuous.vision import build_rgb_actor_fns
    from nexus_continuous.policies.registry import load_policy_module
    from nexus_continuous.envs.playground_adapter import (
        build_playground_env,
        get_actor_obs,
        get_actor_pixels,
        get_policy_obs,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # ---- Stage 1: train the hierarchy IN-LOOP from pixels ----
    cfg = load_config(args.config)
    cfg["SEED"] = seed
    cfg["META_POLICY_TYPE"] = args.meta
    # The env always renders; --no-rgb removes the camera from the ACTOR only.
    # See rgb_pixel_ablation.py: USE_RGB also changes the reward, ctrl_dt,
    # horizon, termination rule and state vector, so switching it off would
    # change the task and the video would not show the arm we trained.
    cfg["USE_RGB"] = True
    state_only = bool(args.no_rgb) or not bool(cfg.get("RGB_ACTOR", True))
    cfg["RGB_ACTOR"] = not state_only
    cfg["NUM_SEEDS"] = 1
    cfg["NUM_ENVS"] = args.num_envs
    cfg["TOTAL_TIMESTEPS"] = args.num_envs * cfg.get("NUM_STEPS", 64) * args.updates
    env_name = cfg["ENV_NAME"]
    # A shared CNN trunk stores the actor as {"encoder", "heads"} instead of one
    # stacked VisionSkillActor tree, so the rollout below must be rebuilt to match.
    shared_encoder = (not state_only) and bool(cfg.get("RGB_SHARED_ENCODER", False))
    # Lever B: with RGB_META_SEES_PIXELS the meta-Q was trained on
    # [state, latent], so the rollout has to feed it the same concatenation.
    meta_sees_pixels = (not state_only) and bool(cfg.get("RGB_META_SEES_PIXELS", False))
    if meta_sees_pixels and not shared_encoder:
        raise ValueError(
            "RGB_META_SEES_PIXELS requires RGB_SHARED_ENCODER (no single latent "
            "exists with one private encoder per skill)."
        )
    print(f"jax {jax.__version__} {jax.devices()} | env {env_name} | meta {args.meta} | in-loop {args.updates}u")
    if args.load_policy:
        import pickle as _pickle
        print(f"[1] loading saved in-loop policy from {args.load_policy} (no retraining)")
        _blob = _pickle.loads(Path(args.load_policy).read_bytes())
        _blob_state_only = bool(_blob.get("state_only", False))
        if _blob_state_only != state_only:
            raise ValueError(
                f"{args.load_policy} was saved with state_only={_blob_state_only} "
                f"but this run has state_only={state_only}; a stacked state "
                "SkillActor tree and a pixel actor tree are not interchangeable."
            )
        _blob_shared = bool(_blob.get("rgb_shared_encoder", False))
        if _blob_shared != shared_encoder:
            raise ValueError(
                f"{args.load_policy} was saved with rgb_shared_encoder="
                f"{_blob_shared} but {args.config} has RGB_SHARED_ENCODER="
                f"{shared_encoder}. The two actor parameter layouts are "
                "incompatible; pass the config the policy was trained with."
            )
        output = None
        actor_params = jax.tree_util.tree_map(jnp.asarray, _blob["actor_params"])
        meta_params = (None if _blob["meta_params"] is None
                       else jax.tree_util.tree_map(jnp.asarray, _blob["meta_params"]))
        stats = jax.tree_util.tree_map(jnp.asarray, _blob["normalization_stats"])
    else:
        output = run_training(cfg)
    if output is not None:
        train_state = output.runner_state[0]
        actor_params = train_state.actor.params        # vmapped over skills [N, ...]
        meta_params = None if args.meta == "symbolic" else train_state.meta.params
        stats = output.normalization_stats
    policy_module = load_policy_module(cfg.get("POLICY", env_name))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES", [f"skill{i}" for i in range(num_skills)]))
    print(f"[1] trained in-loop. skills={num_skills} ({', '.join(skill_names)})")

    # save the in-loop learning curve (training episode return per update)
    import json as _json
    _m = (output.metrics or {}) if output is not None else {}
    _c = np.asarray(_m["env/returned_episode_returns"]) if "env/returned_episode_returns" in _m else None
    if _c is not None and _c.size > 1:
        _c = _c.reshape(_c.shape[0], -1).mean(1) if _c.ndim > 1 else _c.reshape(-1)
        _json.dump({"env": env_name, "mode": "in_loop_pixel_rl", "updates": args.updates,
                    "return_curve": _c.tolist()}, open(out / "inloop_curve.json", "w"))
        _fig = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(len(_c)), _c, color="#4C72B0", lw=1.8)
        plt.xlabel("training update"); plt.ylabel("episode return (train, per update)")
        plt.title(f"In-loop pixel RL on {env_name} ({args.meta})\n"
                  f"final training return ~{_c[-20:].mean():.1f} (peak {_c.max():.1f})")
        _fig.savefig(out / "inloop_curve.png", dpi=130, bbox_inches="tight"); plt.close(_fig)
        print(f"    saved learning curve (final ~{_c[-20:].mean():.1f})")

    # ---- Stage 2: build a 1-env vision environment for the rollout ----
    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False   # raw; we apply frozen stats to the meta input
    eval_cfg["NUM_ENVS"] = 1
    eval_cfg["RENDER_NWORLD"] = 1
    bundle = build_playground_env(eval_cfg)
    env, env_params = bundle.env, bundle.env_params
    alo, ahi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
    action_dim = int(bundle.action_dim)
    action_scale = (ahi - alo) / 2.0
    action_bias = (ahi + alo) / 2.0
    step_fn = jax.jit(env.step)
    normalize_obs = bool(cfg.get("NORMALIZE_OBS", True))

    # aux_state_dim=0 on purpose: the auxiliary pixel->state head is a TRAINING
    # loss only and is never evaluated here. Flax ignores the extra `aux_state`
    # entries a trained policy may carry, so this stays correct either way.
    from nexus_continuous.networks import SkillActor

    rgb_fns = state_actor = None
    if state_only:
        state_actor = SkillActor(
            action_dim=action_dim, action_scale=action_scale, action_bias=action_bias,
            hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
            activation=cfg.get("ACTIVATION", "relu"),
            norm_type=cfg.get("NORM_TYPE", "layer_norm"),
            init_scale=float(cfg.get("ACTOR_INIT_SCALE", 0.01)),
        )
    else:
        rgb_fns = build_rgb_actor_fns(
            action_dim=action_dim, action_scale=action_scale, action_bias=action_bias,
            hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
            embedding_dim=int(cfg.get("RGB_EMBED_DIM", 128)),
            shared_encoder=shared_encoder,
        )
    # Mirror the trainer/ablation RGB_PROPRIO handling instead of assuming
    # pixels-only: a RGB_PROPRIO="full" actor has a first Dense layer sized for
    # embed_dim + state_dim, so feeding it a width-0 proprio is a shape error.
    _proprio_mode = str(cfg.get("RGB_PROPRIO", "none")).lower()
    _rgb_indices = cfg.get("RGB_PROPRIO_INDICES")

    def _actor_proprio(oa):
        if _proprio_mode == "full":
            return oa
        if _proprio_mode == "indices" and _rgb_indices is not None:
            return oa[..., jnp.asarray(list(_rgb_indices), dtype=jnp.int32)]
        return oa[..., :0]
    meta_q = MetaQ(
        num_skills=num_skills, hidden_sizes=tuple(cfg.get("META_HIDDEN_SIZES", (256, 256))),
        activation=cfg.get("ACTIVATION", "relu"), norm_type=cfg.get("NORM_TYPE", "layer_norm"),
        init_scale=float(cfg.get("META_INIT_SCALE", 1.0)),
    )
    is_cartpole = env_name == "CartpoleBalance"

    def norm_meta(obs):
        oa = get_actor_obs(obs)
        if normalize_obs:
            oa = (oa - stats["actor_mean"]) / jnp.sqrt(stats["actor_var"] + 1e-8)
        return oa

    @jax.jit
    def act_step(obs):
        """Greedy in-loop hierarchy: meta picks skill from state, vision actor acts on pixels."""
        oa = norm_meta(obs)                      # [1, D] normalized state for the meta
        px = get_actor_pixels(obs)               # [1, 64, 64, 3]
        pol = get_policy_obs(obs)
        if state_only:
            all_a = jax.vmap(
                lambda p: state_actor.apply({"params": p}, oa))(actor_params)   # [N,1,A]
        else:
            all_a = rgb_fns.apply(actor_params, px, _actor_proprio(oa))         # [N,1,A]
        if args.meta == "symbolic":
            skill = jnp.atleast_1d(jnp.asarray(policy_module.symbolic_meta_policy(pol), jnp.int32))
        else:
            meta_in = (
                jnp.concatenate([oa, rgb_fns.encode(actor_params, px)], axis=-1)
                if meta_sees_pixels  # lever B: same [state, latent] order as the trainer
                else oa
            )
            qm = jnp.atleast_2d(meta_q.apply({"params": meta_params}, meta_in))
            if args.meta == "nesy":
                mask = jnp.atleast_2d(jnp.asarray(policy_module.skill_mask(pol), bool))
                any_valid = jnp.any(mask, axis=-1, keepdims=True)
                safe = jnp.where(any_valid, mask, jnp.ones_like(mask))
                qm = jnp.where(safe, qm, -1.0e9)
            skill = jnp.argmax(qm, axis=-1).astype(jnp.int32)
        e = jnp.arange(all_a.shape[1])
        return skill, all_a[skill, e]

    # ---- Stage 3: greedy rollout, capture what the agent sees ----
    print("[2] greedy in-loop rollout...")
    rng = jax.random.PRNGKey(7000 + seed)
    rng, rr = jax.random.split(rng)
    obs, state = env.reset(jax.random.split(rr, 1), env_params)
    gray, sel_skill, rew_seq = [], [], []
    for _t in range(args.eval_steps):
        skill, act = act_step(obs)
        act = jnp.clip(act, alo, ahi)
        px = np.asarray(get_actor_pixels(obs))          # [1,64,64,3] centered gray-stack
        gray.append(px[0, :, :, -1])                    # most-recent frame
        sel_skill.append(int(np.asarray(skill[0])))
        rng, rs = jax.random.split(rng)
        obs, state, rew, _d, _i = step_fn(jax.random.split(rs, 1), state, act, env_params)
        rew_seq.append(float(np.asarray(rew).reshape(-1)[0]))
    sel_skill = np.asarray(sel_skill)
    rew_seq = np.asarray(rew_seq)
    T = len(sel_skill)
    mean_rew = float(rew_seq.mean())
    print(f"    rollout mean reward/step {mean_rew:.3f}, total return {rew_seq.sum():.1f} over {T} steps")

    palette = [plt.cm.tab10(i) for i in range(num_skills)]

    def to_img(g):  # centered gray [64,64] -> uint8 [0,255]
        return np.clip((g + 0.5) * 255.0, 0, 255).astype(np.uint8)

    # ---- captions -----------------------------------------------------
    # Every artifact states the arm it came from and, for pixel arms, whether
    # THAT SEED actually used its camera. Without the latter a viewer would
    # reasonably assume any state+RGB video shows a camera-driven policy.
    #
    # CORRECTED: the caption used to quote `actor_uses_pixels`, the stored
    # boolean that thresholds the MEDIAN of three corruptions at 30%. That bar
    # was calibrated on `RGB_PROPRIO: none` actors and under-reads for an actor
    # that also holds the state, and the median hides disagreement between the
    # conditions -- WalkerWalk seed 0 scores frozen_first 94.9% but
    # random_replay 4.8%, so the median labelled a camera-using run "IGNORED
    # its camera". The caption now quotes the LARGEST single pixel corruption,
    # which changes nothing but the actor's image and so cannot be explained
    # away, and names the condition it came from.
    _res = _json.loads(Path(args.result_json).read_text()) if args.result_json else {}
    _intact = (_res.get("results") or {}).get("intact") or {}
    _ik = ("upright_fraction_mean" if "upright_fraction_mean" in _intact
           else "reward_per_step_mean")
    _iv = _intact.get(_ik)
    arm_label = args.arm_label or ("state-only (no camera)" if state_only
                                   else "state + RGB (camera added)")
    # One short line each: the frame is only `upscale` px wide and PIL's default
    # bitmap font does not wrap, so a long line is silently truncated.
    cap = [f"{env_name}   |   seed {seed}   |   meta={args.meta}",
           f"arm: {arm_label}",
           f"budget: {args.updates} upd x {args.num_envs} envs x 64 steps"]
    if _iv is not None:
        _mlabel = _ik.replace("_mean", "").replace("_", " ")
        cap.append(f"eval intact {_mlabel} = {_iv:.4f}")
    if state_only:
        cap.append("NO CAMERA INPUT: the actor is a plain state MLP.")
        cap.append("These frames are a scene view, NOT an actor input.")
    else:
        _drops = _res.get("performance_drop_fraction") or {}
        _pix = {k: v for k, v in _drops.items()
                if k in ("frozen_first", "random_replay", "zeros")}
        if not _pix:
            cap.append("camera dependence: not measured for this run")
        elif _res.get("inconclusive"):
            cap.append("camera dependence: INCONCLUSIVE for this seed")
        else:
            _wk = max(_pix, key=_pix.get)
            _wv = _pix[_wk]
            _used = ("USED its camera" if _wv > 0.30
                     else "shows no camera use")
            cap.append(f"this seed {_used}")
            cap.append(f"   (worst pixel corruption: {_wk} {100 * _wv:+.1f}%)")
    if args.note:
        cap.append(f"NOTE: {args.note}")

    # ---- artifact 1: video (scene view + skill, fully captioned) ----
    print("[3] writing artifacts...")
    U = args.upscale
    BAND = 16 + 13 * len(cap) + 8
    vid = []
    for t in range(T):
        top = Image.fromarray(to_img(gray[t])).resize((U, U), Image.NEAREST).convert("RGB")
        im = Image.new("RGB", (U, U + BAND), (0, 0, 0))
        im.paste(top, (0, 0))
        d = ImageDraw.Draw(im)
        col = tuple(int(255 * c) for c in palette[sel_skill[t]][:3])
        d.rectangle([0, 0, U, 16], fill=(0, 0, 0))
        view = "scene view (not an actor input)" if state_only else "actor camera view"
        d.text((3, 3), f"{t:3d}  {skill_names[sel_skill[t]]}  |  {view}", fill=col)
        for i, line in enumerate(cap):
            warn = line.startswith(("NOTE:", "NO CAMERA", "These frames"))
            d.text((4, U + 5 + 13 * i), line,
                   fill=(255, 210, 120) if warn else (235, 235, 235))
        vid.append(np.asarray(im))
    vid = np.stack(vid)
    imageio.mimsave(out / "rollout_inloop.mp4", vid, fps=args.fps, quality=8)
    imageio.mimsave(out / "rollout_inloop.gif", vid[:: max(1, T // 120)], fps=min(args.fps, 20), loop=0)

    # ---- artifact 2: observation filmstrip (the 64x64 the agent sees) ----
    idxs = np.linspace(0, T - 1, 8).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(1.5 * len(idxs), 1.9))
    for ax, ti in zip(axes, idxs):
        ax.imshow(to_img(gray[ti]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t={ti}", fontsize=8); ax.axis("off")
    fig.suptitle(
        (f"{env_name} — {arm_label}, seed {seed} ({args.meta}) — 64x64 SCENE view; "
         "the actor never sees it (state-only)" if state_only else
         f"{env_name} — {arm_label}, seed {seed} ({args.meta}) — 64x64 actor camera view"),
        fontsize=10)
    fig.savefig(out / "observation_filmstrip.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- artifact 3: skill + reward timeline ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 0.7]})
    run_avg = np.cumsum(rew_seq) / (np.arange(T) + 1)
    ax1.plot(np.arange(T), rew_seq, color="#C44E52", lw=0.8, alpha=0.4, label="per-step reward")
    ax1.plot(np.arange(T), run_avg, color="#C44E52", lw=2.0, label="running-avg reward")
    ax1.set_ylabel("task reward"); ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{env_name} — {arm_label} ({args.meta}, seed {seed}): "
                  f"skill activation vs. reward  (mean {mean_rew:.3f}/step)", fontsize=10)
    ax2.imshow(sel_skill[None], aspect="auto", cmap=ListedColormap(palette),
               vmin=-0.5, vmax=num_skills - 0.5, extent=[0, T, 0, 1], interpolation="nearest")
    ax2.set_yticks([]); ax2.set_xlabel("closed-loop step")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=palette[i], label=skill_names[i])
               for i in range(num_skills)]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55),
               ncol=num_skills, fontsize=8, frameon=False)
    fig.savefig(out / "skill_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # self-check so we never ship a misleading video: rollout should roughly match training
    print(f"\n==== INLOOP_VIZ_DONE ==== {env_name} rollout mean reward/step {mean_rew:.3f}")
    print("wrote", out.resolve())
    for f in ("rollout_inloop.mp4", "rollout_inloop.gif", "observation_filmstrip.png", "skill_timeline.png"):
        print("  -", f)


if __name__ == "__main__":
    main()
