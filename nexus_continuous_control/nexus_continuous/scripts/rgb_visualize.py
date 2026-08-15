"""RGB extension -- publication-style demonstration artifacts for one run.

Trains the real NEXUS hierarchy (like rgb_distill_nexus), behavior-clones its
skills into a VisionSkillActor, then runs ONE instrumented closed-loop rollout of
the PIXEL hierarchy and renders the qualitative figures used to demonstrate
pixel-based control. Env is taken from --config's ENV_NAME and rendered offline,
so it works for CartpoleBalance (fixed cam) and the locomotion suite
(CheetahRun/WalkerWalk/HopperHop, tracking side cameras).

  rollout_pixel.mp4 / .gif   video of the pixel hierarchy acting, each frame
                             annotated with the meta-selected skill (hi-res render)
  observation_filmstrip.png  the 64x64 grayscale frames the skill actor ACTUALLY
                             sees (the network input, not the human-facing render)
  skill_timeline.png         the trajectory (cartpole: pole angle + cart position;
                             others: per-step task reward) aligned to a colored
                             skill-activation strip -- the "disentangled skills
                             from pixels" figure
  action_tracking.png        per-skill teacher-vs-student action scatter on held-out
                             data (pooled over action dims) with Pearson correlation

    MUJOCO_GL=egl python -m nexus_continuous.scripts.rgb_visualize \
        --config configs/cheetah_run_neural.yaml --meta neural --seed 0 \
        --teacher-steps 3000 --out runs/rgb_viz_cheetah
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
    raise AttributeError("could not locate .data on the wrapped env state")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cartpole_balance_neural.yaml")
    ap.add_argument("--meta", default="neural", choices=["nesy", "neural", "symbolic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/rgb_viz")
    ap.add_argument("--camera", type=int, default=0, help="mj_model camera id (0 = cartpole / locomotion side)")
    ap.add_argument("--teacher-steps", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--min-samples", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--render-res", type=int, default=240, help="hi-res video render size (÷16)")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import jax
    import jax.numpy as jnp
    import optax
    import mujoco
    from mujoco_playground import registry

    from nexus_continuous.utils import load_config
    from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
    from nexus_continuous.networks import MetaQ, SkillActor
    from nexus_continuous.policies.registry import load_policy_module
    from nexus_continuous.envs.playground_adapter import (
        build_playground_env,
        get_actor_obs,
        get_policy_obs,
    )
    from nexus_continuous.vision import VisionSkillActor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    np.random.seed(1234 + seed)  # reproducible BC shuffles / held-out split
    R = args.render_res

    cfg = load_config(args.config)
    cfg["SEED"] = seed
    cfg["META_POLICY_TYPE"] = args.meta
    cfg["NUM_SEEDS"] = 1
    env_name = cfg["ENV_NAME"]
    is_cartpole = env_name == "CartpoleBalance"
    print("jax", jax.__version__, jax.devices(), "| env", env_name, "| meta", args.meta, "| seed", seed)

    # ---- renderer (state sim, impl='jax' => no warp) for the config's env ----
    rcfg = registry.get_default_config(env_name)
    rcfg.impl = "jax"
    render_env = registry.load(env_name, rcfg)
    mj_model = getattr(render_env, "mj_model", None) or getattr(render_env, "_mj_model")
    cam = args.camera
    try:
        _d = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, _d)
        with mujoco.Renderer(mj_model, height=64, width=64) as _r:
            _r.update_scene(_d, camera=cam)
    except Exception:
        cam = -1
    print(f"camera={cam}")

    def to_gray64(rgb):
        """hi-res RGB uint8 -> [64,64] float32 centered (mirrors the training preproc)."""
        small = np.asarray(Image.fromarray(rgb).resize((64, 64), Image.BILINEAR), np.float32)
        return small.mean(-1) / 255.0 - 0.5

    def stack3(grays):
        g = np.asarray(grays)
        return np.stack([g[i - 2 : i + 1] for i in range(2, len(g))], 0).transpose(0, 2, 3, 1)

    empty = lambda b: jnp.zeros((b, 0), jnp.float32)

    # ---- Stage 1: train the state NEXUS teacher ----
    print(f"[1] training state NEXUS teacher ({args.meta} meta)...")
    output = run_training(cfg)
    train_state = output.runner_state[0]
    actor_params = train_state.actor.params
    stats = output.normalization_stats
    normalize_obs = bool(cfg.get("NORMALIZE_OBS", True))
    policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES", [f"skill{i}" for i in range(num_skills)]))
    print(f"    teacher trained. skills={num_skills} ({', '.join(skill_names)})")

    # eval env (raw obs; frozen stats in norm_actor) + action space
    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NUM_ENVS"] = 1
    bundle = build_playground_env(eval_cfg)
    env, env_params = bundle.env, bundle.env_params
    alo, ahi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
    action_dim = int(bundle.action_dim)
    action_scale = (ahi - alo) / 2.0
    action_bias = (ahi + alo) / 2.0
    step_fn = jax.jit(env.step)

    actor = SkillActor(
        action_dim=action_dim, action_scale=action_scale, action_bias=action_bias,
        hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
        activation=cfg.get("ACTIVATION", "relu"), norm_type=cfg.get("NORM_TYPE", "layer_norm"),
    )
    meta_q = MetaQ(
        num_skills=num_skills, hidden_sizes=tuple(cfg.get("META_HIDDEN_SIZES", (256, 256))),
        activation=cfg.get("ACTIVATION", "relu"), norm_type=cfg.get("NORM_TYPE", "layer_norm"),
        init_scale=float(cfg.get("META_INIT_SCALE", 1.0)),
    )
    meta_params = None if args.meta == "symbolic" else train_state.meta.params

    def norm_actor(obs):
        oa = get_actor_obs(obs)
        if normalize_obs:
            oa = (oa - stats["actor_mean"]) / jnp.sqrt(stats["actor_var"] + 1e-8)
        return oa

    @jax.jit
    def teacher_step(obs):
        oa = norm_actor(obs)
        all_a = jax.vmap(lambda p: actor.apply({"params": p}, oa))(actor_params)
        pol = get_policy_obs(obs)
        if args.meta == "symbolic":
            skill = jnp.atleast_1d(jnp.asarray(policy_module.symbolic_meta_policy(pol), jnp.int32))
        else:
            qm = jnp.atleast_2d(meta_q.apply({"params": meta_params}, oa))
            if args.meta == "nesy":
                mask = jnp.atleast_2d(jnp.asarray(policy_module.skill_mask(pol), bool))
                any_valid = jnp.any(mask, axis=-1, keepdims=True)
                safe = jnp.where(any_valid, mask, jnp.ones_like(mask))
                qm = jnp.where(safe, qm, -1.0e9)
            skill = jnp.argmax(qm, axis=-1).astype(jnp.int32)
        e = jnp.arange(all_a.shape[1])
        return skill, all_a[skill, e]

    _rdata = mujoco.MjData(mj_model)
    _renderer = mujoco.Renderer(mj_model, height=R, width=R)  # one persistent GL context

    def render_rgb(qpos, qvel):
        _rdata.qpos[:] = np.asarray(qpos)
        _rdata.qvel[:] = np.asarray(qvel)
        mujoco.mj_forward(mj_model, _rdata)
        _renderer.update_scene(_rdata, camera=cam)
        return _renderer.render()

    # ---- Stage 2: teacher rollout + record (hi-res render -> gray64 dataset) ----
    print("[2] rolling out teacher, collecting distillation data...")
    rng = jax.random.PRNGKey(1000 + seed)
    rng, rr = jax.random.split(rng)
    obs, state = env.reset(jax.random.split(rr, 1), env_params)
    grays, skill_seq, act_seq = [], [], []
    for _t in range(args.teacher_steps):
        skill, act = teacher_step(obs)
        act = jnp.clip(act, alo, ahi)
        sd = _mjx_data(state)
        grays.append(to_gray64(render_rgb(sd.qpos[0], sd.qvel[0])))
        skill_seq.append(int(np.asarray(skill[0])))
        act_seq.append(np.asarray(act[0]))
        rng, rs = jax.random.split(rng)
        obs, state, _r, _d, _i = step_fn(jax.random.split(rs, 1), state, act, env_params)
    X = stack3(grays)
    skills_ds = np.asarray(skill_seq[2:], np.int32)
    Y = np.asarray(act_seq[2:], np.float32)  # [T-2, A]
    print(f"    {len(X)} samples; skill hist={np.bincount(skills_ds, minlength=num_skills)}")

    # ---- Stage 3: distill per-skill vision actors ----
    print("[3] distilling per-skill vision actors...")
    vactor = VisionSkillActor(action_dim=action_dim, action_scale=action_scale,
                              action_bias=action_bias, hidden_sizes=(128, 128),
                              embedding_dim=args.embed_dim)
    opt = optax.adam(3e-4)

    def loss_fn(p, x, y):
        return jnp.mean((vactor.apply({"params": p}, x, empty(x.shape[0])) - y) ** 2)

    @jax.jit
    def bc_step(p, st, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    skill_params = []
    ho_t, ho_p, ho_k = [], [], []  # held-out teacher/pixel actions (pooled over dims) + skill id
    for k in range(num_skills):
        m = skills_ds == k
        n = int(m.sum())
        if n < args.min_samples:
            print(f"    skill {k} ({skill_names[k]}): {n} samples -> teacher fallback")
            skill_params.append(None)
            continue
        xk_all, yk_all = X[m], Y[m]
        n_ho = max(1, int(0.15 * n))
        perm0 = np.random.permutation(n)
        ho_idx, tr_idx = perm0[:n_ho], perm0[n_ho:]
        ntr = len(tr_idx)
        bs = min(args.batch_size, ntr)
        xk, yk = jnp.asarray(xk_all[tr_idx]), jnp.asarray(yk_all[tr_idx])
        p = vactor.init(jax.random.PRNGKey(k), xk[:1], empty(1))["params"]
        st = opt.init(p)
        last = float("nan")
        for _e in range(args.epochs):
            perm = np.random.permutation(ntr)
            for i in range(0, ntr, bs):
                p, st, last = bc_step(p, st, xk[perm[i : i + bs]], yk[perm[i : i + bs]])
                last = float(last)
        skill_params.append(p)
        pred = np.asarray(vactor.apply({"params": p}, jnp.asarray(xk_all[ho_idx]), empty(len(ho_idx))))
        tt = yk_all[ho_idx]  # [n_ho, A]
        ho_mse = float(np.mean((pred - tt) ** 2))
        ho_t.append(tt.reshape(-1)); ho_p.append(pred.reshape(-1))
        ho_k.append(np.full(tt.size, k))
        print(f"    skill {k} ({skill_names[k]}): {ntr} train / {n_ho} held-out, "
              f"train MSE {last:.4f}, held-out MSE {ho_mse:.4f}")
    ho_t = np.concatenate(ho_t) if ho_t else np.zeros(0)
    ho_p = np.concatenate(ho_p) if ho_p else np.zeros(0)
    ho_k = np.concatenate(ho_k).astype(int) if ho_k else np.zeros(0, int)

    # ---- Stage 4: instrumented closed-loop rollout of the PIXEL hierarchy ----
    print("[4] instrumented closed-loop pixel rollout...")
    rng2 = jax.random.PRNGKey(5000 + seed)
    rng2, rr2 = jax.random.split(rng2)
    obs, state = env.reset(jax.random.split(rr2, 1), env_params)
    buf = []
    hires, obs64, sel_skill, rew_seq, used_pix = [], [], [], [], []
    angle_deg, cart_pos = [], []
    for _t in range(args.eval_steps):
        skill_t, teacher_act = teacher_step(obs)
        k = int(np.asarray(skill_t[0]))
        sd = _mjx_data(state)
        frame = render_rgb(sd.qpos[0], sd.qvel[0])
        g = to_gray64(frame)
        buf.append(g)
        buf = buf[-3:]
        pixel_used = False
        if skill_params[k] is not None and len(buf) == 3:
            stack = jnp.asarray(np.stack(buf, -1)[None])
            act = jnp.asarray(np.asarray(vactor.apply({"params": skill_params[k]}, stack, empty(1))))
            pixel_used = True
        else:
            act = teacher_act
        act = jnp.clip(act, alo, ahi)
        hires.append(frame)
        obs64.append(g)
        sel_skill.append(k)
        used_pix.append(pixel_used)
        if is_cartpole:
            ang = float(np.asarray(sd.qpos[0, 1]))
            angle_deg.append(np.degrees(np.arctan2(np.sin(ang), np.cos(ang))))
            cart_pos.append(float(np.asarray(sd.qpos[0, 0])))
        rng2, rs2 = jax.random.split(rng2)
        obs, state, rew, _d, _i = step_fn(jax.random.split(rs2, 1), state, act, env_params)
        rew_seq.append(float(np.asarray(rew).reshape(-1)[0]))

    sel_skill = np.asarray(sel_skill)
    rew_seq = np.asarray(rew_seq)
    used_pix = np.asarray(used_pix)
    T = len(sel_skill)
    palette = [plt.cm.tab10(i) for i in range(num_skills)]

    # ---- artifact 1: annotated video (mp4 + gif) ----
    print("[5] writing artifacts...")
    vid = []
    for t in range(T):
        im = Image.fromarray(hires[t]).convert("RGB")
        d = ImageDraw.Draw(im)
        col = tuple(int(255 * c) for c in palette[sel_skill[t]][:3])
        d.rectangle([0, 0, R, 16], fill=(0, 0, 0))
        tag = skill_names[sel_skill[t]] + ("" if used_pix[t] else " (teacher)")
        d.text((3, 3), f"{t:3d}  {tag}", fill=col)
        vid.append(np.asarray(im))
    vid = np.stack(vid)
    imageio.mimsave(out / "rollout_pixel.mp4", vid, fps=args.fps, quality=8)
    imageio.mimsave(out / "rollout_pixel.gif", vid[:: max(1, T // 120)], fps=min(args.fps, 20), loop=0)

    # ---- artifact 2: observation filmstrip (what the network sees) ----
    idxs = np.linspace(0, T - 1, 8).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(1.5 * len(idxs), 1.9))
    for ax, ti in zip(axes, idxs):
        ax.imshow(obs64[ti] + 0.5, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={ti}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"What the pixel skill actor sees (64x64 grayscale) - {env_name} ({args.meta})", fontsize=10)
    fig.savefig(out / "observation_filmstrip.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- artifact 3: skill-activation timeline over the trajectory ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 0.7]})
    if is_cartpole:
        angle_deg = np.asarray(angle_deg); cart_pos = np.asarray(cart_pos)
        ax1.axhspan(-14.3, 14.3, color="#8fbf8f", alpha=0.22, label="upright band |θ|<0.25 rad")
        ln1 = ax1.plot(np.arange(T), angle_deg, color="#C44E52", lw=1.5, label="pole angle (deg)")
        ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax1.set_ylabel("pole angle (deg)", color="#C44E52")
        ax1.tick_params(axis="y", labelcolor="#C44E52")
        ax1.set_ylim(-20, 20)
        axc = ax1.twinx()
        ln2 = axc.plot(np.arange(T), cart_pos, color="#4C72B0", lw=1.2, alpha=0.85, label="cart position (m)")
        axc.axhline(1.0, color="#4C72B0", ls=":", lw=0.9, alpha=0.7)
        axc.axhline(-1.0, color="#4C72B0", ls=":", lw=0.9, alpha=0.7)
        axc.set_ylabel("cart position (m)  [fail at ±1.0]", color="#4C72B0")
        axc.tick_params(axis="y", labelcolor="#4C72B0")
        axc.set_ylim(-1.15, 1.15)
        lns = ln1 + ln2
        ax1.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=8)
    else:
        # cumulative-average per-step reward = running task performance
        run_avg = np.cumsum(rew_seq) / (np.arange(T) + 1)
        ax1.plot(np.arange(T), rew_seq, color="#C44E52", lw=0.8, alpha=0.4, label="per-step reward")
        ax1.plot(np.arange(T), run_avg, color="#C44E52", lw=2.0, label="running-avg reward")
        ax1.set_ylabel("task reward")
        ax1.set_ylim(min(0, float(rew_seq.min())), max(1.0, float(rew_seq.max()) * 1.1))
        ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"NEXUS pixel hierarchy on {env_name} ({args.meta}, seed {seed}): "
                  f"skill activation vs. trajectory", fontsize=10)
    cmap = ListedColormap(palette)
    ax2.imshow(sel_skill[None], aspect="auto", cmap=cmap, vmin=-0.5, vmax=num_skills - 0.5,
               extent=[0, T, 0, 1], interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_xlabel("closed-loop step")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=palette[i], label=skill_names[i])
               for i in range(num_skills)]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55),
               ncol=num_skills, fontsize=8, frameon=False)
    fig.savefig(out / "skill_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- artifact 4: held-out imitation fidelity scatter (pooled over action dims) ----
    fig, ax = plt.subplots(figsize=(5, 4.6))
    r = float(np.corrcoef(ho_t, ho_p)[0, 1]) if len(ho_t) > 1 else float("nan")
    for i in range(num_skills):
        sel = ho_k == i
        if sel.any():
            ax.scatter(ho_t[sel], ho_p[sel], s=8, alpha=0.4, color=palette[i], label=skill_names[i])
    if len(ho_t):
        lim = [float(min(ho_t.min(), ho_p.min())), float(max(ho_t.max(), ho_p.max()))]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("teacher action")
    ax.set_ylabel("pixel-student action")
    ax.set_title(f"{env_name}: held-out imitation fidelity\n(all action dims, Pearson r = {r:.3f})")
    ax.legend(fontsize=8)
    fig.savefig(out / "action_tracking.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    perf = f"pixel mean reward/step {rew_seq.mean():.3f}"
    if is_cartpole:
        perf += (f", upright {float(np.mean((np.abs(angle_deg) < 14.3) & (np.abs(cart_pos) < 1.0))):.3f}")
    print(f"\n==== VIZ DONE ==== {perf} over {T} steps, held-out fidelity r={r:.3f}")
    print("wrote", out.resolve())
    for f in ("rollout_pixel.mp4", "rollout_pixel.gif", "observation_filmstrip.png",
              "skill_timeline.png", "action_tracking.png"):
        print("  -", f)


if __name__ == "__main__":
    main()
