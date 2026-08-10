"""On-thesis RGB extension: distill the TRAINED NEXUS hierarchy to pixel skills.

This is the report-grade continuous-NEXUS + RGB experiment (CartpoleBalance, the
only vision-capable Playground task). The teacher is the *actual* trained NEXUS
hierarchy -- not a hand-coded controller:

  1. Train state NEXUS for one seed. The meta-policy variant is selectable via
     --meta: `nesy` (default, the flagship neuro-symbolic meta: learned meta-Q
     masked by hand-written skill preconditions), `neural` (unmasked learned
     meta-Q), or `symbolic` (rule-based selection). Skill actors are trained
     the same way in all cases.
  2. Roll the trained hierarchy out; record (rendered 64x64 frame, selected
     skill, that skill's action) -- on-policy, so pixel students see the states
     they will be deployed on.
  3. Behavior-clone each real NEXUS skill (with >= --min-samples uses) into a
     VisionSkillActor. Rarely-used skills keep the privileged teacher and are
     reported as a pixel-fallback fraction.
  4. Closed-loop eval: the UNCHANGED meta (on privileged state) selects the
     skill and the PIXEL student acts. Measure per-step task success.
  5. Compare pixel-hierarchy vs state-hierarchy (the privileged upper bound),
     aggregated over seeds (mean +/- std), + skill-activation histograms.

Meta/critic/meta-Q always stay on privileged state; only the skill actor moves
to pixels -- the asymmetric actor-critic design (Pinto 2017) with
Learning-by-Cheating distillation (Chen 2020). Reuses the tested trainer, env
adapter, and VisionSkillActor.

Run headless on a GPU (student pool / SSH). CartpoleBalance renders offscreen via
software EGL (llvmpipe) when no GPU display device is accessible:
    MUJOCO_GL=egl python -m nexus_continuous.scripts.rgb_distill_nexus \\
        --config configs/cartpole_balance_symbolic.yaml --meta nesy --seeds 0,1,2 \\
        --out runs/rgb_nexus
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _mjx_data(state):
    """Dig through the env wrappers (LogVec/Clip/Normalize) to the mjx State's .data."""
    for _ in range(8):
        if hasattr(state, "data"):
            return state.data
        state = getattr(state, "env_state", None)
        if state is None:
            break
    raise AttributeError("could not locate .data on the wrapped env state")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cartpole_balance_nesy.yaml",
                    help="teacher config; use the variant matching --meta "
                         "(nesy/neural/symbolic) for correctly-tuned hyperparameters")
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--out", default="runs/rgb_nexus")
    ap.add_argument("--teacher-steps", type=int, default=4000, help="teacher rollout steps for the distillation dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-steps", type=int, default=250, help="closed-loop eval horizon")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--rtpt-initials", default="RB")
    ap.add_argument("--meta", default="nesy", choices=["nesy", "neural", "symbolic"],
                    help="NEXUS meta-policy used as the teacher (nesy = flagship neuro-symbolic)")
    ap.add_argument("--min-samples", type=int, default=64,
                    help="min per-skill samples to distill a pixel actor; below this the "
                         "skill keeps the privileged teacher (reported as pixel-fallback)")
    args = ap.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("jax", jax.__version__, jax.devices(), "| seeds", seeds)

    rtpt = None
    try:
        from rtpt import RTPT

        rtpt = RTPT(args.rtpt_initials, "nexus-rgb-distill", max_iterations=len(seeds))
        rtpt.start()
    except Exception as exc:  # pragma: no cover
        print(f"[rtpt] not active ({exc!r}); install `rtpt` for lab etiquette")

    # ---- offline renderer (CartpoleBalance, state sim, impl='jax' => no warp) ----
    rcfg = registry.get_default_config("CartpoleBalance")
    rcfg.impl = "jax"
    render_env = registry.load("CartpoleBalance", rcfg)
    mj_model = getattr(render_env, "mj_model", None) or getattr(render_env, "_mj_model")
    cam = 0
    try:
        _d = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, _d)
        with mujoco.Renderer(mj_model, height=64, width=64) as _r:
            _r.update_scene(_d, camera=cam)
    except Exception:
        cam = -1

    def render_qpos_qvel(qpos_seq, qvel_seq):
        """qpos_seq/qvel_seq: [T, nq]/[T, nv] numpy -> frames [T,64,64,3] uint8."""
        data = mujoco.MjData(mj_model)
        frames = np.empty((len(qpos_seq), 64, 64, 3), np.uint8)
        with mujoco.Renderer(mj_model, height=64, width=64) as r:
            for t in range(len(qpos_seq)):
                data.qpos[:] = np.asarray(qpos_seq[t])
                data.qvel[:] = np.asarray(qvel_seq[t])
                mujoco.mj_forward(mj_model, data)
                r.update_scene(data, camera=cam)
                frames[t] = r.render()
        return frames

    def gray_stack(frames):
        """[T,64,64,3] uint8 -> stacks [T-2,64,64,3] float centered, aligned targets idx."""
        g = frames.mean(-1).astype(np.float32) / 255.0 - 0.5
        return np.stack([g[i - 2 : i + 1] for i in range(2, len(g))], 0).transpose(0, 2, 3, 1)

    empty = lambda b: jnp.zeros((b, 0), jnp.float32)
    per_seed = []

    for seed in seeds:
        print(f"\n================= seed {seed} =================")
        cfg = load_config(args.config)
        cfg["SEED"] = seed
        cfg["META_POLICY_TYPE"] = args.meta   # nesy/neural = trained meta-Q; symbolic = rule
        cfg["NUM_SEEDS"] = 1

        # ---- Stage 1: train the state NEXUS teacher (reuses the tested trainer) ----
        print(f"[1] training state NEXUS teacher ({args.meta} meta)...")
        output = run_training(cfg)
        train_state = output.runner_state[0]
        actor_params = train_state.actor.params           # vmapped over skills: [N, ...]
        stats = output.normalization_stats
        normalize_obs = bool(cfg.get("NORMALIZE_OBS", True))
        policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
        num_skills = int(policy_module.NUM_SKILLS)
        teacher_eval = {k: float(np.asarray(v)) for k, v in output.eval_metrics.items()
                        if np.asarray(v).ndim == 0}
        print(f"    teacher trained. skills={num_skills} eval={teacher_eval.get('primary_success_rate')}")

        actor = SkillActor(
            action_dim=1,
            action_scale=jnp.ones(1),
            action_bias=jnp.zeros(1),
            hidden_sizes=tuple(cfg.get("ACTOR_HIDDEN_SIZES", (256, 256))),
            activation=cfg.get("ACTIVATION", "relu"),
            norm_type=cfg.get("NORM_TYPE", "layer_norm"),
        )

        # Trained meta-Q (nesy/neural). It stays on privileged state; only the
        # skill actor is later distilled to pixels (the asymmetric AC design).
        meta_q = MetaQ(
            num_skills=num_skills,
            hidden_sizes=tuple(cfg.get("META_HIDDEN_SIZES", (256, 256))),
            activation=cfg.get("ACTIVATION", "relu"),
            norm_type=cfg.get("NORM_TYPE", "layer_norm"),
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
            """Greedy NEXUS hierarchy: the meta selects a skill from privileged
            state, that skill's actor acts. atleast_1d/2d restore the batch axis
            the policy helpers' jnp.squeeze drops at NUM_ENVS=1 (mirrors the
            trainer's masked-argmax selection in _select_action)."""
            oa = norm_actor(obs)                                              # normalized actor obs [E,D]
            all_a = jax.vmap(lambda p: actor.apply({"params": p}, oa))(actor_params)  # [N,E,A]
            pol = get_policy_obs(obs)                                         # raw obs for masks/rules
            if args.meta == "symbolic":
                skill = jnp.atleast_1d(
                    jnp.asarray(policy_module.symbolic_meta_policy(pol), jnp.int32))  # [E]
            else:
                qm = jnp.atleast_2d(meta_q.apply({"params": meta_params}, oa))        # [E,N]
                if args.meta == "nesy":
                    mask = jnp.atleast_2d(jnp.asarray(policy_module.skill_mask(pol), bool))
                    any_valid = jnp.any(mask, axis=-1, keepdims=True)
                    safe = jnp.where(any_valid, mask, jnp.ones_like(mask))
                    qm = jnp.where(safe, qm, -1.0e9)
                skill = jnp.argmax(qm, axis=-1).astype(jnp.int32)            # [E]
            e = jnp.arange(all_a.shape[1])
            act = all_a[skill, e]  # [E,A]
            return skill, act

        # ---- Stage 2: teacher rollout + record (qpos,qvel,skill,action) ----
        print("[2] rolling out teacher, collecting data...")
        eval_cfg = dict(cfg)
        eval_cfg["NORMALIZE_OBS"] = False   # raw obs; we apply frozen stats in norm_actor
        eval_cfg["NUM_ENVS"] = 1
        bundle = build_playground_env(eval_cfg)
        env, env_params = bundle.env, bundle.env_params
        alo, ahi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
        step_fn = jax.jit(env.step)

        rng = jax.random.PRNGKey(1000 + seed)
        rng, rr = jax.random.split(rng)
        obs, state = env.reset(jax.random.split(rr, 1), env_params)
        qpos_seq, qvel_seq, skill_seq, act_seq = [], [], [], []
        for _t in range(args.teacher_steps):
            skill, act = teacher_step(obs)
            act = jnp.clip(act, alo, ahi)
            sd = _mjx_data(state)
            qpos_seq.append(np.asarray(sd.qpos[0]))
            qvel_seq.append(np.asarray(sd.qvel[0]))
            skill_seq.append(int(np.asarray(skill[0])))
            act_seq.append(np.asarray(act[0]))
            rng, rs = jax.random.split(rng)
            obs, state, _r, _d, _i = step_fn(jax.random.split(rs, 1), state, act, env_params)
        frames = render_qpos_qvel(qpos_seq, qvel_seq)
        X = gray_stack(frames)                       # [T-2,64,64,3]
        skills = np.asarray(skill_seq[2:], np.int32) # align to X
        Y = np.asarray(act_seq[2:], np.float32)
        print(f"    collected {len(X)} samples; skill hist={np.bincount(skills, minlength=num_skills)}")

        # ---- Stage 3: distill each real skill into a VisionSkillActor ----
        print("[3] distilling per-skill vision actors...")
        vactor = VisionSkillActor(action_dim=1, action_scale=jnp.ones(1), action_bias=jnp.zeros(1),
                                  hidden_sizes=(128, 128), embedding_dim=args.embed_dim)
        opt = optax.adam(3e-4)

        def loss_fn(p, x, y):
            return jnp.mean((vactor.apply({"params": p}, x, empty(x.shape[0])) - y) ** 2)

        @jax.jit
        def bc_step(p, st, x, y):
            l, g = jax.value_and_grad(loss_fn)(p, x, y)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l

        skill_params = []
        for k in range(num_skills):
            m = skills == k
            n = int(m.sum())
            if n < args.min_samples:
                print(f"    skill {k}: {n} samples (<{args.min_samples}) -> kept privileged teacher fallback")
                skill_params.append(None)
                continue
            bs = min(args.batch_size, n)
            xk, yk = jnp.asarray(X[m]), jnp.asarray(Y[m])
            p = vactor.init(jax.random.PRNGKey(k), xk[:1], empty(1))["params"]
            st = opt.init(p)
            last = float("nan")
            for _e in range(args.epochs):
                perm = np.random.permutation(n)
                for i in range(0, n, bs):  # >=1 non-empty batch
                    idx = perm[i : i + bs]
                    p, st, l = bc_step(p, st, xk[idx], yk[idx])
                    last = float(l)
            skill_params.append(p)
            print(f"    skill {k}: trained on {n} samples (bs={bs}), final MSE {last:.4f}")

        # ---- Stage 4: closed-loop eval (symbolic meta on state, PIXEL student acts) ----
        print("[4] closed-loop eval: pixel hierarchy vs state hierarchy...")

        def closed_loop(use_pixels: bool):
            rng2 = jax.random.PRNGKey(5000 + seed)
            rng2, rr2 = jax.random.split(rng2)
            obs, state = env.reset(jax.random.split(rr2, 1), env_params)
            buf, up, fb = [], 0, 0
            data = mujoco.MjData(mj_model)
            renderer = mujoco.Renderer(mj_model, height=64, width=64) if use_pixels else None
            for _t in range(args.eval_steps):
                skill_t, teacher_act = teacher_step(obs)
                k = int(np.asarray(skill_t[0]))
                pixel_used = False
                if use_pixels and skill_params[k] is not None:
                    sd = _mjx_data(state)
                    data.qpos[:] = np.asarray(sd.qpos[0])
                    data.qvel[:] = np.asarray(sd.qvel[0])
                    mujoco.mj_forward(mj_model, data)
                    renderer.update_scene(data, camera=cam)
                    buf.append(renderer.render().mean(-1).astype(np.float32) / 255.0 - 0.5)
                    buf = buf[-3:]
                    if len(buf) == 3:
                        stack = jnp.asarray(np.stack(buf, -1)[None])
                        act = jnp.asarray(np.asarray(
                            vactor.apply({"params": skill_params[k]}, stack, empty(1))))
                        pixel_used = True
                    else:
                        act = teacher_act
                else:
                    act = teacher_act
                if use_pixels and not pixel_used:
                    fb += 1  # privileged-teacher fallback (undistilled skill or 2-frame warmup)
                act = jnp.clip(act, alo, ahi)
                # success: pole upright & cart centered (matches task_metrics)
                sd2 = _mjx_data(state)
                cart = float(np.asarray(sd2.qpos[0, 0]))
                angle = float(np.asarray(sd2.qpos[0, 1]))
                up += int((abs(np.arctan2(np.sin(angle), np.cos(angle))) < 0.25) and (abs(cart) < 1.0))
                rng2, rs2 = jax.random.split(rng2)
                obs, state, _r, _d, _i = step_fn(jax.random.split(rs2, 1), state, act, env_params)
            if renderer is not None:
                renderer.close()
            return up / args.eval_steps, fb / args.eval_steps

        state_success, _ = closed_loop(use_pixels=False)
        pixel_success, pixel_fallback = closed_loop(use_pixels=True)
        distilled = [k for k in range(num_skills) if skill_params[k] is not None]
        print(f"    seed {seed}: state {state_success:.3f} | pixel {pixel_success:.3f} "
              f"| pixel-fallback {pixel_fallback:.2f} | distilled skills {distilled}")
        per_seed.append({
            "seed": seed,
            "state_success": state_success,
            "pixel_success": pixel_success,
            "pixel_fallback_fraction": pixel_fallback,
            "distilled_skills": distilled,
            "teacher_eval_primary_success": teacher_eval.get("primary_success_rate"),
            "skill_histogram": np.bincount(skills, minlength=num_skills).tolist(),
        })
        if rtpt is not None:
            rtpt.step()

    # ---- aggregate + report ----
    ss = np.array([r["state_success"] for r in per_seed])
    ps = np.array([r["pixel_success"] for r in per_seed])
    fbf = np.array([r["pixel_fallback_fraction"] for r in per_seed])
    summary = {
        "task": "CartpoleBalance",
        "meta_policy": args.meta,
        "seeds": seeds,
        "state_hierarchy_success_mean": float(ss.mean()),
        "state_hierarchy_success_std": float(ss.std()),
        "pixel_hierarchy_success_mean": float(ps.mean()),
        "pixel_hierarchy_success_std": float(ps.std()),
        "retention_fraction": float(ps.mean() / max(ss.mean(), 1e-6)),
        "pixel_fallback_fraction_mean": float(fbf.mean()),
        "per_seed": per_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig = plt.figure(figsize=(5, 4))
    x = [0, 1]
    plt.bar(x, [ss.mean(), ps.mean()], yerr=[ss.std(), ps.std()], capsize=6,
            color=["#4C72B0", "#DD8452"])
    plt.xticks(x, ["state hierarchy\n(privileged)", "pixel hierarchy\n(distilled)"])
    plt.ylabel("closed-loop success rate")
    plt.title(f"NEXUS ({args.meta}) on CartpoleBalance ({len(seeds)} seeds)")
    plt.ylim(0, 1)
    fig.savefig(out / "state_vs_pixel.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
