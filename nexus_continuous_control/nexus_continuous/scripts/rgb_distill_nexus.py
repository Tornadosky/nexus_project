"""On-thesis RGB extension: distill the TRAINED NEXUS hierarchy to pixel skills.

This is the report-grade continuous-NEXUS + RGB experiment (CartpoleBalance, the
only vision-capable Playground task). The teacher is the *actual* trained NEXUS
hierarchy -- not a hand-coded controller:

  1. Train state NEXUS for one seed. The meta-policy variant is selectable via
     --meta: `nesy` (default, the flagship neuro-symbolic meta: learned meta-Q
     masked by hand-written skill preconditions), `neural` (unmasked learned
     meta-Q), or `symbolic` (rule-based selection). Skill actors are trained
     the same way in all cases.
  2. DAgger distillation (Ross 2011) into per-skill VisionSkillActors, rendering
     64x64 frames on the fly. Round 0 rolls out the TEACHER (plain behavior
     cloning); each subsequent --dagger-iters round rolls out the current PIXEL
     student and relabels the states it actually visits with the teacher action,
     aggregating into the dataset. This closes the covariate-shift gap that pure
     BC leaves (near-perfect open-loop imitation but drift in closed loop).
     Skills used < --min-samples keep the privileged teacher (pixel-fallback).
  3. Closed-loop eval: the UNCHANGED meta (on privileged state) selects the
     skill and the PIXEL student acts. Report state vs BC-pixel vs DAgger-pixel.
  4. Compare pixel-hierarchy vs state-hierarchy (the privileged upper bound),
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
    ap.add_argument("--teacher-steps", type=int, default=4000, help="round-0 teacher rollout steps (BC dataset)")
    ap.add_argument("--dagger-iters", type=int, default=0,
                    help="DAgger aggregation rounds after BC (0 = plain behavior cloning; "
                         "note: DAgger did not robustly beat BC on CartpoleBalance -- see results/rgb)")
    ap.add_argument("--dagger-steps", type=int, default=1500,
                    help="student-rollout steps added per DAgger round (teacher-relabelled)")
    ap.add_argument("--dagger-beta", type=float, default=0.5,
                    help="DAgger mixing base: execute teacher w.p. beta**iter (keeps the "
                         "rollout near-distribution so aggregated states stay useful)")
    ap.add_argument("--dagger-reset-every", type=int, default=50,
                    help="reset to upright every N steps during rollouts (0 = never); stops "
                         "a diverging student from flooding the set with fallen states")
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

        # ---- Stage 2+3: DAgger distillation (teacher-relabelled aggregation) ----
        eval_cfg = dict(cfg)
        eval_cfg["NORMALIZE_OBS"] = False   # raw obs; we apply frozen stats in norm_actor
        eval_cfg["NUM_ENVS"] = 1
        bundle = build_playground_env(eval_cfg)
        env, env_params = bundle.env, bundle.env_params
        alo, ahi = jnp.asarray(bundle.action_low), jnp.asarray(bundle.action_high)
        step_fn = jax.jit(env.step)

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

        def distill(X, Y, skills):
            """Retrain one VisionSkillActor per skill on the full aggregated dataset."""
            sp = []
            for k in range(num_skills):
                m = skills == k
                n = int(m.sum())
                if n < args.min_samples:
                    print(f"      skill {k}: {n} samples (<{args.min_samples}) -> teacher fallback")
                    sp.append(None)
                    continue
                bs = min(args.batch_size, n)
                xk, yk = jnp.asarray(X[m]), jnp.asarray(Y[m])
                p = vactor.init(jax.random.PRNGKey(k), xk[:1], empty(1))["params"]
                st = opt.init(p)
                last = float("nan")
                for _e in range(args.epochs):
                    perm = np.random.permutation(n)
                    for i in range(0, n, bs):  # >=1 non-empty batch
                        p, st, l = bc_step(p, st, xk[perm[i : i + bs]], yk[perm[i : i + bs]])
                        last = float(l)
                sp.append(p)
                print(f"      skill {k}: {n} samples (bs={bs}), MSE {last:.4f}")
            return sp

        def collect_rollout(sp, n_steps, key, beta):
            """Roll out the hierarchy for n_steps rendering 64x64 frames on the fly.
            The meta always selects the skill from privileged state. Execution mixes
            teacher/student: teacher w.p. `beta`, else the distilled pixel actor
            (DAgger-beta, Ross 2011) -- keeping the trajectory near-distribution so
            the aggregated states are small student perturbations, not garbage.
            Every visited state is labelled with the TEACHER action (expert
            relabelling). Periodic resets stop a diverging student from flooding the
            dataset with fallen states."""
            key, rk = jax.random.split(key)
            obs, state = env.reset(jax.random.split(rk, 1), env_params)
            buf, Xs, Ys, Ss = [], [], [], []
            data = mujoco.MjData(mj_model)
            with mujoco.Renderer(mj_model, height=64, width=64) as r:
                for t in range(n_steps):
                    if args.dagger_reset_every > 0 and t > 0 and t % args.dagger_reset_every == 0:
                        key, rk = jax.random.split(key)
                        obs, state = env.reset(jax.random.split(rk, 1), env_params)
                        buf = []
                    sd = _mjx_data(state)
                    data.qpos[:] = np.asarray(sd.qpos[0])
                    data.qvel[:] = np.asarray(sd.qvel[0])
                    mujoco.mj_forward(mj_model, data)
                    r.update_scene(data, camera=cam)
                    buf.append(r.render().mean(-1).astype(np.float32) / 255.0 - 0.5)
                    buf = buf[-3:]
                    skill_t, teacher_act = teacher_step(obs)
                    k = int(np.asarray(skill_t[0]))
                    act = teacher_act
                    if len(buf) == 3:
                        Xs.append(np.stack(buf, -1))            # [64,64,3]
                        Ys.append(np.asarray(teacher_act[0]))   # teacher (expert) label
                        Ss.append(k)
                        if sp[k] is not None:
                            key, kb = jax.random.split(key)
                            if float(jax.random.uniform(kb)) >= beta:   # else keep teacher
                                stack = jnp.asarray(np.stack(buf, -1)[None])
                                act = jnp.asarray(np.asarray(
                                    vactor.apply({"params": sp[k]}, stack, empty(1))))
                    act = jnp.clip(act, alo, ahi)
                    key, ks = jax.random.split(key)
                    obs, state, _r, _d, _i = step_fn(jax.random.split(ks, 1), state, act, env_params)
            return (np.asarray(Xs, np.float32), np.asarray(Ys, np.float32), np.asarray(Ss, np.int32))

        print(f"[2/3] DAgger distillation ({args.dagger_iters} iters after BC, "
              f"beta={args.dagger_beta}, reset={args.dagger_reset_every})...")
        agg_X, agg_Y, agg_S = [], [], []
        skill_params = [None] * num_skills
        skill_params_bc = None
        drng = jax.random.PRNGKey(1000 + seed)
        for it in range(args.dagger_iters + 1):
            if it == 0:
                n_steps, beta = args.teacher_steps, 1.0            # pure teacher (BC)
            else:
                n_steps, beta = args.dagger_steps, args.dagger_beta ** it
            drng, ck = jax.random.split(drng)
            Xr, Yr, Sr = collect_rollout(skill_params, n_steps, ck, beta)
            agg_X.append(Xr); agg_Y.append(Yr); agg_S.append(Sr)
            X = np.concatenate(agg_X); Y = np.concatenate(agg_Y); skills = np.concatenate(agg_S)
            tag = "BC: teacher rollout" if it == 0 else f"DAgger {it}: beta={beta:.2f}"
            print(f"    [{tag}] +{len(Xr)} new, {len(X)} total; "
                  f"hist={np.bincount(skills, minlength=num_skills)}")
            skill_params = distill(X, Y, skills)
            if it == 0:
                skill_params_bc = list(skill_params)   # BC baseline for before/after

        # ---- Stage 4: closed-loop eval (meta on state, PIXEL student acts) ----
        print("[4] closed-loop eval: state vs BC-pixel vs DAgger-pixel...")

        def closed_loop(sp, use_pixels: bool):
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
                if use_pixels and sp[k] is not None:
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
                            vactor.apply({"params": sp[k]}, stack, empty(1))))
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

        state_success, _ = closed_loop(skill_params, use_pixels=False)
        bc_success, _ = closed_loop(skill_params_bc, use_pixels=True)
        pixel_success, pixel_fallback = closed_loop(skill_params, use_pixels=True)
        distilled = [k for k in range(num_skills) if skill_params[k] is not None]
        print(f"    seed {seed}: state {state_success:.3f} | BC-pixel {bc_success:.3f} "
              f"| DAgger-pixel {pixel_success:.3f} | fallback {pixel_fallback:.2f} | skills {distilled}")
        per_seed.append({
            "seed": seed,
            "state_success": state_success,
            "bc_pixel_success": bc_success,
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
    bc = np.array([r["bc_pixel_success"] for r in per_seed])
    ps = np.array([r["pixel_success"] for r in per_seed])
    fbf = np.array([r["pixel_fallback_fraction"] for r in per_seed])
    summary = {
        "task": "CartpoleBalance",
        "meta_policy": args.meta,
        "dagger_iters": args.dagger_iters,
        "seeds": seeds,
        "state_hierarchy_success_mean": float(ss.mean()),
        "state_hierarchy_success_std": float(ss.std()),
        "bc_pixel_success_mean": float(bc.mean()),
        "bc_pixel_success_std": float(bc.std()),
        "pixel_hierarchy_success_mean": float(ps.mean()),
        "pixel_hierarchy_success_std": float(ps.std()),
        "retention_fraction": float(ps.mean() / max(ss.mean(), 1e-6)),
        "bc_retention_fraction": float(bc.mean() / max(ss.mean(), 1e-6)),
        "pixel_fallback_fraction_mean": float(fbf.mean()),
        "per_seed": per_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig = plt.figure(figsize=(6, 4))
    x = [0, 1, 2]
    plt.bar(x, [ss.mean(), bc.mean(), ps.mean()], yerr=[ss.std(), bc.std(), ps.std()], capsize=6,
            color=["#4C72B0", "#C44E52", "#DD8452"])
    for xi, v in zip(x, [ss.mean(), bc.mean(), ps.mean()]):
        plt.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(x, ["state\n(privileged)", "pixel BC", f"pixel DAgger\n({args.dagger_iters} it)"])
    plt.ylabel("closed-loop success rate")
    plt.title(f"NEXUS ({args.meta}) on CartpoleBalance ({len(seeds)} seeds)")
    plt.ylim(0, 1.1)
    fig.savefig(out / "state_vs_pixel.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print("wrote", out.resolve())


if __name__ == "__main__":
    main()
