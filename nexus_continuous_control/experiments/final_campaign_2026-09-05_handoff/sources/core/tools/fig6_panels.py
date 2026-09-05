"""Paper Fig. 6 analogue: ambiguous states, with the masked meta-Q values that resolved them.

The paper's Fig. 6 shows Seaquest frames where several rules fire at once, next to the learned
meta-Q value of each admissible skill — interpretability at the rule level, flexibility at the
value level. This produces the continuous-control version: roll one greedy NeSy episode, find
the steps where the hand-written mask leaves MORE than one skill admissible and the top-two
admissible Q-values are closest (the genuinely ambiguous decisions), and render each as a
two-pane panel:

  left   the MuJoCo frame at that step
  right  a horizontal bar per skill: meta-Q value, masked-out skills greyed and hatched,
         the selected skill outlined; underneath, the mask bit per skill and the semantic
         driver values (command_yaw etc.) the rule read at that step

Eval-only, CPU + osmesa; selection matches robustness_eval / render_rollout exactly.

Usage
-----
    MUJOCO_GL=osmesa JAX_PLATFORMS=cpu python tools/fig6_panels.py \
        --checkpoint runs/verify/go1_joystick_nesy_v2_s0.pkl --panels 3 --out runs/fig6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_rollout import STRIP_DRIVERS, _unwrap_data  # noqa: E402
from robustness_eval import (  # noqa: E402
    _build_networks,
    _load_checkpoint,
    _make_normalizer,
    _restore_params,
)

from nexus_continuous.envs.playground_adapter import (  # noqa: E402
    build_playground_env,
    get_actor_obs,
    get_policy_obs,
)
from nexus_continuous.policies.registry import load_policy_module  # noqa: E402

SKILL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a NeSy checkpoint")
    ap.add_argument("--panels", type=int, default=3)
    ap.add_argument("--min-gap", type=int, default=60,
                    help="minimum steps between two selected panels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=368)
    ap.add_argument("--out", default="runs/fig6")
    args = ap.parse_args(argv)

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    ck = _load_checkpoint(args.checkpoint)
    cfg = dict(ck["config"])
    rs = ck["runner_state"]
    stats = ck.get("normalization_stats")
    meta_type = str(cfg.get("META_POLICY_TYPE", "nesy")).lower()
    if meta_type != "nesy":
        print(f"checkpoint is {meta_type}; Fig. 6 panels need a nesy checkpoint (mask + meta-Q)")
        return 1

    policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES",
                               tuple(f"skill_{i}" for i in range(num_skills))))
    policy_key = str(cfg.get("POLICY", cfg["ENV_NAME"]))
    driver_spec = STRIP_DRIVERS.get(policy_key, [])
    diag_fn = getattr(policy_module, "diagnostics", None)

    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NORMALIZE_REWARD"] = False
    eval_cfg["NUM_ENVS"] = 1
    eval_cfg["USE_RGB"] = False
    bundle = build_playground_env(eval_cfg)
    env, env_params = bundle.env, bundle.env_params
    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    steps = int(args.steps or bundle.episode_length)

    actor, meta_q = _build_networks(cfg, num_skills, bundle.action_dim,
                                    (action_high - action_low) / 2.0,
                                    (action_high + action_low) / 2.0)
    raw_obs, env_state = env.reset(jax.random.split(jax.random.PRNGKey(args.seed), 1), env_params)
    dummy_actor = get_actor_obs(raw_obs)
    fresh_actor = jax.vmap(lambda k: actor.init(k, dummy_actor)["params"])(
        jax.random.split(jax.random.PRNGKey(0), num_skills))
    actor_params = _restore_params(fresh_actor, rs["0"]["actor"]["params"])
    fresh_meta = meta_q.init(jax.random.PRNGKey(1), dummy_actor)["params"]
    meta_params = _restore_params(fresh_meta, rs["0"]["meta"]["params"])
    normalize_obs = _make_normalizer(stats, cfg.get("NORMALIZE_OBS", True))

    @jax.jit
    def decide(obs):
        obs_actor = get_actor_obs(obs)
        policy_obs = get_policy_obs(obs)
        acts = jax.vmap(lambda p: actor.apply({"params": p}, obs_actor))(actor_params)
        acts = jnp.swapaxes(acts, 0, 1)  # [1, N, A]
        q = meta_q.apply({"params": meta_params}, obs_actor)  # [1, N]
        mask = jnp.asarray(policy_module.skill_mask(policy_obs)).astype(bool)
        qm = jnp.where(mask, q, -1.0e9)
        skill = jnp.argmax(qm, axis=-1).astype(jnp.int32)
        action = jnp.clip(acts[jnp.arange(1), skill], action_low, action_high)
        return action, skill, q, mask

    # ---- pass 1: roll the episode, record q/mask/qpos/qvel at every step --------------- #
    rec = []
    rng = jax.random.PRNGKey(args.seed + 1)
    for t in range(steps):
        obs = normalize_obs(raw_obs)
        action, skill, q, mask = decide(obs)
        data = _unwrap_data(env_state)
        drivers = {}
        if diag_fn is not None and driver_spec:
            diag = diag_fn(get_policy_obs(obs), get_policy_obs(obs),
                           action, jnp.zeros((1,)), jnp.zeros((1,), bool), None)
            for key, _thr, _txt in driver_spec:
                if key in diag:
                    drivers[key] = float(np.asarray(diag[key]).reshape(-1)[0])
        rec.append({
            "t": t,
            "q": np.asarray(q)[0].astype(float),
            "mask": np.asarray(mask)[0].astype(bool),
            "skill": int(np.asarray(skill)[0]),
            "qpos": np.asarray(data.qpos)[0].copy(),
            "qvel": np.asarray(data.qvel)[0].copy(),
            "drivers": drivers,
        })
        rng, k = jax.random.split(rng)
        raw_obs, env_state, reward, done, _ = env.step(
            jax.random.split(k, 1), env_state, action, env_params)
        if bool(np.asarray(done)[0]):
            break

    # ---- pick the most ambiguous multi-admissible steps, spaced apart ------------------ #
    def ambiguity(r):
        avail = r["q"][r["mask"]]
        if avail.size < 2:
            return None
        top2 = np.sort(avail)[-2:]
        return float(top2[1] - top2[0])  # small gap == ambiguous

    candidates = sorted(
        (r for r in rec if ambiguity(r) is not None),
        key=lambda r: ambiguity(r),
    )
    chosen: list[dict] = []
    for r in candidates:
        if len(chosen) >= args.panels:
            break
        if all(abs(r["t"] - c["t"]) >= args.min_gap for c in chosen):
            chosen.append(r)
    chosen.sort(key=lambda r: r["t"])
    if not chosen:
        print("no multi-admissible states found — nothing to draw")
        return 1

    # ---- pass 2: render the chosen frames from the recorded qpos/qvel ------------------ #
    mj_model = getattr(bundle, "mj_model", None)
    if mj_model is None:
        from mujoco_playground import registry  # noqa: PLC0415

        probe_cfg = registry.get_default_config(cfg["ENV_NAME"])
        probe_cfg.impl = cfg.get("PLAYGROUND_IMPL", "jax")
        probe_env = registry.load(cfg["ENV_NAME"], probe_cfg)
        mj_model = getattr(probe_env, "mj_model", None) or probe_env._mj_model
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    mj_data = mujoco.MjData(mj_model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem

    for r in chosen:
        mj_data.qpos[:] = r["qpos"]
        mj_data.qvel[:] = r["qvel"]
        mujoco.mj_forward(mj_model, mj_data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = min(1, mj_model.nbody - 1)
        cam.distance = float(mj_model.stat.extent) * 3.0
        cam.elevation, cam.azimuth = -15.0, 120.0
        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()

        fig = plt.figure(figsize=(9.6, 3.9), dpi=150)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.25)
        axf = fig.add_subplot(gs[0])
        axf.imshow(frame)
        axf.set_axis_off()
        axf.set_title(f"{cfg['ENV_NAME']} — step {r['t']}", fontsize=9, loc="left")

        axq = fig.add_subplot(gs[1])
        q, mask, sel = r["q"], r["mask"], r["skill"]
        ys = np.arange(num_skills)[::-1]
        for i in range(num_skills):
            color = SKILL_COLORS[i % len(SKILL_COLORS)]
            admissible = bool(mask[i])
            axq.barh(
                ys[i], q[i],
                color=color if admissible else "#c9ccd4",
                alpha=1.0 if admissible else 0.55,
                hatch=None if admissible else "///",
                edgecolor="#151A23" if i == sel else "none",
                linewidth=1.6 if i == sel else 0.0,
                height=0.62,
            )
            axq.text(0.01, ys[i],
                     f" {skill_names[i]}  [{'✓' if admissible else '✗ masked'}]"
                     f"{'  ← selected' if i == sel else ''}",
                     transform=axq.get_yaxis_transform(),
                     va="center", fontsize=8,
                     color="#151A23" if admissible else "#8a8f99")
        axq.set_yticks([])
        axq.set_xlabel("meta-Q value", fontsize=8)
        axq.tick_params(labelsize=7)
        axq.grid(axis="x", alpha=0.15, lw=0.5)
        drv = "   ".join(f"{k.split('/')[-1]}={v:+.3f}" for k, v in r["drivers"].items())
        axq.set_title(f"rule inputs:  {drv}", fontsize=7.5, loc="left", color="#3D4756")

        path = out_dir / f"{stem}_t{r['t']:04d}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        avail_names = [skill_names[i] for i in range(num_skills) if mask[i]]
        print(f"wrote {path}  (admissible: {', '.join(avail_names)}; "
              f"selected {skill_names[sel]}; q-gap {ambiguity(r):.3f})")

    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
