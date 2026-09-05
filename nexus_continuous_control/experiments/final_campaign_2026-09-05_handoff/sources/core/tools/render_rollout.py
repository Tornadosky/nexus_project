"""Render a trained NEXUS checkpoint to video, with the active skill burned into each frame.

Why this exists
---------------
Nothing in this repo could render a state-based agent. ``rgb_distill_nexus.py`` renders
CartpoleBalance as part of the RGB work, but there was no way to *watch* a trained hierarchy
act on any of the six environments — so every behavioural claim ("stands but does not walk",
"lifts the cube") rested on scalar metrics alone. The reliability campaign showed how far
that can go wrong: a walker with return 931 and zero net displacement looks excellent in a
table and obviously broken on video.

What it produces
----------------
An MP4 of a greedy (no-exploration) episode. Each frame is annotated with the step index, the
running return, and **which skill is active** — the meta-policy's decision made visible, which
is the paper's Q2 interpretability claim in its most direct form. ``--strip`` additionally
writes a skill-activation timeline PNG next to the video.

Selection, normalization and network reconstruction are imported from ``robustness_eval`` so
that what you watch is exactly what the deterministic evaluation scores — if these two ever
disagree, one of them is lying.

Usage
-----
    MUJOCO_GL=osmesa JAX_PLATFORMS=cpu python tools/render_rollout.py \\
        --checkpoint runs/verify/walker_walk_nesy_s0.pkl \\
        --out runs/videos/walker_nesy.mp4 --strip

Render on CPU and overlap it with GPU training — it costs nothing on the accelerator. This
must run locally: Viper compute nodes have no ``/dev/dri`` and cannot render at all.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# MuJoCo picks its GL backend at import time, so this has to happen before mujoco loads.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustness_eval import (  # noqa: E402
    _build_networks,
    _load_checkpoint,
    _make_normalizer,
    _make_selector,
    init_hold_state,
    _restore_params,
)

from nexus_continuous.envs.playground_adapter import (  # noqa: E402
    build_playground_env,
    get_actor_obs,
    get_policy_obs,
)
from nexus_continuous.policies.registry import load_policy_module
from nexus_continuous.policies import hopper_hop as _HOP
import math as _math  # noqa: E402


# --------------------------------------------------------------------------- #
# frame annotation
# --------------------------------------------------------------------------- #

# Distinct, colour-blind-safe hues; index i is skill i. Deliberately the same order as the
# activation strip so the video and the timeline read as one artifact.
SKILL_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
]


def _annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Draw a small caption block in the top-left corner of a frame."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    pad, lh = 6, 14
    box_h = pad * 2 + lh * len(lines)
    box_w = max(int(img.width * 0.55), 190)
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 140))
    for i, (text, color) in enumerate(lines):
        draw.text((pad, pad + i * lh), text, fill=color)
    return np.asarray(img)


# Which semantic signals DRIVE each hand-written rule, with the rule's own threshold — so the
# activation strip answers "why did the meta hold this skill" instead of only "that it did".
# Keys are diagnostics() keys; thresholds are transcribed from the rules in the policy modules
# (also documented in README.md). Without this, Go1's strip shows `turn` held for 290 straight
# steps and the cause cannot be read off the plot.
STRIP_DRIVERS: dict[str, list[tuple[str, float, str]]] = {
    "go1_joystick": [
        ("go1/command_yaw", 0.15, "|command_yaw| > 0.15 -> turn"),
        ("go1/command_xy_norm", 0.10, "|command_xy| > 0.10 -> track_velocity"),
        # 0.28, not 0.22: the recover mask in policies/go1_joystick.py:97 is
        # `height < 0.28 | |roll| > 0.25 | |pitch| > 0.25`. 0.22 is the *fallen* threshold
        # (line 89/117), a different predicate — drawing it here made the overlay contradict
        # the rule it exists to explain, showing a comfortable margin where the real rule sits
        # ~2 cm under nominal standing height.
        ("go1/base_height", 0.28, "height < 0.28 -> recover"),
    ],
    "hopper_hop": [
        # Synced to the LIVE rule in policies/hopper_hop.py:105-106, which is
        # `height < ENV_STAND_HEIGHT (0.6) | cos(pitch) < UPRIGHT_COS (0.7)`.
        # The overlay previously drew 0.9 and |pitch| > 0.45 — the PRE-AUDIT values that the
        # comment at hopper_hop.py:101-104 records as the cause of one-skill degeneracy
        # ("hop cycles dip below 0.9 every bounce"). Two of this figure's three threshold lines
        # were therefore explaining a rule the code no longer runs, in the published strip.
        # Same defect class as the go1 entry above (0.22 vs 0.28), fixed there and missed here.
        ("hopper/height", _HOP.ENV_STAND_HEIGHT,
         f"height < {_HOP.ENV_STAND_HEIGHT} -> stand_recover"),
        ("hopper/forward_velocity", _HOP.TARGET_HOP_SPEED,
         f"v_x < {_HOP.TARGET_HOP_SPEED} -> hop_forward"),
        # cos(pitch) < 0.7  <=>  |pitch| > arccos(0.7) = 0.795 rad.
        ("hopper/pitch", _math.acos(_HOP.UPRIGHT_COS),
         f"|pitch| > {_math.acos(_HOP.UPRIGHT_COS):.3f} (cos < {_HOP.UPRIGHT_COS}) -> stand_recover"),
    ],
    "cartpole_balance": [
        ("cartpole/pole_angle", 0.20, "|pole_angle| > 0.20 -> recover_balance"),
        ("cartpole/cart_position", 0.35, "|cart_position| > 0.35 -> center_cart"),
    ],
    "walker_walk": [
        ("walker/height", 0.85, "height < 0.85 -> stand_recover"),
        ("walker/forward_velocity", 1.2, "v_x < 1.2 -> walk_forward"),
        ("walker/pitch", 0.45, "|pitch| > 0.45 -> stand_recover"),
    ],
    "cheetah_run": [
        ("cheetah/torso_pitch", 0.55, "|pitch| > 0.55 -> stabilize_posture"),
        ("cheetah/forward_velocity", 6.0, "v_x > 6 -> energy_efficient_run"),
    ],
    "panda_pick_cube": [
        ("panda/dist_tcp_cube", 0.06, "dist > 0.06 -> reach_cube"),
        ("panda/cube_height", 0.12, "height < 0.12 -> lift_cube"),
    ],
}


def _activation_strip(
    skills: np.ndarray,
    names: list[str],
    path: Path,
    title: str,
    drivers: list[tuple[str, str, np.ndarray, float]] | None = None,
) -> None:
    """Timeline of which skill was active at each step.

    `drivers` — optional list of (key, rule text, per-step series, rule threshold) — adds one
    thin panel per signal under the band, with the threshold as a dashed line, so the reader can
    see the semantic cause of every hold/switch directly above the value that triggered it.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = len(names)
    colors = np.array([SKILL_COLORS[i % len(SKILL_COLORS)] for i in range(n)]) / 255.0
    band = colors[np.clip(skills, 0, n - 1)][None, :, :]  # [1, T, 3]

    drivers = drivers or []
    n_drv = len(drivers)
    fig, axes = plt.subplots(
        1 + n_drv,
        1,
        figsize=(11, 2.1 + 1.05 * n_drv),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15] + [1.0] * n_drv} if n_drv else None,
        squeeze=False,
    )
    ax = axes[0][0]
    ax.imshow(band, aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, loc="left")

    for i, (key, rule_text, series, thr) in enumerate(drivers):
        axd = axes[1 + i][0]
        xs = np.arange(len(series))
        axd.plot(xs, series, lw=1.0, color="#343C96")
        axd.axhline(thr, ls="--", lw=0.8, color="#8A3227")
        if np.nanmin(series) < 0 and thr > 0:
            # abs-valued rules trigger on either sign; show both bounds
            axd.axhline(-thr, ls="--", lw=0.8, color="#8A3227")
        axd.set_ylabel(key.split("/")[-1], fontsize=7)
        axd.text(
            0.995, 0.92, rule_text,
            transform=axd.transAxes, ha="right", va="top", fontsize=7, color="#8A3227",
        )
        axd.tick_params(labelsize=7)
        axd.grid(alpha=0.15, lw=0.5)
    axes[-1][0].set_xlabel("environment step", labelpad=4)

    # The legend sits under the axis label, not on top of it: with a band this short the
    # default placement lands the swatches straight through the tick numbers.
    fig.legend(
        handles=[mpatches.Patch(color=colors[i], label=names[i]) for i in range(n)],
        loc="lower center",
        ncol=min(n, 4),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.18 / (1 + n_drv), 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# rollout + render
# --------------------------------------------------------------------------- #


def _unwrap_data(state: Any):
    """Dig through the env wrappers (LogVec/Clip/Normalize) down to the mjx State's .data."""
    for _ in range(8):
        if hasattr(state, "data"):
            return state.data
        state = getattr(state, "env_state", None)
        if state is None:
            break
    raise AttributeError("could not locate .data on the wrapped env state")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="MP4 path (default: next to the checkpoint)")
    ap.add_argument("--steps", type=int, default=None, help="default: the env's episode length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default=None, help="camera name; overrides the tracking camera")
    ap.add_argument(
        "--track",
        default="auto",
        help="body to keep centred: 'auto' (pick the root/torso body), a body name, or 'off' "
        "for the static free camera. Locomotion agents walk out of a static frame within a "
        "couple of seconds, which makes the clip useless for judging behaviour.",
    )
    ap.add_argument("--track-distance", type=float, default=0.0,
                    help="camera distance; 0 = auto from the model's extent")
    ap.add_argument("--track-elevation", type=float, default=-15.0)
    ap.add_argument("--track-azimuth", type=float, default=120.0)
    # Both default to multiples of 16: libx264 silently resizes anything else.
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=368)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--strip", action="store_true", help="also write a skill-activation timeline PNG")
    args = ap.parse_args(argv)

    import imageio.v2 as imageio  # noqa: PLC0415
    import mujoco  # noqa: PLC0415

    ck = _load_checkpoint(args.checkpoint)
    cfg = dict(ck["config"])
    rs = ck["runner_state"]
    stats = ck.get("normalization_stats")
    meta_type = str(cfg.get("META_POLICY_TYPE", "nesy")).lower()

    policy_module = load_policy_module(cfg.get("POLICY", cfg["ENV_NAME"]))
    num_skills = int(policy_module.NUM_SKILLS)
    skill_names = list(getattr(policy_module, "SKILL_NAMES", tuple(f"skill_{i}" for i in range(num_skills))))

    # One env, unnormalized: we apply the frozen training statistics ourselves, exactly as
    # the in-trainer deterministic evaluation does.
    eval_cfg = dict(cfg)
    eval_cfg["NORMALIZE_OBS"] = False
    eval_cfg["NORMALIZE_REWARD"] = False
    eval_cfg["NUM_ENVS"] = 1
    eval_cfg["USE_RGB"] = False  # the in-loop renderer is a separate path; this is state-based
    bundle = build_playground_env(eval_cfg)
    env, env_params = bundle.env, bundle.env_params

    action_low = jnp.asarray(bundle.action_low)
    action_high = jnp.asarray(bundle.action_high)
    action_scale = (action_high - action_low) / 2.0
    steps = int(args.steps or bundle.episode_length)

    actor, meta_q = _build_networks(
        cfg, num_skills, bundle.action_dim, action_scale, (action_high + action_low) / 2.0
    )

    raw_obs, env_state = env.reset(jax.random.split(jax.random.PRNGKey(args.seed), 1), env_params)
    dummy_actor = get_actor_obs(raw_obs)
    fresh_actor = jax.vmap(lambda k: actor.init(k, dummy_actor)["params"])(
        jax.random.split(jax.random.PRNGKey(0), num_skills)
    )
    actor_params = _restore_params(fresh_actor, rs["0"]["actor"]["params"])
    if meta_type in ("neural", "nesy"):
        fresh_meta = meta_q.init(jax.random.PRNGKey(1), dummy_actor)["params"]
        meta_params = _restore_params(fresh_meta, rs["0"]["meta"]["params"])
    else:
        meta_params = None

    normalize_obs = _make_normalizer(stats, cfg.get("NORMALIZE_OBS", True))
    decision_interval = int(cfg.get("META_DECISION_INTERVAL", 1))
    if decision_interval > 1:
        print(f"note: META_DECISION_INTERVAL={decision_interval} — rendering with option commitment")
    select = jax.jit(
        _make_selector(
            actor, meta_q, actor_params, meta_params, policy_module,
            meta_type, action_low, action_high, decision_interval,
        )
    )
    hold = init_hold_state(1, decision_interval)

    mj_model = getattr(bundle, "mj_model", None)
    if mj_model is None:
        from mujoco_playground import registry  # noqa: PLC0415

        probe_cfg = registry.get_default_config(cfg["ENV_NAME"])
        probe_cfg.impl = cfg.get("PLAYGROUND_IMPL", "jax")
        probe_env = registry.load(cfg["ENV_NAME"], probe_cfg)
        mj_model = getattr(probe_env, "mj_model", None) or probe_env._mj_model

    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    mj_data = mujoco.MjData(mj_model)

    # Tracking camera. A static free camera is fine for cartpole and panda, which stay put, but
    # locomotion agents (walker, hopper, cheetah, Go1) leave the frame in a second or two — the
    # clip then shows empty floor, which is worse than useless because it looks like a failed
    # rollout. mjCAMERA_TRACKING follows the body's position while keeping the orientation fixed.
    track_cam = None
    if not args.camera and str(args.track).lower() not in ("off", "none", ""):
        body_id = -1
        if str(args.track).lower() != "auto":
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, args.track)
            if body_id < 0:
                print(f"warning: body '{args.track}' not found; falling back to auto")
        if body_id < 0:
            # Auto: prefer a torso/trunk/base-looking body, else the first non-world body, which
            # for these models is the free-floating root.
            names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(mj_model.nbody)
            ]
            preferred = ("torso", "trunk", "base", "root", "pelvis", "chassis")
            for i, nm in enumerate(names):
                if i and nm and any(p in nm.lower() for p in preferred):
                    body_id = i
                    break
            if body_id < 0 and mj_model.nbody > 1:
                body_id = 1
        if body_id >= 0:
            track_cam = mujoco.MjvCamera()
            track_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            track_cam.trackbodyid = body_id
            track_cam.distance = args.track_distance or float(mj_model.stat.extent) * 3.0
            track_cam.elevation = args.track_elevation
            track_cam.azimuth = args.track_azimuth
            tracked = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"camera: tracking body '{tracked}' (id {body_id}), distance {track_cam.distance:.2f}")

    # Per-step semantic drivers for the strip's "why" panels: the signals the hand-written rule
    # reads, recorded through the same diagnostics() the trainer logs, so the strip's cause and
    # the training metrics can never disagree.
    policy_key = str(cfg.get("POLICY", cfg["ENV_NAME"]))
    driver_spec = STRIP_DRIVERS.get(policy_key, [])
    diag_fn = getattr(policy_module, "diagnostics", None) or getattr(policy_module, "task_metrics", None)
    driver_series: dict[str, list[float]] = {k: [] for k, _, _ in driver_spec}

    frames: list[np.ndarray] = []
    skills: list[int] = []
    total_return = 0.0
    rng = jax.random.PRNGKey(args.seed + 1)

    for t in range(steps):
        obs = normalize_obs(raw_obs)
        action, skill, hold = select(obs, hold)
        skill_i = int(np.asarray(skill)[0])

        rng, k_step = jax.random.split(rng)
        raw_obs, env_state, reward, done, _info = env.step(
            jax.random.split(k_step, 1), env_state, action, env_params
        )
        total_return += float(np.asarray(reward)[0])
        if driver_spec and diag_fn is not None:
            next_obs = normalize_obs(raw_obs)
            diag = diag_fn(
                get_policy_obs(obs), get_policy_obs(next_obs), action, reward, done, None
            )
            for k in driver_series:
                if k in diag:
                    driver_series[k].append(float(np.asarray(diag[k]).reshape(-1)[0]))
        if hold is not None:
            # Mirror the trainer: a reset forces a re-decision on the next step, so a held
            # skill never survives an episode boundary.
            hold = (hold[0], hold[1], jnp.asarray(done).astype(bool))

        # Push the simulator state into a plain MjData so the CPU renderer can draw it.
        data = _unwrap_data(env_state)
        mj_data.qpos[:] = np.asarray(data.qpos)[0]
        mj_data.qvel[:] = np.asarray(data.qvel)[0]
        mujoco.mj_forward(mj_model, mj_data)
        if args.camera:
            renderer.update_scene(mj_data, camera=args.camera)
        elif track_cam is not None:
            renderer.update_scene(mj_data, camera=track_cam)
        else:
            renderer.update_scene(mj_data)

        name = skill_names[skill_i] if skill_i < len(skill_names) else f"skill_{skill_i}"
        color = SKILL_COLORS[skill_i % len(SKILL_COLORS)]
        frames.append(
            _annotate(
                renderer.render(),
                [
                    (f"{cfg['ENV_NAME']}  [{meta_type}]", (235, 235, 235)),
                    (f"step {t + 1}/{steps}   return {total_return:8.1f}", (200, 200, 200)),
                    (f"skill {skill_i}: {name}", color),
                ],
            )
        )
        skills.append(skill_i)

        if bool(np.asarray(done)[0]):
            break

    renderer.close()

    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=args.fps, codec="libx264", quality=7)

    counts = np.bincount(np.asarray(skills), minlength=num_skills) / max(len(skills), 1)

    # Sidecar manifest so the dashboard can attach this clip to the right environment without
    # guessing from the filename — the checkpoint knows what it is, so record it.
    import json  # noqa: PLC0415

    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "env": cfg["ENV_NAME"],
                "variant": meta_type,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
                "frames": len(frames),
                "return": total_return,
                "skill_names": skill_names,
                "skill_usage": [float(c) for c in counts],
                "video": out.name,
                "strip": out.stem + "_skills.png" if args.strip else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {out}  ({len(frames)} frames, return {total_return:.1f})")
    print("skill usage:")
    for i, n in enumerate(skill_names):
        print(f"  {i} {n:<22} {counts[i]:6.1%}")

    if args.strip:
        strip = out.with_name(out.stem + "_skills.png")
        drivers = [
            (key, rule_text, np.asarray(driver_series[key]), thr)
            for key, thr, rule_text in driver_spec
            if len(driver_series.get(key, [])) == len(skills)
        ]
        _activation_strip(
            np.asarray(skills),
            skill_names,
            strip,
            f"{cfg['ENV_NAME']} [{meta_type}] — skill activation, return {total_return:.1f}",
            drivers=drivers,
        )
        print(f"wrote {strip}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
