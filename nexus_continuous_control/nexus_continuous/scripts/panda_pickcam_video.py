"""Rollout VIDEO for the PandaPickCube RGB camera probe (no training involved).

Companion to `panda_pickcam_probe.py` (which only saves stills). Reuses that
module's render plumbing verbatim -- `build_env` (which keeps the strong
`RenderContext` reference `refit_bvh` needs) and `render`.

The arm is driven by a **scripted open-loop joint trajectory** (a keyframe
interpolation `home -> pickup` plus a per-episode sinusoidal base-yaw sweep and
a timed gripper close). There is NO trained panda pixel policy; every frame is
captioned to say so.

Outputs (under --out):
  * rollout_SCRIPTED_<ghost>.mp4 / .gif
        side-by-side: LEFT 64x64 actor camera NEAREST-upscaled to 256 (the true
        pixel budget), RIGHT an independent 256x256 render for human reading.
        Overlays the live box (x, y), the timestep and the episode index.
  * grayscale_vs_colour_<ghost>.mp4 / .gif / .png
        the same 64x64 actor frames, colour vs the pipeline's grayscale
        (`mean(rgb, -1) - 0.5`, see envs/dm_control_vision.py).
  * grayscale_metrics_<ghost>.json
        measured cube/ghost vs surround contrast in colour and in grayscale.

    python -m nexus_continuous.scripts.panda_pickcam_video \
        --out results/rgb/panda_probe/video
    python -m nexus_continuous.scripts.panda_pickcam_video \
        --out results/rgb/panda_probe/video --ghost-rgba "0 0.9 0.25 1" --tag greenghost
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

from nexus_continuous.scripts.panda_pickcam_probe import (
    arrayify,
    build_env,
    render,
    save_png,
    to_u8,
)

PANEL = 256
PAD = 4
HEADER = 24
SUBLAB = 15
FOOTER = 54


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to01(img):
    """MJWarp `get_rgb` may hand back 0..1 floats or 0..255; normalise to 0..1."""
    a = np.asarray(img, np.float32)
    if a.max() > 1.001:
        a = a / 255.0
    return np.clip(a[..., :3], 0.0, 1.0)


def fonts():
    import matplotlib
    from PIL import ImageFont

    d = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    reg = [ImageFont.truetype(str(d / "DejaVuSans.ttf"), s) for s in (12, 11, 10, 9, 8)]
    bold = [ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), s) for s in (13, 12, 11, 10, 9)]
    return bold, reg


def draw_fit(d, xy, text, font_ladder, maxw, fill):
    """Draw `text`, shrinking through `font_ladder` until it fits in `maxw` px."""
    f = font_ladder[-1]
    for cand in font_ladder:
        if d.textlength(text, font=cand) <= maxw:
            f = cand
            break
    d.text(xy, text, font=f, fill=fill)


def upscale(img01, size, nearest=True):
    from PIL import Image

    im = Image.fromarray(to_u8(img01 * 255.0))
    return im.resize((size, size), Image.NEAREST if nearest else Image.BICUBIC)


def gray_of(rgb01):
    """EXACTLY the production conversion in envs/dm_control_vision.py:
    `gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5`, then displayed as
    `(gray + 0.5) * 255` (see rgb_inloop_visualize.to_img)."""
    return rgb01.mean(-1)


def dilate(mask, it=2):
    out = mask.copy()
    for _ in range(it):
        p = np.pad(out, 1)
        out = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | out
    return out


def sat(x):
    return x.max(-1) - x.min(-1)


def write_video(path_mp4, path_gif, frames, fps, gif_stride, gif_fps):
    """mp4 + gif fallback. Returns (mp4_bytes, gif_bytes); mp4_bytes==0 on failure."""
    import imageio.v2 as imageio

    arr = np.stack([np.asarray(f, np.uint8) for f in frames])
    # libx264 / yuv420p needs even dimensions
    h, w = arr.shape[1:3]
    ph, pw = h % 2, w % 2
    if ph or pw:
        arr = np.pad(arr, ((0, 0), (0, ph), (0, pw), (0, 0)))
    mp4_bytes = 0
    try:
        imageio.mimsave(path_mp4, arr, fps=fps, codec="libx264",
                        pixelformat="yuv420p", macro_block_size=1, quality=8)
        mp4_bytes = path_mp4.stat().st_size
    except Exception as exc:  # noqa: BLE001
        print(f"  !! mp4 encode failed for {path_mp4.name}: {exc}")
    imageio.mimsave(path_gif, arr[::gif_stride], fps=gif_fps, loop=0)
    return mp4_bytes, path_gif.stat().st_size


# --------------------------------------------------------------------------- #
# scripted trajectory
# --------------------------------------------------------------------------- #
def scripted_ctrl_ref(t, T, ctrl_home, ctrl_pickup, n):
    """Per-world reference ctrl at step `t`.

    home -> pickup keyframe ramp (first 35%), a per-episode sinusoidal base-yaw
    sweep so each episode looks different and the arm crosses the frame, plus a
    timed gripper close. Purely open loop: it never looks at the cube.
    """
    s = float(np.clip(t / (0.35 * T), 0.0, 1.0))
    s = s * s * (3 - 2 * s)  # smoothstep
    ref = np.tile(ctrl_home + s * (ctrl_pickup - ctrl_home), (n, 1))
    w = np.arange(n)
    amp = 0.28 + 0.09 * w                     # per-episode sweep amplitude (rad)
    period = 78.0 + 11.0 * w                  # per-episode sweep period (steps)
    phase = 0.55 * w
    ramp = float(np.clip((t - 0.25 * T) / (0.2 * T), 0.0, 1.0))
    ref[:, 0] += ramp * amp * np.sin(2 * np.pi * t / period + phase)
    ref[:, 1] += ramp * 0.10 * np.sin(2 * np.pi * t / (period * 0.5) + phase)
    ref[:, 7] = 0.04 if t < 0.62 * T else 0.0  # gripper: open -> close
    return ref


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="nexus_continuous/envs/xmls/mjx_single_cube_pickcam.xml")
    ap.add_argument("--out", default="results/rgb/panda_probe/video")
    ap.add_argument("--episode-seeds", type=int, nargs="+", default=[0, 2, 7, 5, 4])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-stride", type=int, default=5)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--ghost-rgba", default=None,
                    help='recolour the mocap_target geom, e.g. "0 0.9 0.25 1"')
    ap.add_argument("--tag", default="redghost")
    args = ap.parse_args()

    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx as _mjx
    from PIL import Image, ImageDraw

    root = pathlib.Path(__file__).resolve().parents[2]
    xml_path = root / args.xml
    xml_text = xml_path.read_text()
    ghost_desc = "RED (upstream rgba 1 0 0 0.2, alpha ignored by rasteriser)"
    if args.ghost_rgba:
        xml_text, k = re.subn(
            r'(<geom type="box" size="0.02 0.02 0.03" rgba=")1 0 0 0.2(")',
            rf"\g<1>{args.ghost_rgba}\g<2>", xml_text)
        assert k == 1, f"mocap_target geom rgba not found (n={k})"
        ghost_desc = f"RECOLOURED rgba={args.ghost_rgba}"
    outdir = root / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = list(args.episode_seeds)
    n = len(seeds)
    T = int(args.steps)
    print(f"episodes={n} seeds={seeds} steps={T} ghost={ghost_desc}")

    env64, rc64, mjm = build_env(xml_text, n, (64, 64))
    env256, rc256, _ = build_env(xml_text, n, (PANEL, PANEL))

    ctrl_home = np.asarray(mjm.keyframe("home").ctrl, np.float64).copy()
    ctrl_pickup = np.asarray(mjm.keyframe("pickup").ctrl, np.float64).copy()
    lo = np.asarray(env64._lowers, np.float64)
    hi = np.asarray(env64._uppers, np.float64)
    ascale = float(getattr(env64, "_action_scale", env64._config.action_scale))
    qadr = env64._obj_qposadr

    reset_fn = jax.jit(jax.vmap(env64.reset))
    step_fn = jax.jit(jax.vmap(env64.step))
    fwd = jax.jit(jax.vmap(_mjx.forward, in_axes=(None, 0)))

    rngs = jp.stack([jax.random.PRNGKey(s) for s in seeds])
    st = reset_fn(rngs)
    box0 = np.asarray(st.data.qpos[:, qadr:qadr + 3])
    print("initial box xyz per episode:\n", box0.round(3))
    # reset data is PRE-kinematics -- must forward before the first render.
    st = st.replace(data=fwd(env64.mjx_model, arrayify(st.data)))

    # ------------------------------ rollout ------------------------------ #
    fr64, fr256, boxes = [], [], []
    for t in range(T):
        r64 = to01(render(env64, rc64, st.data))
        r256 = to01(render(env256, rc256, st.data))
        fr64.append(r64)
        fr256.append(r256)
        boxes.append(np.asarray(st.data.qpos[:, qadr:qadr + 3]))
        ref = scripted_ctrl_ref(t, T, ctrl_home, ctrl_pickup, n)
        ctrl = np.asarray(st.data.ctrl)
        act = np.clip((np.clip(ref, lo, hi) - ctrl) / ascale, -1.0, 1.0)
        st = step_fn(st, jp.asarray(act, jp.float32))
        if t % 30 == 0:
            print(f"  t={t:3d} ok")
    fr64 = np.stack(fr64)      # [T, n, 64, 64, 3]
    fr256 = np.stack(fr256)    # [T, n, 256, 256, 3]
    boxes = np.stack(boxes)    # [T, n, 3]

    # ------------------------- composite video --------------------------- #
    fb, fr = fonts()
    W = PAD + PANEL + 2 * PAD + PANEL + PAD
    H = HEADER + SUBLAB + PANEL + FOOTER
    TW = W - 2 * PAD - 4
    ghost_short = "GREEN (fixed)" if args.ghost_rgba else "RED (upstream -> confusable)"
    frames = []
    for e in range(n):
        for t in range(T):
            im = Image.new("RGB", (W, H), (24, 24, 26))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, W, HEADER - 1], fill=(120, 20, 20))
            draw_fit(d, (PAD + 2, 5),
                     "SCRIPTED OPEN-LOOP TRAJECTORY - NOT A LEARNED POLICY",
                     fb, TW, (255, 235, 235))
            y = HEADER
            draw_fit(d, (PAD + 2, y + 2), "actor view: 64x64, NEAREST x4",
                     fr[2:], PANEL - 4, (190, 210, 255))
            draw_fit(d, (PAD + PANEL + 2 * PAD + 2, y + 2), "human view: 256x256 render",
                     fr[2:], PANEL - 4, (190, 210, 255))
            y += SUBLAB
            im.paste(upscale(fr64[t, e], PANEL, nearest=True), (PAD, y))
            im.paste(upscale(fr256[t, e], PANEL, nearest=False),
                     (PAD + PANEL + 2 * PAD, y))
            d.rectangle([PAD - 1, y - 1, PAD + PANEL, y + PANEL], outline=(90, 90, 100))
            d.rectangle([PAD + PANEL + 2 * PAD - 1, y - 1,
                         PAD + 2 * PANEL + 2 * PAD, y + PANEL], outline=(90, 90, 100))
            y += PANEL + 4
            bx, by, bz = boxes[t, e]
            draw_fit(d, (PAD + 2, y),
                     f"episode {e + 1}/{n} (seed {seeds[e]})   t = {t:3d}/{T}"
                     f"   box xy = ({bx:+.3f}, {by:+.3f}) m   z = {bz:.3f}",
                     fr, TW, (235, 235, 235))
            draw_fit(d, (PAD + 2, y + 16), f"mocap-target ghost: {ghost_short}",
                     fr, TW, (150, 255, 170) if args.ghost_rgba else (255, 150, 150))
            draw_fit(d, (PAD + 2, y + 32),
                     "PandaPickCube joint-space (8-dim) | front cam fovy 54 | "
                     "no textures, no shadows, geom groups 0/1/2",
                     fr[2:], TW, (160, 160, 170))
            frames.append(np.asarray(im))

    mp4 = outdir / f"rollout_SCRIPTED_{args.tag}.mp4"
    gif = outdir / f"rollout_SCRIPTED_{args.tag}.gif"
    nb_mp4, nb_gif = write_video(mp4, gif, frames, args.fps, args.gif_stride, args.gif_fps)
    print(f"  {mp4.name}: {nb_mp4/1e6:.2f} MB   {gif.name}: {nb_gif/1e6:.2f} MB")
    save_png(outdir / f"rollout_SCRIPTED_{args.tag}_still.png", frames[int(0.55 * T)])

    # -------------------- grayscale contrast measurement ----------------- #
    # ablation renders (mocap hidden / box+mocap hidden) at reset AND mid-rollout,
    # same recipe as panda_pickcam_probe, so we get exact cube / ghost masks.
    st_r = reset_fn(rngs)
    st_m = st  # final rollout state (arm down, gripper closed)

    def hide_mocap(state):
        far = jp.broadcast_to(jp.array([10.0, 10.0, 10.0]),
                              state.data.mocap_pos[:, env64._mocap_target, :].shape)
        return state.data.replace(
            mocap_pos=state.data.mocap_pos.at[:, env64._mocap_target, :].set(far))

    def hide_box(data):
        return data.replace(qpos=data.qpos.at[:, qadr:qadr + 3].set(
            jp.broadcast_to(jp.array([10.0, 10.0, 10.0]), data.qpos[:, qadr:qadr + 3].shape)))

    metrics = {"ghost": ghost_desc, "episode_seeds": seeds,
               "grayscale_formula": "mean(rgb, axis=-1) - 0.5   (envs/dm_control_vision.py)"}
    for phase, state in (("reset_home_pose", st_r), ("end_of_scripted_rollout", st_m)):
        full = to01(render(env64, rc64, fwd(env64.mjx_model, arrayify(state.data))))
        nomo = to01(render(env64, rc64, fwd(env64.mjx_model, arrayify(hide_mocap(state)))))
        nobx = to01(render(env64, rc64,
                           fwd(env64.mjx_model, arrayify(hide_box(hide_mocap(state))))))
        cube_mask = (np.abs(nomo - nobx).sum(-1) > 0.05) & (sat(nomo) > 0.25)
        ghost_mask = (np.abs(full - nomo).sum(-1) > 0.05) & (sat(full) > 0.25)
        entry = {}
        for name, mask, src in (("cube", cube_mask, nomo), ("ghost", ghost_mask, full)):
            cm, rm, gc, gr, gcon = [], [], [], [], []
            for i in range(n):
                m = mask[i]
                if not m.any():
                    continue
                ring = dilate(m, 2) & ~m
                cm.append(src[i][m].mean(0))
                rm.append(src[i][ring].mean(0))
                g = gray_of(src[i])
                gc.append(g[m].mean())
                gr.append(g[ring].mean())
                gcon.append(abs(g[m].mean() - g[ring].mean()))
            if not cm:
                entry[name] = {"pixels_found": 0}
                continue
            cm_all, rm_all = np.stack(cm), np.stack(rm)   # [k, 3] per-episode means
            cm, rm = cm_all.mean(0), rm_all.mean(0)
            # Which 1-D projection of RGB preserves the cube? Measured on the same
            # real frames; `mean_rgb` is what the production pipeline actually uses.
            proj = {
                "mean_rgb_PRODUCTION": (1 / 3, 1 / 3, 1 / 3),
                "itu601_luma": (0.299, 0.587, 0.114),
                "red_channel": (1.0, 0.0, 0.0),
                "red_minus_blue": (1.0, 0.0, -1.0),
                "green_channel": (0.0, 1.0, 0.0),
            }
            entry[name] = {
                "n_episodes_with_pixels": len(gc),
                "mean_pixels": float(mask.reshape(n, -1).sum(1).mean()),
                f"{name}_mean_rgb_0_255": (cm * 255).round(1).tolist(),
                "surround_mean_rgb_0_255": (rm * 255).round(1).tolist(),
                "COLOUR_mean_abs_rgb_contrast_0_255": round(float(np.abs(cm - rm).mean() * 255), 2),
                "COLOUR_max_channel_contrast_0_255": round(float(np.abs(cm - rm).max() * 255), 2),
                f"GRAY_{name}_mean_0_255": round(float(np.mean(gc) * 255), 2),
                "GRAY_surround_mean_0_255": round(float(np.mean(gr) * 255), 2),
                "GRAY_contrast_pooled_means_0_255": round(
                    float(abs(np.mean(gc) - np.mean(gr)) * 255), 2),
                "GRAY_contrast_0_255": round(float(np.mean(gcon) * 255), 2),
                "GRAY_contrast_per_episode_0_255": [round(float(v * 255), 2) for v in gcon],
                "colour_over_gray_contrast_ratio": round(
                    float(np.abs(cm - rm).mean() / max(np.mean(gcon), 1e-9)), 1),
                "projection_contrast_0_255": {
                    k: round(float(np.abs((cm_all - rm_all) @ np.asarray(w)).mean() * 255), 2)
                    for k, w in proj.items()
                },
            }
        # whole-frame information content, colour vs grayscale
        entry["frame_std_0_255_colour"] = round(float(full.std() * 255), 2)
        entry["frame_std_0_255_gray"] = round(float(gray_of(full).std() * 255), 2)
        metrics[phase] = entry

    (outdir / f"grayscale_metrics_{args.tag}.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== GRAYSCALE METRICS ===")
    print(json.dumps(metrics, indent=2))

    # ------------------ colour vs grayscale video + still ---------------- #
    gW = PAD + PANEL + 2 * PAD + PANEL + PAD
    gH = HEADER + SUBLAB + PANEL + 40
    gcon_r = metrics["reset_home_pose"]["cube"].get("GRAY_contrast_0_255", float("nan"))
    ccon_r = metrics["reset_home_pose"]["cube"].get("COLOUR_mean_abs_rgb_contrast_0_255",
                                                    float("nan"))
    gframes = []
    for e in range(n):
        for t in range(0, T, 2):
            im = Image.new("RGB", (gW, gH), (24, 24, 26))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, gW, HEADER - 1], fill=(20, 60, 110))
            draw_fit(d, (PAD + 2, 5),
                     "64x64 ACTOR VIEW - colour vs pipeline grayscale (SCRIPTED)",
                     fb, TW, (230, 240, 255))
            y = HEADER
            draw_fit(d, (PAD + 2, y + 2), f"COLOUR  cube-vs-surround {ccon_r:.1f}/255",
                     fr[2:], PANEL - 4, (190, 255, 190))
            draw_fit(d, (PAD + PANEL + 2 * PAD + 2, y + 2),
                     f"GRAY mean(rgb)  cube-vs-surround {gcon_r:.1f}/255",
                     fr[2:], PANEL - 4, (255, 190, 190))
            y += SUBLAB
            im.paste(upscale(fr64[t, e], PANEL, nearest=True), (PAD, y))
            g = gray_of(fr64[t, e])
            im.paste(upscale(np.repeat(g[..., None], 3, -1), PANEL, nearest=True),
                     (PAD + PANEL + 2 * PAD, y))
            y += PANEL + 4
            bx, by = boxes[t, e, 0], boxes[t, e, 1]
            draw_fit(d, (PAD + 2, y),
                     f"ep {e + 1}/{n} (seed {seeds[e]})  t={t:3d}  "
                     f"box xy = ({bx:+.3f}, {by:+.3f})  ghost: {ghost_short}",
                     fr, TW, (235, 235, 235))
            draw_fit(d, (PAD + 2, y + 17),
                     "grayscale collapses the red cube onto the blue-grey mat: "
                     "near-equal luminance, opposite hue",
                     fr[2:], TW, (255, 200, 120))
            gframes.append(np.asarray(im))
    gmp4 = outdir / f"grayscale_vs_colour_{args.tag}.mp4"
    ggif = outdir / f"grayscale_vs_colour_{args.tag}.gif"
    nb1, nb2 = write_video(gmp4, ggif, gframes, 20, 4, 8)
    print(f"  {gmp4.name}: {nb1/1e6:.2f} MB   {ggif.name}: {nb2/1e6:.2f} MB")
    save_png(outdir / f"grayscale_vs_colour_{args.tag}.png",
             gframes[min(len(gframes) - 1, 8)])

    # a multi-episode colour/gray contact sheet (t=0 poses, all episodes)
    S = 128
    sheet = Image.new("RGB", (PAD + n * (S + PAD), 2 * (S + 18) + PAD), (24, 24, 26))
    ds = ImageDraw.Draw(sheet)
    for e in range(n):
        x = PAD + e * (S + PAD)
        sheet.paste(upscale(fr64[0, e], S, True), (x, 16))
        ds.text((x, 2), f"s{seeds[e]} x{box0[e,0]:.2f} y{box0[e,1]:+.2f}", font=fr[2],
                fill=(220, 220, 220))
        sheet.paste(upscale(np.repeat(gray_of(fr64[0, e])[..., None], 3, -1), S, True),
                    (x, 16 + S + 18))
        ds.text((x, S + 20), "same frame, grayscale", font=fr[4], fill=(255, 190, 190))
    save_png(outdir / f"contact_colour_vs_gray_{args.tag}.png", np.asarray(sheet))

    print("\nsaved to", outdir)


if __name__ == "__main__":
    main()
