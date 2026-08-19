"""One-off camera probe for PandaPickCube (no training).

Renders the joint-space `PandaPickCube` task through the re-aimed `front` camera
added by `nexus_continuous/envs/xmls/mjx_single_cube_pickcam.xml`, using the same
MJWarp path production uses (`mjx.create_render_context` -> `refit_bvh` ->
`mjx.render` -> `mjx.get_rgb`) and the exact `default_vision_config()` knobs from
`pick_cartesian.py` (use_textures=False, use_shadows=False, geom groups [0,1,2]).

Saves 64x64 (actor view) and 256x256 (human view) PNGs + a contact sheet, and
prints a JSON block of visibility / contrast / discriminability metrics.

    python -m nexus_continuous.scripts.panda_pickcam_probe --out results/rgb/panda_probe
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np


# --------------------------------------------------------------------------- #
# camera helper
# --------------------------------------------------------------------------- #
def lookat_xyaxes(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo `xyaxes` (camera right, camera up) for a camera at `cam_pos`
    looking at `target`."""
    c = np.asarray(cam_pos, float)
    t = np.asarray(target, float)
    fwd = t - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    zc = -fwd
    upc = np.cross(zc, right)
    upc /= np.linalg.norm(upc)
    return np.concatenate([right, upc])


_CAM_RE = re.compile(r'<camera name="front"[^/]*/>')


def patch_camera(xml_text, cam_pos, target, fovy):
    xy = lookat_xyaxes(cam_pos, target)
    cam = (
        '<camera name="front" fovy="{:.4g}" pos="{:.5g} {:.5g} {:.5g}" '
        'xyaxes="{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}"/>'
    ).format(fovy, *cam_pos, *xy)
    out, n = _CAM_RE.subn(cam, xml_text)
    assert n == 1, f"camera element not found/ambiguous (n={n})"
    return out, cam


# --------------------------------------------------------------------------- #
def build_env(xml_text, nworld, cam_res):
    import jax
    import mujoco
    from mujoco import mjx
    from mujoco_playground._src.manipulation.franka_emika_panda import panda, pick

    from nexus_continuous.envs.playground_adapter import (
        ensure_mjwarp_graphmode,
        ensure_mjx_render_compat,
    )

    ensure_mjwarp_graphmode()
    ensure_mjx_render_compat()

    cfg = pick.default_config()
    cfg.impl = "warp"
    env = pick.PandaPickCube(config=cfg)

    assets = panda.get_assets()
    mjm = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
    mjm.opt.timestep = env.sim_dt
    env._mj_model = mjm
    env._mjx_model = mjx.put_model(mjm, impl=cfg.impl)
    env._post_init(obj_name="box", keyframe="home")
    env._floor_hand_found_sensor = [
        mjm.sensor(f"{g}_floor_found").id
        for g in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

    rc_kw = dict(
        nworld=int(nworld),
        cam_res=tuple(cam_res),
        use_textures=False,
        use_shadows=False,
        render_rgb=(True,),
        render_depth=(False,),
        enabled_geom_groups=[0, 1, 2],
        cam_active=None,
    )
    # NB: keep a strong reference -- RenderContext.__del__ frees the global
    # _MJX_RENDER_CONTEXT_BUFFERS entry, and refit_bvh then KeyErrors.
    env._rc = mjx.create_render_context(mjm=mjm, **rc_kw)
    return env, env._rc.pytree(), mjm


def arrayify(data):
    import jax
    import jax.numpy as jp

    return jax.tree_util.tree_map(
        lambda x: x if hasattr(x, "shape") else jp.asarray(x, jp.float32), data
    )


def render(env, rc_pt, data):
    from mujoco import mjx

    data = arrayify(data)
    data = mjx.refit_bvh(env.mjx_model, data, rc_pt)
    out = mjx.render(env.mjx_model, data, rc_pt)
    rgb = mjx.get_rgb(rc_pt, 0, out[0])
    return np.asarray(rgb)


# --------------------------------------------------------------------------- #
def to_u8(img):
    a = np.asarray(img, np.float32)
    if a.max() <= 1.001:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def save_png(path, img):
    from PIL import Image

    a = to_u8(img)
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    Image.fromarray(a).save(path)


def montage(frames, cols, scale, labels=None):
    """Tile HxWx3 uint8 frames into a labelled grid (nearest-neighbour upscale)."""
    from PIL import Image, ImageDraw

    fr = [to_u8(f)[..., :3] for f in frames]
    h, w = fr[0].shape[:2]
    rows = int(np.ceil(len(fr) / cols))
    pad, lab = 4, (12 if labels else 0)
    tw, th = w * scale + pad, h * scale + pad + lab
    sheet = Image.new("RGB", (cols * tw + pad, rows * th + pad), (32, 32, 32))
    dr = ImageDraw.Draw(sheet)
    for i, f in enumerate(fr):
        r, c = divmod(i, cols)
        im = Image.fromarray(f).resize((w * scale, h * scale), Image.NEAREST)
        x, y = pad + c * tw, pad + r * th
        sheet.paste(im, (x, y))
        if labels:
            dr.text((x + 2, y + h * scale + 1), labels[i], fill=(230, 230, 230))
    return np.asarray(sheet)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="nexus_continuous/envs/xmls/mjx_single_cube_pickcam.xml")
    ap.add_argument("--out", default="results/rgb/panda_probe")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=None)
    ap.add_argument("--lookat", type=float, nargs=3, default=None)
    ap.add_argument("--fovy", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ghost-rgba", default=None,
                    help='e.g. "0 0.9 0.2 1" -- recolour the mocap_target geom '
                         "(the MJWarp rasterizer ignores its alpha, so by default "
                         "it renders as an opaque box in the same red as the cube)")
    ap.add_argument("--sweep-only", action="store_true",
                    help="only render the 256px seed-0 frame (fast pose iteration)")
    ap.add_argument("--rollout-steps", type=int, default=40)
    ap.add_argument("--probe-seeds", type=int, default=96,
                    help="extra reset batch used only for the linear-decodability test")
    args = ap.parse_args()

    import jax
    import jax.numpy as jp

    root = pathlib.Path(__file__).resolve().parents[2]
    xml_path = (root / args.xml) if not pathlib.Path(args.xml).is_absolute() else pathlib.Path(args.xml)
    xml_text = xml_path.read_text()
    cam_desc = "(as authored in xml)"
    if args.cam_pos is not None:
        xml_text, cam_desc = patch_camera(
            xml_text, args.cam_pos, args.lookat or [0.68, 0.0, 0.10], args.fovy or 45.0
        )
    if args.ghost_rgba:
        xml_text, k = re.subn(r'(<geom type="box" size="0.02 0.02 0.03" rgba=")1 0 0 0.2(")',
                              rf"\g<1>{args.ghost_rgba}\g<2>", xml_text)
        assert k == 1, f"mocap_target geom rgba not found (n={k})"
    outdir = root / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    n = int(args.seeds)
    nworld = n + 1  # +1 slot for the mid-episode rollout frame

    # ---------------- reset states (production reset, verbatim) -------------- #
    env, rc64, mjm = build_env(xml_text, nworld, (64, 64))
    print("ncam:", mjm.ncam, "| cam names:",
          [mujoco_name(mjm, i) for i in range(mjm.ncam)])
    print("init_obj_pos:", np.asarray(env._init_obj_pos))
    print("keyframes:", [mjm.keyframe(i).name for i in range(mjm.nkey)])

    rngs = jax.vmap(jax.random.PRNGKey)(jp.arange(nworld))
    reset_fn = jax.jit(jax.vmap(env.reset))
    st = reset_fn(rngs)
    box_xy = np.asarray(st.data.qpos[:, env._obj_qposadr: env._obj_qposadr + 3])
    mocap = np.asarray(st.data.mocap_pos[:, env._mocap_target, :])

    # ------------- mid-episode state: real dynamics, scripted push ----------- #
    step_fn = jax.jit(jax.vmap(env.step))
    st_mid = st
    key = jax.random.PRNGKey(0)
    for t in range(args.rollout_steps):
        key, k = jax.random.split(key)
        # drive the arm down/forward toward the cube, close gripper late
        act = jp.tile(jp.array([0.0, 1.0, 0.0, -0.6, 0.0, 0.9, 0.0,
                                -1.0 if t > 25 else 1.0]), (nworld, 1))
        act = act + 0.05 * jax.random.normal(k, act.shape)
        st_mid = step_fn(st_mid, jp.clip(act, -1, 1))

    # -------------------------- ablation renders ----------------------------- #
    def with_mocap_hidden(state):
        far = jp.broadcast_to(jp.array([10.0, 10.0, 10.0]),
                              state.data.mocap_pos[:, env._mocap_target, :].shape)
        return state.data.replace(
            mocap_pos=state.data.mocap_pos.at[:, env._mocap_target, :].set(far))

    def with_box_hidden(data):
        a = env._obj_qposadr
        return data.replace(qpos=data.qpos.at[:, a: a + 3].set(
            jp.broadcast_to(jp.array([10.0, 10.0, 10.0]), data.qpos[:, a:a + 3].shape)))

    from mujoco import mjx as _mjx
    fwd = jax.jit(jax.vmap(_mjx.forward, in_axes=(None, 0)))

    # reset data is pre-kinematics (make_data does not run forward); production's
    # vision path calls mjx.forward before rendering -- do the same here.
    d_full = fwd(env.mjx_model, arrayify(st.data))
    d_nomocap = fwd(env.mjx_model, arrayify(with_mocap_hidden(st)))
    d_nobox = fwd(env.mjx_model, arrayify(with_box_hidden(with_mocap_hidden(st))))
    d_mid = fwd(env.mjx_model, arrayify(st_mid.data))
    d_mid_nomocap = fwd(env.mjx_model, arrayify(with_mocap_hidden(st_mid)))
    d_mid_nobox = fwd(env.mjx_model, arrayify(with_box_hidden(with_mocap_hidden(st_mid))))

    r64 = dict(
        full=render(env, rc64, d_full),
        nomocap=render(env, rc64, d_nomocap),
        nobox=render(env, rc64, d_nobox),
        mid=render(env, rc64, d_mid),
        mid_nomocap=render(env, rc64, d_mid_nomocap),
        mid_nobox=render(env, rc64, d_mid_nobox),
    )

    # 256px human view (fresh context, identical settings, higher res)
    env2, rc256, _ = build_env(xml_text, nworld, (256, 256))
    r256_full = render(env2, rc256, d_full)
    r256_mid = render(env2, rc256, d_mid)

    tag = f"_{args.tag}" if args.tag else ""
    for i in range(n):
        save_png(outdir / f"seed{i}_64{tag}.png", r64["full"][i])
        save_png(outdir / f"seed{i}_256{tag}.png", r256_full[i])
    save_png(outdir / f"mid_episode_64{tag}.png", r64["mid"][0])
    save_png(outdir / f"mid_episode_256{tag}.png", r256_mid[0])
    for i in (1, 2):
        save_png(outdir / f"mid_episode{i}_256{tag}.png", r256_mid[i])
        save_png(outdir / f"mid_episode{i}_64{tag}.png", r64["mid"][i])

    labels = [f"s{i} x{box_xy[i,0]:.2f} y{box_xy[i,1]:+.2f}" for i in range(n)]
    sheet = montage([r64["full"][i] for i in range(n)] + [r64["mid"][i] for i in range(3)],
                    cols=4, scale=3, labels=labels + ["mid0", "mid1", "mid2"])
    save_png(outdir / f"contact_sheet_64{tag}.png", sheet)
    sheet_nm = montage([r64["nomocap"][i] for i in range(n)], cols=4, scale=3, labels=labels)
    save_png(outdir / f"contact_sheet_64_no_mocap{tag}.png", sheet_nm)
    save_png(outdir / f"contact_sheet_256{tag}.png",
             montage([r256_full[i] for i in range(min(n, 8))], cols=4, scale=1, labels=labels))

    if args.sweep_only:
        return

    # ------------------------------- metrics --------------------------------- #
    f = np.asarray(r64["full"], np.float32)[:n]
    nm = np.asarray(r64["nomocap"], np.float32)[:n]
    nb = np.asarray(r64["nobox"], np.float32)[:n]
    if f.max() > 1.001:
        f, nm, nb = f / 255.0, nm / 255.0, nb / 255.0
    f, nm, nb = f[..., :3], nm[..., :3], nb[..., :3]

    cube_diff = np.abs(nm - nb).sum(-1)           # cube-only footprint
    ghost_diff = np.abs(f - nm).sum(-1)           # mocap ghost footprint

    # The rasteriser flips a handful of pixels on the arm's silhouette edge between
    # otherwise-identical renders, so gate the ablation diffs on chroma: the mat is
    # only weakly blue (sat ~0.10) and the panda shells are achromatic (~0.02),
    # while the cube / target box are saturated (~0.55).
    def sat(x):
        return x.max(-1) - x.min(-1)

    cube_mask = (cube_diff > 0.05) & (sat(nm) > 0.25)
    ghost_mask = (ghost_diff > 0.05) & (sat(f) > 0.25)

    res = {"camera": cam_desc, "n_seeds": n,
           "box_xyz": box_xy[:n].round(4).tolist(),
           "mocap_xyz": mocap[:n].round(4).tolist()}

    # (a) visibility
    res["cube_pixels_per_seed"] = cube_mask.reshape(n, -1).sum(1).tolist()
    res["cube_visible_all_seeds"] = bool((cube_mask.reshape(n, -1).sum(1) > 0).all())
    cen = []
    for i in range(n):
        ys, xs = np.nonzero(cube_mask[i])
        cen.append([float(xs.mean()), float(ys.mean())] if len(xs) else [np.nan, np.nan])
    cen = np.array(cen)
    res["cube_centroid_col_row"] = cen.round(2).tolist()
    res["cube_touches_border"] = [
        bool(cube_mask[i][0].any() or cube_mask[i][-1].any()
             or cube_mask[i][:, 0].any() or cube_mask[i][:, -1].any()) for i in range(n)]

    # (b) contrast: cube pixels vs a 2px ring around them
    def dilate(mask, it=2):
        out = mask.copy()
        for _ in range(it):
            p = np.pad(out, 1)
            out = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | out)
        return out

    contrasts, cube_mu, ring_mu = [], [], []
    for i in range(n):
        m = cube_mask[i]
        if not m.any():
            continue
        ring = dilate(m, 2) & ~m
        cm, rm = nm[i][m].mean(0), nm[i][ring].mean(0)
        cube_mu.append(cm)
        ring_mu.append(rm)
        contrasts.append(float(np.abs(cm - rm).mean()))
    res["cube_mean_rgb_0_255"] = (np.mean(cube_mu, 0) * 255).round(1).tolist()
    res["surround_mean_rgb_0_255"] = (np.mean(ring_mu, 0) * 255).round(1).tolist()
    res["cube_vs_surround_mean_abs_contrast_0_255"] = round(float(np.mean(contrasts)) * 255, 2)
    res["cube_vs_surround_per_seed_0_255"] = [round(c * 255, 1) for c in contrasts]
    res["global_frame_mean_rgb_0_255"] = (nm.mean((0, 1, 2)) * 255).round(1).tolist()
    res["global_frame_std_0_255"] = round(float(nm.std()) * 255, 2)

    # (c) arm/gripper: fraction of non-background pixels (background = floor+mat rows)
    #     use the box-and-mocap-free frame; arm is the only remaining foreground.
    #     The mat is chromatic (blue-ish); the panda shells are bright+achromatic.
    achro = (nb.min(-1) > 0.5) & ((nb.max(-1) - nb.min(-1)) < 0.10)
    res["arm_pixels_per_seed_home_pose"] = achro.reshape(n, -1).sum(1).tolist()
    mid = np.asarray(r64["mid_nobox"], np.float32)[:3]
    mid = (mid / 255.0 if mid.max() > 1.001 else mid)[..., :3]
    am = (mid.min(-1) > 0.5) & ((mid.max(-1) - mid.min(-1)) < 0.10)
    res["arm_pixels_mid_episode"] = am.reshape(3, -1).sum(1).tolist()
    # explicit: is the gripper site inside the 64x64 frame?
    cam = 0
    gp = np.asarray(d_full.site_xpos[:, env._gripper_site, :])
    cpos = np.asarray(d_full.cam_xpos[:, cam, :])
    cmat = np.asarray(d_full.cam_xmat[:, cam, :]).reshape(-1, 3, 3)
    fovy = float(mjm.cam_fovy[cam])
    th = np.tan(np.deg2rad(fovy / 2))
    v = np.einsum("bij,bi->bj", cmat, gp - cpos)  # into camera frame
    dep = -v[:, 2]
    col = v[:, 0] / (dep * th) * 32 + 32
    row = -v[:, 1] / (dep * th) * 32 + 32
    res["gripper_site_px_col_row"] = np.stack([col, row], 1)[:n].round(1).tolist()
    res["gripper_in_frame_all_seeds"] = bool(
        ((col[:n] > 0) & (col[:n] < 64) & (row[:n] > 0) & (row[:n] < 64)).all())
    gm2 = np.asarray(d_mid.site_xpos[:, env._gripper_site, :])
    v2 = np.einsum("bij,bi->bj", np.asarray(d_mid.cam_xmat[:, cam, :]).reshape(-1, 3, 3),
                   gm2 - np.asarray(d_mid.cam_xpos[:, cam, :]))
    dep2 = -v2[:, 2]
    res["gripper_site_px_mid_episode"] = np.stack(
        [v2[:, 0] / (dep2 * th) * 32 + 32, -v2[:, 1] / (dep2 * th) * 32 + 32], 1
    )[:3].round(1).tolist()
    res["gripper_world_xyz_mid_episode"] = gm2[:3].round(3).tolist()
    # occlusion check: cube footprint once the arm has actually moved into the scene
    mnm = np.asarray(r64["mid_nomocap"], np.float32)[:n]
    mnb = np.asarray(r64["mid_nobox"], np.float32)[:n]
    if mnm.max() > 1.001:
        mnm, mnb = mnm / 255.0, mnb / 255.0
    mid_cube = (np.abs(mnm[..., :3] - mnb[..., :3]).sum(-1) > 0.05) & (sat(mnm[..., :3]) > 0.25)
    res["cube_pixels_mid_episode"] = mid_cube.reshape(n, -1).sum(1).tolist()

    # (d) mocap ghost
    res["ghost_pixels_per_seed"] = ghost_mask.reshape(n, -1).sum(1).tolist()
    res["ghost_visible_any_seed"] = bool(ghost_mask.any())

    # (e) discriminability
    def pairwise_mad(x):
        v = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                v.append(float(np.abs(x[i] - x[j]).mean()))
        return np.array(v)

    mad_full = pairwise_mad(f) * 255
    mad_nomocap = pairwise_mad(nm) * 255
    mad_nobox = pairwise_mad(nb) * 255
    res["pairwise_MAD_0_255"] = {
        "production_frame(cube+ghost)": dict(mean=round(float(mad_full.mean()), 3),
                                             min=round(float(mad_full.min()), 3),
                                             max=round(float(mad_full.max()), 3)),
        "cube_only(no ghost)": dict(mean=round(float(mad_nomocap.mean()), 3),
                                    min=round(float(mad_nomocap.min()), 3),
                                    max=round(float(mad_nomocap.max()), 3)),
        "noise_floor(no cube,no ghost)": dict(mean=round(float(mad_nobox.mean()), 4),
                                              max=round(float(mad_nobox.max()), 4)),
    }
    changed = [(np.abs(nm[i] - nm[j]).sum(-1) > 0.05).sum()
               for i in range(n) for j in range(i + 1, n)]
    res["pixels_changed_between_seed_pairs(cube_only)"] = dict(
        mean=round(float(np.mean(changed)), 1), min=int(np.min(changed)), max=int(np.max(changed)))

    # does image position track world position?
    ok = ~np.isnan(cen[:, 0])
    if ok.sum() >= 3:
        res["corr_boxY_vs_imgCol"] = round(float(np.corrcoef(box_xy[:n][ok, 1], cen[ok, 0])[0, 1]), 4)
        res["corr_boxX_vs_imgRow"] = round(float(np.corrcoef(box_xy[:n][ok, 0], cen[ok, 1])[0, 1]), 4)
        res["img_col_range_px"] = round(float(np.ptp(cen[ok, 0])), 2)
        res["img_row_range_px"] = round(float(np.ptp(cen[ok, 1])), 2)

    # cube vs mocap-ghost colour/size (they share rgba 1 0 0 by default)
    if ghost_mask.any():
        gm = np.array([f[i][ghost_mask[i]].mean(0) for i in range(n) if ghost_mask[i].any()])
        res["ghost_mean_rgb_0_255"] = (gm.mean(0) * 255).round(1).tolist()
        res["ghost_vs_cube_mean_abs_rgb_diff_0_255"] = round(
            float(np.abs(gm.mean(0) - np.mean(cube_mu, 0)).mean()) * 255, 2)
        res["ghost_px_vs_cube_px"] = [int(np.mean(ghost_mask.reshape(n, -1).sum(1))),
                                      int(np.mean(cube_mask.reshape(n, -1).sum(1)))]

    # (e2) linear decodability of the cube pose from raw 64x64 pixels
    if args.probe_seeds and args.probe_seeds > 8:
        m = int(args.probe_seeds)
        env3, rc3, _ = build_env(xml_text, m, (64, 64))
        rngs3 = jax.vmap(jax.random.PRNGKey)(jp.arange(1000, 1000 + m))
        st3 = jax.jit(jax.vmap(env3.reset))(rngs3)
        fwd3 = jax.jit(jax.vmap(_mjx.forward, in_axes=(None, 0)))
        imgs = np.asarray(render(env3, rc3, fwd3(env3.mjx_model, arrayify(st3.data))),
                          np.float32)
        if imgs.max() > 1.001:
            imgs = imgs / 255.0
        X = imgs[..., :3].reshape(m, -1)
        Y = np.asarray(st3.data.qpos[:, env3._obj_qposadr: env3._obj_qposadr + 2])
        ntr, nva = int(m * 0.6), int(m * 0.2)
        sl = (slice(0, ntr), slice(ntr, ntr + nva), slice(ntr + nva, m))
        Xtr, Xva, Xte = (X[s] for s in sl)
        Ytr, Yva, Yte = (Y[s] for s in sl)
        mu, ym = Xtr.mean(0), Ytr.mean(0)
        A = Xtr - mu

        def fit_predict(lam, Xq):
            W = np.linalg.solve(A @ A.T + lam * np.eye(len(A)), Ytr - ym)
            return (Xq - mu) @ (A.T @ W) + ym

        lams = [10.0 ** k for k in range(-3, 5)]
        best = min(lams, key=lambda l: np.abs(fit_predict(l, Xva) - Yva).mean())
        pred = fit_predict(best, Xte)
        r2 = 1 - ((pred - Yte) ** 2).sum(0) / ((Yte - ym) ** 2).sum(0)
        # 1-NN pixel retrieval: nonlinear, no fitting -- "do similar frames mean
        # similar cube positions?"
        dist = ((Xte[:, None] - X[None, :ntr]) ** 2).sum(-1)
        nn = Ytr[dist.argmin(1)]
        r2nn = 1 - ((nn - Yte) ** 2).sum(0) / ((Yte - ym) ** 2).sum(0)
        res["decodability"] = {
            "n_total": m, "n_test": m - ntr - nva, "ridge_lambda": best,
            "ridge_mean_abs_err_m_[x,y]": np.abs(pred - Yte).mean(0).round(4).tolist(),
            "ridge_R2_[x,y]": r2.round(3).tolist(),
            "1nn_mean_abs_err_m_[x,y]": np.abs(nn - Yte).mean(0).round(4).tolist(),
            "1nn_R2_[x,y]": r2nn.round(3).tolist(),
            "chance_mean_abs_err_m": np.abs(Yte - ym).mean(0).round(4).tolist(),
        }

    print("\n===METRICS===")
    print(json.dumps(res, indent=2))
    (outdir / f"metrics{tag}.json").write_text(json.dumps(res, indent=2))
    print("saved to", outdir)


def mujoco_name(m, i):
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)


if __name__ == "__main__":
    main()
