"""Analysis figures for the PandaPickCube camera probe -- SCENE MEASUREMENTS, NOT TRAINING.

Everything here comes from `nexus_continuous/scripts/panda_pickcam_probe.py`, which
renders 8 production resets through the re-aimed `front` camera and measures the
image: is the cube visible, does its image position track its world position, how
far above the renderer's noise floor is the between-reset signal, and can a
*linear ridge probe* read the cube's (x, y) straight out of the raw 64x64x3 frame.

No panda policy has been trained. None of these panels is a learning curve, a
return, or a success rate -- they are properties of the rendered observation and
of a closed-form least-squares readout of it.

Reads the committed JSON (and, for the arm / mat mean colours only, the committed
64px PNGs). CPU-only, no MuJoCo, no rendering, no training.

    python tools/plot_panda_probe_figures.py --out results/rgb/panda_probe/figures
    python tools/plot_panda_probe_figures.py --figures ghost,summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PALETTE = ["#C44E52", "#55A868", "#4C72B0", "#8172B2", "#CCB974", "#64B5CD"]
RED, GREEN, BLUE, PURPLE, SAND, CYAN = PALETTE
GREY = "#8C8C8C"

# The probe's own colour rules (panda_pickcam_probe.py), in 0-255 units, reused
# here so the arm / mat statistics are measured the same way the JSON's
# `arm_pixels_per_seed_home_pose` was.
ARM_MIN_BRIGHT = 0.5 * 255      # panda shells are bright ...
ARM_MAX_SAT = 0.10 * 255        # ... and achromatic
CUBE_MIN_SAT = 0.25 * 255       # cube / mocap target are saturated

# hand-tuned label placement for the luminance/saturation scatters, whose points
# genuinely pile up (the mat and the 2px ring around the cube are the same colour)
LABEL_OFFSET = {                    # key -> (dx pt, dy pt, ha, va)
    "cube": (9, 13, "left", "bottom"),
    "ghost_red": (0, -15, "center", "top"),
    "ghost_green": (0, 15, "center", "bottom"),
    "ring": (14, 13, "left", "bottom"),
    "arm": (18, 0, "left", "center"),
    "mat": (14, -13, "left", "top"),
}

DISCLAIMER = ("MEASUREMENT OF THE RENDERED SCENE AND OF A LINEAR PIXEL PROBE -- "
              "NOT a training result. No panda policy has been trained yet: "
              "there is no return, no success rate and no learning curve on this figure.")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(root: Path) -> tuple[dict, dict | None]:
    base = json.loads((root / "metrics.json").read_text())
    rec_path = root / "ghost_recolour" / "metrics.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    if rec is None:
        print(f"[warn] {rec_path} missing -- green-ghost panels will be omitted")
    return base, rec


def scene_classes(root: Path, n_seeds: int):
    """Mean RGB / pixel count for arm shells and mat+floor, recomputed from the
    committed 64px PNGs because the probe JSON stores arm/mat *counts* but not
    their colours. Returns None if the PNGs are not there."""
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        print("[warn] Pillow missing -- arm / mat colour classes omitted")
        return None
    paths = [root / f"seed{i}_64.png" for i in range(n_seeds)]
    if not all(p.exists() for p in paths):
        print("[warn] seed*_64.png missing -- arm / mat colour classes omitted")
        return None
    im = np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32) for p in paths])
    sat = im.max(-1) - im.min(-1)
    arm = (im.min(-1) > ARM_MIN_BRIGHT) & (sat < ARM_MAX_SAT)
    mat = (im[..., 2] == im.max(-1)) & (im[..., 2] - im[..., 0] > 20)
    out = {}
    for name, m in (("arm", arm), ("mat", mat)):
        out[name] = dict(rgb=im[m].mean(0).tolist(), px=float(m.sum()) / n_seeds)
    return out


def lum(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def sat_of(rgb) -> float:
    return max(rgb) - min(rgb)


def banner(fig, extra: str = "", y: float = -0.035) -> None:
    txt = DISCLAIMER + (("  " + extra) if extra else "")
    fig.text(0.5, y, txt, ha="center", va="top", fontsize=8.2, color="#333333",
             wrap=True,
             bbox=dict(boxstyle="round", fc="#FFF7E6", ec="#D9C48A", lw=1.0))


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("wrote", p.resolve())
    return p


# --------------------------------------------------------------------------- #
# 1. cube localizability
# --------------------------------------------------------------------------- #
def fig_localizability(d, out: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    box = np.array(d["box_xyz"], float)
    cen = np.array(d["cube_centroid_col_row"], float)
    n = len(box)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.92]))

    pairs = [
        (axes[0], box[:, 1] * 100, cen[:, 0], d["corr_boxY_vs_imgCol"],
         d["img_col_range_px"], BLUE),
        (axes[1], box[:, 0] * 100, cen[:, 1], d["corr_boxX_vs_imgRow"],
         d["img_row_range_px"], PURPLE),
    ]
    for ax, w, i_, r, rng, color in pairs:
        s, b0 = np.polyfit(w, i_, 1)
        xs = np.linspace(w.min() - 2, w.max() + 2, 50)
        ax.plot(xs, s * xs + b0, color=GREEN, lw=2, zorder=1,
                label=f"least-squares fit\n{s:.2f} px/cm  ({1/abs(s):.2f} cm per px)")
        ax.scatter(w, i_, s=78, color=color, edgecolor="white", lw=1.2, zorder=3,
                   label=f"{n} production resets")
        resid = np.abs(i_ - (s * w + b0)).mean()
        ax.text(0.03, 0.965, f"r = {r:.4f}\nresidual = {resid:.2f} px\ntravel = {rng:.1f} px",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", alpha=0.95))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
        ax.grid(alpha=0.25, ls=":")
    axes[0].set_xlabel("cube world y  (cm, left-right)", fontsize=9.5)
    axes[0].set_ylabel("cube centroid, image COLUMN (px)", fontsize=9.5)
    axes[1].set_xlabel("cube world x  (cm, near-far)", fontsize=9.5)
    axes[1].set_ylabel("cube centroid, image ROW (px; larger = lower in frame)", fontsize=9.5)
    axes[0].set_title("world y maps to image column", fontsize=10.5)
    axes[1].set_title("world x maps to image row", fontsize=10.5)

    # ---- panel 3: where the cube actually lands inside the 64x64 frame ------ #
    ax = axes[2]
    ax.add_patch(plt.Rectangle((0, 0), 64, 64, fc="#EDF1F7", ec="#4C72B0", lw=1.5))
    ax.scatter(cen[:, 0], cen[:, 1], s=90, color=RED, edgecolor="white", lw=1.2,
               zorder=3, label=f"cube centroid ({n}/{n} resets visible)")
    for i, (c, r_) in enumerate(cen):
        ax.annotate(f"s{i}", (c, r_), textcoords="offset points", xytext=(7, 3),
                    fontsize=8, color="#444444")
    gp = np.array(d["gripper_site_px_col_row"], float)
    ax.scatter(gp[:1, 0], gp[:1, 1], marker="P", s=130, color=SAND,
               edgecolor="#7A6A2E", lw=1.0, zorder=3,
               label="gripper site at reset (home pose)")
    lo_c, hi_c = cen[:, 0].min(), cen[:, 0].max()
    lo_r, hi_r = cen[:, 1].min(), cen[:, 1].max()
    ax.add_patch(plt.Rectangle((lo_c, lo_r), hi_c - lo_c, hi_r - lo_r, fc="none",
                               ec=RED, ls="--", lw=1.2,
                               label=f"reset spread {hi_c-lo_c:.0f} x {hi_r-lo_r:.0f} px"))
    nb = sum(1 for b in d["cube_touches_border"] if b)
    ax.set_xlim(-2, 66)
    ax.set_ylim(66, -2)                       # image convention: row 0 on top
    ax.set_aspect("equal")
    ax.set_xlabel("image column (px)", fontsize=9.5)
    ax.set_ylabel("image row (px)", fontsize=9.5)
    ax.set_title(f"inside the 64x64 actor frame\n{nb}/{n} resets clip the border",
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.95)

    fig.suptitle("Cube localizability: the cube's image position is an almost perfectly "
                 "linear function of its world position\n"
                 f"colour-threshold centroid of the cube footprint, {n} production resets "
                 "of PandaPickCube through the actor camera",
                 fontsize=12.5, y=1.045)
    banner(fig, "Panels 1-2 are geometric measurements of the render; the centroid is "
                "read off by colour segmentation, not by a learned model.")
    fig.tight_layout()
    return save(fig, out, "fig1_cube_localizability.png")


# --------------------------------------------------------------------------- #
# 2. signal vs render noise floor
# --------------------------------------------------------------------------- #
def fig_noise_floor(d, rec, out: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    mad = d["pairwise_MAD_0_255"]
    bars = [
        ("render noise floor\n(no cube, no ghost)", mad["noise_floor(no cube,no ghost)"],
         GREY),
        ("cube only\n(ghost hidden)", mad["cube_only(no ghost)"], BLUE),
        ("production frame\ncube + RED ghost", mad["production_frame(cube+ghost)"], RED),
    ]
    if rec is not None:
        bars.append(("production frame\ncube + GREEN ghost",
                     rec["pairwise_MAD_0_255"]["production_frame(cube+ghost)"], GREEN))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                                  gridspec_kw=dict(width_ratios=[1.5, 1.0]))

    x = np.arange(len(bars))
    vals = [b[1]["mean"] for b in bars]
    lo = [b[1]["mean"] - b[1].get("min", b[1]["mean"]) for b in bars]
    hi = [b[1]["max"] - b[1]["mean"] for b in bars]
    ax.bar(x, vals, 0.62, color=[b[2] for b in bars],
           yerr=[lo, hi], capsize=6, ecolor="#444444", error_kw=dict(lw=1.2))
    # a 0.000 bar has no height -- draw its footprint so the slot does not read empty
    ax.plot([-0.31, 0.31], [0, 0], color=GREY, lw=7, solid_capstyle="butt", zorder=4)
    for xi, v, b in zip(x, vals, bars):
        rngtxt = (f"\n[{b[1].get('min', 0.0):.3f}, {b[1]['max']:.3f}]"
                  if b[1]["max"] > 0 else "\n(exactly 0)")
        ax.text(xi, b[1]["max"] + 0.10, f"{v:.3f}{rngtxt}", ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("mean abs. difference between two resets\n(0-255 per channel, "
                  "averaged over ALL 4096 px)", fontsize=9.5)
    ax.set_ylim(0, max(b[1]["max"] for b in bars) * 1.38)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_title("Whole-frame difference between resets\nvs the renderer's own "
                 "noise floor (bars = min/max over all reset pairs)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, ls=":")

    # ---- panel 2: the same signal, measured where it actually lives --------- #
    cube_px = float(np.mean(d["cube_pixels_per_seed"]))
    changed = d["pixels_changed_between_seed_pairs(cube_only)"]["mean"]
    per_changed = mad["cube_only(no ghost)"]["mean"] * 4096.0 / changed
    xx = np.arange(2)
    ax2.bar(xx, [0.0, per_changed], 0.55, color=[GREY, BLUE])
    ax2.plot([-0.275, 0.275], [0, 0], color=GREY, lw=7, solid_capstyle="butt", zorder=4)
    ax2.axhline(255, ls="--", lw=1.4, color="#333333")
    ax2.text(1.48, 250, "255 = full 8-bit range", ha="right", va="top", fontsize=8.5)
    ax2.text(0, 6, "0.000\n(deterministic\nrasteriser)", ha="center", va="bottom",
             fontsize=9)
    ax2.text(1, per_changed + 6, f"{per_changed:.0f}/255\n= {100*per_changed/255:.0f}% "
             "of full range", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(["render noise floor", "cube only\n(ghost hidden)"], fontsize=9)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(0, 300)
    ax2.set_ylabel("same difference, averaged over ONLY the\n~%.0f px that change "
                   "(0-255)" % changed, fontsize=9.5)
    ax2.set_title("Derived: %.3f x 4096 / %.1f\nthe cube's signal is near-maximal "
                  "where it exists" % (mad["cube_only(no ghost)"]["mean"], changed),
                  fontsize=10.5)
    ax2.grid(axis="y", alpha=0.25, ls=":")

    fig.suptitle("Signal vs render noise floor: small whole-frame numbers, "
                 "near-maximal local contrast\n"
                 "the cube covers only %.1f of 4096 px (%.2f%% of the frame), so any "
                 "whole-frame average is diluted ~%.0fx"
                 % (cube_px, 100 * cube_px / 4096, 4096 / changed),
                 fontsize=12.5, y=1.03)
    banner(fig, "Left panel is measured; right panel is the left panel's cube-only "
                "value rescaled by the measured number of changing pixels.")
    fig.tight_layout()
    return save(fig, out, "fig2_signal_vs_noise_floor.png")


# --------------------------------------------------------------------------- #
# 3. ghost confusion + the recolour fix  (headline)
# --------------------------------------------------------------------------- #
def fig_ghost(d, rec, out: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    if rec is None:
        print("[skip] ghost figure needs results/rgb/panda_probe/ghost_recolour/metrics.json")
        return None

    a, b = d["decodability"], rec["decodability"]
    chance = np.array(a["chance_mean_abs_err_m"]) * 100
    series_mae = [
        ("ridge, RED ghost (as shipped)", RED, np.array(a["ridge_mean_abs_err_m_[x,y]"]) * 100, None),
        ("ridge, GREEN ghost (recoloured)", GREEN, np.array(b["ridge_mean_abs_err_m_[x,y]"]) * 100, None),
        ("1-NN pixel retrieval, RED ghost", RED, np.array(a["1nn_mean_abs_err_m_[x,y]"]) * 100, "//"),
        ("1-NN pixel retrieval, GREEN ghost", GREEN, np.array(b["1nn_mean_abs_err_m_[x,y]"]) * 100, "//"),
    ]
    series_r2 = [
        (series_mae[0][0], RED, np.array(a["ridge_R2_[x,y]"]), None),
        (series_mae[1][0], GREEN, np.array(b["ridge_R2_[x,y]"]), None),
        (series_mae[2][0], RED, np.array(a["1nn_R2_[x,y]"]), "//"),
        (series_mae[3][0], GREEN, np.array(b["1nn_R2_[x,y]"]), "//"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.4),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0, 0.85]))
    x = np.arange(2)
    w = 0.19
    labels = ["cube x\n(near-far)", "cube y\n(left-right)"]

    ax = axes[0]
    for i, (lab, col, v, hatch) in enumerate(series_mae):
        off = (i - 1.5) * w
        bb = ax.bar(x + off, v, w, color=col, hatch=hatch, edgecolor="white",
                    lw=0.8, label=lab)
        for r_, vv in zip(bb, v):
            ax.text(r_.get_x() + r_.get_width() / 2, vv + 0.14, f"{vv:.1f}",
                    ha="center", va="bottom", fontsize=8)
    for i, c in enumerate(chance):
        ax.plot([x[i] - 0.42, x[i] + 0.42], [c, c], ls="--", lw=1.6, color="#333333",
                label="chance (predict the mean)" if i == 0 else None)
        ax.text(x[i] + 0.44, c, f" {c:.1f}", va="center", fontsize=8.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("mean abs. error of the linear probe (cm)", fontsize=9.5)
    ax.set_ylim(0, max(chance.max(), max(v.max() for *_x, v, _h in series_mae)) * 1.62)
    ax.set_title("Decode error -- lower is better\n(recolouring cuts ridge error ~40%)",
                 fontsize=10.5)
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.95, ncol=1)
    ax.grid(axis="y", alpha=0.25, ls=":")

    ax = axes[1]
    for i, (lab, col, v, hatch) in enumerate(series_r2):
        off = (i - 1.5) * w
        bb = ax.bar(x + off, v, w, color=col, hatch=hatch, edgecolor="white", lw=0.8,
                    label=lab)
        for r_, vv in zip(bb, v):
            ax.text(r_.get_x() + r_.get_width() / 2, vv + (0.035 if vv >= 0 else -0.09),
                    f"{vv:.2f}", ha="center",
                    va="bottom" if vv >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#333333", lw=1.2,
               label="R2 = 0: no better than predicting the mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("R2 of the linear probe on held-out resets", fontsize=9.5)
    ax.set_ylim(-1.05, 1.18)
    ax.set_title("Explained variance -- higher is better\n"
                 f"ridge on raw 64x64x3, {a['n_total']} resets / {a['n_test']} held out",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, ls=":")

    # ---- panel 3: WHY -- the distractor was the same colour, and bigger ----- #
    ax = axes[2]
    cube_rgb = d["cube_mean_rgb_0_255"]
    gr, gg = d["ghost_mean_rgb_0_255"], rec["ghost_mean_rgb_0_255"]
    dr = d["ghost_vs_cube_mean_abs_rgb_diff_0_255"]
    dg = rec["ghost_vs_cube_mean_abs_rgb_diff_0_255"]
    ax.bar([0, 1], [dr, dg], 0.5, color=[RED, GREEN])
    for xi, v in zip([0, 1], [dr, dg]):
        ax.text(xi, v + 5, f"{v:.1f}/255", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    for xi, ghost in zip([0, 1], [gr, gg]):
        for k, (rgbv, tag) in enumerate([(cube_rgb, "cube"), (ghost, "ghost")]):
            ax.add_patch(plt.Rectangle((xi - 0.26 + 0.28 * k, 222), 0.22, 28,
                                       fc=tuple(c / 255 for c in rgbv[:3]),
                                       ec="#333333", lw=1.0, clip_on=False))
            ax.text(xi - 0.15 + 0.28 * k, 217, tag, ha="center", va="top", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mocap ghost\nAS SHIPPED (red)", "mocap ghost\nRECOLOURED (green)"],
                       fontsize=9)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 268)
    ax.set_ylabel("mean |RGB| distance, cube vs mocap ghost (0-255)", fontsize=9.5)
    ax.set_title("Why: the distractor WAS the target\n"
                 "MJWarp ignores rgba alpha, so the 0.2-alpha\nmocap target renders "
                 "fully opaque -- and %d px vs %d px,\nit is %.1fx LARGER than the cube"
                 % (d["ghost_px_vs_cube_px"][0], d["ghost_px_vs_cube_px"][1],
                    d["ghost_px_vs_cube_px"][0] / max(d["ghost_px_vs_cube_px"][1], 1)),
                 fontsize=9.8)
    ax.grid(axis="y", alpha=0.25, ls=":")

    fig.suptitle("Ghost confusion and the fix: a same-coloured, larger mocap "
                 "\"ghost\" was masking the cube\n"
                 "linear ridge readout of cube (x, y) from the raw 64x64x3 frame -- "
                 "recolouring the mocap target lifts R2 from ~0.67 to ~0.86",
                 fontsize=12.5, y=1.045)
    banner(fig, "This is a closed-form least-squares probe on rendered frames, "
                "measuring what information the image contains. It is NOT a trained "
                "policy and says nothing yet about task success.")
    fig.tight_layout()
    return save(fig, out, "fig3_ghost_confusion_and_fix.png")


# --------------------------------------------------------------------------- #
# 4. contrast breakdown
# --------------------------------------------------------------------------- #
def fig_contrast(d, rec, cls, out: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    # (key, long label for the bar chart, short label for the scatter, rgb, px/frame)
    rows = [("cube", "cube (the target)", "cube", d["cube_mean_rgb_0_255"],
             float(np.mean(d["cube_pixels_per_seed"]))),
            ("ghost_red", "mocap ghost, red (as shipped)", "ghost (red)",
             d["ghost_mean_rgb_0_255"], float(d["ghost_px_vs_cube_px"][0]))]
    if rec is not None:
        rows.append(("ghost_green", "mocap ghost, green (fixed)", "ghost (green)",
                     rec["ghost_mean_rgb_0_255"], float(rec["ghost_px_vs_cube_px"][0])))
    rows.append(("ring", "2 px ring around the cube", "2 px ring\naround cube",
                 d["surround_mean_rgb_0_255"], None))
    omitted = []
    if cls is not None:
        rows.append(("arm", "panda arm (bright shells)", "panda arm",
                     cls["arm"]["rgb"], cls["arm"]["px"]))
        rows.append(("mat", "mat + floor (background)", "mat + floor",
                     cls["mat"]["rgb"], cls["mat"]["px"]))
    else:
        omitted.append("arm / mat")
    rows.append(("frame", "whole frame (mean)", None,
                 d["global_frame_mean_rgb_0_255"], 4096.0))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0, 0.72]))

    # ---- RGB channel means per scene class --------------------------------- #
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.26
    for k, (chname, col) in enumerate([("R", RED), ("G", GREEN), ("B", BLUE)]):
        vals = [r[3][k] for r in rows]
        bb = ax.bar(x + (k - 1) * w, vals, w, color=col, label=chname,
                    edgecolor="white", lw=0.6)
        for r_, vv in zip(bb, vals):
            ax.text(r_.get_x() + r_.get_width() / 2, vv + 4, f"{vv:.0f}",
                    ha="center", va="bottom", fontsize=7.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], fontsize=8.4, rotation=20, ha="right")
    ax.set_ylim(0, 300)
    ax.set_ylabel("mean channel value (0-255)", fontsize=9.5)
    ax.set_title("What each thing in the scene looks like, per channel", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9, ncol=3, title="channel", title_fontsize=8)
    ax.grid(axis="y", alpha=0.25, ls=":")
    keys = [r[0] for r in rows]
    i_cube, i_ring = keys.index("cube"), keys.index("ring")
    ax.annotate("", xy=(i_cube, 232), xytext=(i_ring, 232),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
    ax.text((i_cube + i_ring) / 2, 239, "cube vs its own surround:\n%.1f/255 mean abs. "
            "contrast" % d["cube_vs_surround_mean_abs_contrast_0_255"], ha="center",
            va="bottom", fontsize=8.6,
            bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))

    # ---- luminance / saturation split -------------------------------------- #
    ax = axes[1]
    for key, _long, short, rgb, px in rows:
        if short is None:               # whole-frame mean sits on top of the mat
            continue
        L, S = lum(rgb), sat_of(rgb)
        size = 90.0 if px is None else float(np.clip(38 * np.log1p(px), 60, 700))
        ax.scatter(S, L, s=size, color=tuple(c / 255 for c in rgb[:3]),
                   edgecolor="#333333", lw=1.3, zorder=3)
        dx, dy, ha, va = LABEL_OFFSET.get(key, (12, 8, "left", "bottom"))
        ax.annotate(short + ("" if px is None else f"\n{px:.0f} px"), (S, L),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                    fontsize=8.5, linespacing=1.25)
    ax.set_xlim(-25, 305)
    ax.set_ylim(-30, 275)
    ax.set_xlabel("saturation  max(RGB) - min(RGB)   (0-255)", fontsize=9.5)
    ax.set_ylabel("luminance  0.299R + 0.587G + 0.114B   (0-255)", fontsize=9.5)
    ax.set_title("Three-way split: bright+grey arm, dark+blue mat,\n"
                 "dim+saturated cube -- marker area ~ log(px per frame)", fontsize=10.5)
    ax.grid(alpha=0.25, ls=":")
    ax.axvline(CUBE_MIN_SAT, ls="--", lw=1.1, color=GREY)
    ax.text(CUBE_MIN_SAT + 5, -24, "probe's saturation gate (0.25)", fontsize=7.8,
            color="#555555")
    ax.axhline(ARM_MIN_BRIGHT, ls=":", lw=1.1, color=GREY)
    ax.text(-19, ARM_MIN_BRIGHT - 8, "probe's arm brightness gate (0.5)", fontsize=7.8,
            color="#555555", ha="left", va="top")

    # ---- per-seed contrast stability --------------------------------------- #
    ax = axes[2]
    per = d["cube_vs_surround_per_seed_0_255"]
    xs = np.arange(len(per))
    ax.bar(xs, per, 0.66, color=BLUE)
    m = d["cube_vs_surround_mean_abs_contrast_0_255"]
    ax.axhline(m, ls="--", lw=1.4, color="#333333")
    ax.text(-0.35, m + 4, f"mean {m:.1f}", ha="left", fontsize=8.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i}" for i in xs], fontsize=8)
    ax.set_ylim(0, max(per) * 1.28)
    ax.set_ylabel("cube vs surround contrast (0-255)", fontsize=9.5)
    ax.set_title("Stable across resets\n%.1f - %.1f over %d seeds"
                 % (min(per), max(per), len(per)), fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, ls=":")

    fig.suptitle("Contrast breakdown: the cube is the only dim, strongly saturated "
                 "thing in the frame\n"
                 "mean colours of each scene class in the 64x64 actor view "
                 "(8 production resets, home pose)", fontsize=12.5, y=1.04)
    src_note = ("Arm and mat+floor mean colours are recomputed from the committed "
                "64px PNGs with the probe's own colour rules (bright+achromatic = arm, "
                "blue-dominant = mat); all other values are read from metrics.json.")
    banner(fig, src_note if cls is not None
           else "Arm / mat panels omitted: seed PNGs not found.")
    fig.tight_layout()
    if omitted:
        print("[note] omitted scene classes:", ", ".join(omitted))
    return save(fig, out, "fig4_contrast_breakdown.png")


# --------------------------------------------------------------------------- #
# 5. combined summary
# --------------------------------------------------------------------------- #
def fig_summary(d, rec, cls, root: Path, out: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.2, 8.9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28,
                          height_ratios=[1.0, 1.0])

    box = np.array(d["box_xyz"], float)
    cen = np.array(d["cube_centroid_col_row"], float)

    # (a) the actual actor input ------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    shown = False
    try:
        from PIL import Image
        p = root / "seed1_64.png"
        if p.exists():
            ax.imshow(np.asarray(Image.open(p).convert("RGB")), interpolation="nearest")
            ax.scatter([cen[1, 0]], [cen[1, 1]], s=260, facecolor="none",
                       edgecolor="#FFD166", lw=2.0)
            ax.annotate("cube centroid", (cen[1, 0], cen[1, 1]),
                        textcoords="offset points", xytext=(14, -14), fontsize=8.5,
                        color="#FFD166",
                        arrowprops=dict(arrowstyle="->", color="#FFD166", lw=1.2))
            # the second red blob is the mocap target; locate it from the recoloured
            # run (same seeds, same mocap_xyz -- only the geom rgba differs)
            gp = root / "ghost_recolour" / "seed1_64.png"
            if gp.exists():
                gi = np.asarray(Image.open(gp).convert("RGB"), np.float32)
                gm = ((gi[..., 1] == gi.max(-1))
                      & (gi[..., 1] - np.maximum(gi[..., 0], gi[..., 2]) > CUBE_MIN_SAT))
                if gm.any():
                    ys, xs = np.nonzero(gm)
                    ax.annotate("mocap ghost\n(same red)", (xs.mean(), ys.mean()),
                                textcoords="offset points", xytext=(-14, -30),
                                ha="right", va="bottom", fontsize=8.5, color=CYAN,
                                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("(a) what the actor would see: 64x64x3\n"
                         "seed 1 -- cube %d px, red mocap ghost %d px"
                         % (d["cube_pixels_per_seed"][1], d["ghost_pixels_per_seed"][1]),
                         fontsize=10)
            shown = True
    except ImportError:
        pass
    if not shown:
        ax.axis("off")
        ax.text(0.5, 0.5, "seed1_64.png / Pillow\nnot available", ha="center",
                va="center", fontsize=10, color=GREY)

    # (b) localizability ---------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    w, i_ = box[:, 1] * 100, cen[:, 0]
    s, b0 = np.polyfit(w, i_, 1)
    xs = np.linspace(w.min() - 2, w.max() + 2, 40)
    ax.plot(xs, s * xs + b0, color=GREEN, lw=2, zorder=1)
    ax.scatter(w, i_, s=70, color=BLUE, edgecolor="white", lw=1.2, zorder=3)
    ax.text(0.03, 0.96, f"r = {d['corr_boxY_vs_imgCol']:.4f}\n{s:.2f} px/cm",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))
    ax.set_xlabel("cube world y (cm)", fontsize=9)
    ax.set_ylabel("image column (px)", fontsize=9)
    ax.set_title("(b) image position tracks world position\n"
                 f"also r = {d['corr_boxX_vs_imgRow']:.4f} for world x -> image row",
                 fontsize=10)
    ax.grid(alpha=0.25, ls=":")

    # (c) the ghost fix ------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 2])
    if rec is not None:
        a, b = d["decodability"], rec["decodability"]
        x = np.arange(2)
        wid = 0.26
        ch = np.array(a["chance_mean_abs_err_m"]) * 100
        for k, (lab, col, v) in enumerate([
                ("RED ghost", RED, np.array(a["ridge_mean_abs_err_m_[x,y]"]) * 100),
                ("GREEN ghost", GREEN, np.array(b["ridge_mean_abs_err_m_[x,y]"]) * 100)]):
            bb = ax.bar(x + (k - 0.5) * wid, v, wid, color=col, label=lab)
            for r_, vv in zip(bb, v):
                ax.text(r_.get_x() + r_.get_width() / 2, vv + 0.12, f"{vv:.1f}",
                        ha="center", va="bottom", fontsize=8)
        for i, c in enumerate(ch):
            ax.plot([x[i] - 0.32, x[i] + 0.32], [c, c], ls="--", lw=1.5, color="#333333",
                    label="chance" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(["cube x", "cube y"], fontsize=9)
        ax.set_ylim(0, ch.max() * 1.35)
        ax.set_ylabel("linear-probe error (cm)", fontsize=9)
        ax.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.95)
        ax.set_title("(c) recolouring the mocap ghost\nR2 %.2f/%.2f -> %.2f/%.2f"
                     % (*a["ridge_R2_[x,y]"], *b["ridge_R2_[x,y]"]), fontsize=10)
        ax.grid(axis="y", alpha=0.25, ls=":")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "ghost_recolour/metrics.json\nnot available", ha="center",
                va="center", fontsize=10, color=GREY)

    # (d) noise floor -------------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    mad = d["pairwise_MAD_0_255"]
    names = ["noise\nfloor", "cube\nonly", "cube +\nRED ghost"]
    vals = [mad["noise_floor(no cube,no ghost)"]["mean"],
            mad["cube_only(no ghost)"]["mean"],
            mad["production_frame(cube+ghost)"]["mean"]]
    cols = [GREY, BLUE, RED]
    if rec is not None:
        names.append("cube +\nGREEN ghost")
        vals.append(rec["pairwise_MAD_0_255"]["production_frame(cube+ghost)"]["mean"])
        cols.append(GREEN)
    ax.bar(np.arange(len(vals)), vals, 0.6, color=cols)
    ax.plot([-0.3, 0.3], [0, 0], color=GREY, lw=6, solid_capstyle="butt", zorder=4)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.06, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel("whole-frame MAD between resets\n(0-255)", fontsize=9)
    ax.set_title("(d) whole-frame signal is small, but the\nrenderer's noise floor is "
                 "exactly 0.000", fontsize=10)
    ax.grid(axis="y", alpha=0.25, ls=":")

    # (e) contrast ----------------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    cl = [("cube", "cube", d["cube_mean_rgb_0_255"]),
          ("ring", "2 px ring", d["surround_mean_rgb_0_255"])]
    if cls is not None:
        cl += [("arm", "panda arm", cls["arm"]["rgb"]),
               ("mat", "mat + floor", cls["mat"]["rgb"])]
    for key, name, rgb in cl:
        ax.scatter(sat_of(rgb), lum(rgb), s=190,
                   color=tuple(c / 255 for c in rgb[:3]), edgecolor="#333333", lw=1.3,
                   zorder=3)
        dx, dy, ha, va = LABEL_OFFSET.get(key, (11, 7, "left", "bottom"))
        ax.annotate(name, (sat_of(rgb), lum(rgb)), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=8.5)
    ax.set_xlim(-25, 290)
    ax.set_ylim(-25, 268)
    ax.set_xlabel("saturation (0-255)", fontsize=9)
    ax.set_ylabel("luminance (0-255)", fontsize=9)
    ax.set_title("(e) the cube is colour-separable\n%.1f/255 mean contrast vs its "
                 "surround" % d["cube_vs_surround_mean_abs_contrast_0_255"], fontsize=10)
    ax.grid(alpha=0.25, ls=":")

    # (f) the numbers -------------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    n = d["n_seeds"]
    cube_px = float(np.mean(d["cube_pixels_per_seed"]))
    lines = [
        ("cube visible", f"{n}/{n} resets, {sum(1 for b in d['cube_touches_border'] if b)}/{n} clipped", GREEN),
        ("cube size", f"{cube_px:.1f} of 4096 px ({100*cube_px/4096:.2f}% of frame)", BLUE),
        ("cube vs surround", f"{d['cube_vs_surround_mean_abs_contrast_0_255']:.1f}/255", GREEN),
        ("world -> image", f"r = {d['corr_boxY_vs_imgCol']:.3f} (y/col), "
                           f"{d['corr_boxX_vs_imgRow']:.3f} (x/row)", GREEN),
        ("render noise floor", f"{mad['noise_floor(no cube,no ghost)']['mean']:.3f}/255 "
                               "(deterministic)", GREEN),
        ("gripper in frame", "yes, all resets" if d["gripper_in_frame_all_seeds"] else "NO", GREEN),
    ]
    if rec is not None:
        lines.append(("linear probe, red ghost",
                      "MAE %.1f/%.1f cm, R2 %.2f/%.2f"
                      % (*[v * 100 for v in d['decodability']['ridge_mean_abs_err_m_[x,y]']],
                         *d['decodability']['ridge_R2_[x,y]']), RED))
        lines.append(("linear probe, GREEN ghost",
                      "MAE %.1f/%.1f cm, R2 %.2f/%.2f"
                      % (*[v * 100 for v in rec['decodability']['ridge_mean_abs_err_m_[x,y]']],
                         *rec['decodability']['ridge_R2_[x,y]']), GREEN))
        lines.append(("chance baseline",
                      "MAE %.1f/%.1f cm"
                      % tuple(v * 100 for v in d['decodability']['chance_mean_abs_err_m']),
                      GREY))
    ax.text(0.0, 1.03, "(f) the whole probe in numbers", fontsize=10.5,
            transform=ax.transAxes, va="bottom")
    top, bottom = 0.96, 0.15
    step = (top - bottom) / max(len(lines) - 1, 1)
    y = top
    for k, v, c in lines:
        ax.add_patch(plt.Rectangle((0.0, y - 0.028), 0.026, 0.048, fc=c, ec="none",
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.048, y, k, fontsize=8.8, transform=ax.transAxes, va="center",
                fontweight="bold")
        ax.text(0.048, y - step * 0.42, v, fontsize=8.4, transform=ax.transAxes,
                va="center", color="#333333")
        y -= step
    ax.text(0.0, bottom - step * 0.72,
            "NOT MEASURED YET: whether a policy trained on\nthese pixels can solve the "
            "task. That needs a run.", fontsize=8.6, transform=ax.transAxes,
            va="top", color="#8A3B3B",
            bbox=dict(boxstyle="round", fc="#FDF0F0", ec="#D9A0A0"))

    fig.suptitle("Can pixels localize the cube in PandaPickCube?  "
                 "Yes -- the information is in the 64x64 frame\n"
                 "camera / scene probe over %d production resets plus a linear ridge "
                 "readout on %d resets -- NO POLICY HAS BEEN TRAINED"
                 % (n, d["decodability"]["n_total"]), fontsize=13.5, y=0.995)
    banner(fig, "Panels (a), (b), (d), (e) describe the rendered image; (c) and (f) "
                "describe a closed-form least-squares readout of it.", y=0.028)
    return save(fig, out, "fig5_summary.png")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/rgb/panda_probe",
                    help="directory holding metrics.json + seed*_64.png")
    ap.add_argument("--out", default="results/rgb/panda_probe/figures",
                    help="output DIRECTORY for the PNGs")
    ap.add_argument("--figures", default="all",
                    help="comma list of localizability,noise,ghost,contrast,summary")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.axisbelow": True})

    here = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = here / root
    out = Path(args.out)
    if not out.is_absolute():
        out = here / out

    d, rec = load(root)
    cls = scene_classes(root, int(d["n_seeds"]))

    want = {w.strip() for w in args.figures.split(",") if w.strip()}
    do_all = "all" in want
    written = []
    if do_all or "localizability" in want:
        written.append(fig_localizability(d, out))
    if do_all or "noise" in want:
        written.append(fig_noise_floor(d, rec, out))
    if do_all or "ghost" in want:
        written.append(fig_ghost(d, rec, out))
    if do_all or "contrast" in want:
        written.append(fig_contrast(d, rec, cls, out))
    if do_all or "summary" in want:
        written.append(fig_summary(d, rec, cls, root, out))
    print(f"\n{len([w for w in written if w])} figure(s) in {out.resolve()}")


if __name__ == "__main__":
    main()
