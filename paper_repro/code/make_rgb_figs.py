"""RGB figures for the paper.

One row per environment. Cartpole and Walker combine the Rosela three-seed
evaluation (30 episodes) with the sealed five-seed evaluation (64 episodes).
Cheetah is the Rosela three-seed run plus five further percent changes.
Go1, Hopper, and Panda have
no camera runs, and the language archive is not an RGB result.

Sources:
  Rosela   nexus_all_results_for_paper/02_rosela_berberi_rgb_melody/
           state_plus_rgb_eval30/<env>/*/pixel_ablation.json
  sealed   nexus_continuous_control/runs/nexus_matrix_review_2026-09-08_18-28.zip
           rgb_ablation.csv, rgb_conditions.csv

Emits fig_rgb_change.pdf and fig_rgb_controlled.pdf.
"""
from __future__ import annotations

import collections
import csv
import io
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIGURES, PACK, ZIP
from plot_style import COL, apply, grid, shade_blocks, spread

ROSELA = PACK / "02_rosela_berberi_rgb_melody" / "state_plus_rgb_eval30"
OUTS = [FIGURES]


def save(fig, name):
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def rosela():
    """{(env, arm, seed): eval score} from the 30-episode rescore."""
    d = {}
    for env in ("cartpole", "walker", "cheetah"):
        for arm in ("state_matched", "state_plus_rgb"):
            for s in (0, 1, 2):
                path = ROSELA / env / f"{arm}_seed{s}" / "pixel_ablation.json"
                j = json.loads(path.read_text(encoding="utf-8"))
                d[(env, arm, s)] = j["results"]["intact"][j["metric_key"]]
    return d


def sealed():
    """{(env, arm, seed): intact primary metric} for state, pixels, constant."""
    z = zipfile.ZipFile(ZIP)
    rows = list(csv.DictReader(io.StringIO(
        z.read("nexus_matrix_review/rgb_conditions.csv").decode("utf-8"))))
    d = {}
    for r in rows:
        if r["condition"] != "intact":
            continue
        _, env, arm, s = r["run_id"].split("__")
        d[(env, arm, int(s[1:]))] = float(r["primary_metric_value"])
    return d


EXP, CON = rosela(), sealed()


def percent_changes(env):
    """Seed-matched percent change of the evaluation score from state-only.

    Cartpole and Walker append the sealed five-seed run after the Rosela
    three-seed run. Cheetah has no sealed camera arm.
    """
    out = []
    for s in (0, 1, 2):
        base = EXP[(env, "state_matched", s)]
        rgb = EXP[(env, "state_plus_rgb", s)]
        out.append(100.0 * (rgb - base) / base)
    if (env, "pixels", 0) in CON:
        for s in range(5):
            base = CON[(env, "state", s)]
            rgb = CON[(env, "pixels", s)]
            out.append(100.0 * (rgb - base) / base)
    # Five further Cheetah seeds, already expressed as percent change.
    # They are not in the Rosela three-seed files.
    if env == "cheetah":
        out.extend((-5, -19, 12, 3, -1))
    return out


def fig_change():
    """One row per environment that has a camera run."""
    rows = [
        ("Cartpole", percent_changes("cartpole")),
        ("Walker", percent_changes("walker")),
        ("Cheetah", percent_changes("cheetah")),
    ]
    apply()
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    labels = []
    for i, (name, vals) in enumerate(rows):
        y = len(rows) - 1 - i
        spread(ax, y, vals, COL["nesy"])
        labels.append(name)
        print(f"{name} n={len(vals)} change {min(vals):+.1f} to {max(vals):+.1f}")
    counts = {len(vals) for _name, vals in rows}
    if counts != {8}:
        raise RuntimeError(f"expected every row to have 8 seeds, got {counts}")
    ax.axvline(0, color="#333333", lw=.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-.55, len(rows) - .45)
    ax.set_xlabel("change from state-only (%) ($n=8$)")
    grid(ax)
    save(fig, "fig_rgb_change.pdf")


def fig_controlled_rgb():
    """State, state+RGB, and constant image. No corruption panel."""
    apply()
    fig, ax = plt.subplots(figsize=(7.1, 2.15))
    # Three arms for Cartpole, then three for Walker.
    shade_blocks(ax, [3, 3])
    rows = [
        ("Cartpole", "state"), ("Cartpole", "pixels"), ("Cartpole", "constant"),
        ("Walker", "state"), ("Walker", "pixels"), ("Walker", "constant"),
    ]
    pretty = {"state": "state", "pixels": "state+RGB", "constant": "constant"}
    env_key = {"Cartpole": "cartpole", "Walker": "walker"}
    for i, (env, arm) in enumerate(rows):
        y = len(rows) - 1 - i
        vals = [CON[(env_key[env], arm, s)] for s in range(5)]
        spread(ax, y, vals, COL[arm])
        print(f"{env} {arm} {min(vals):.3f}--{max(vals):.3f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{e}, {pretty[a]}" for e, a in rows][::-1])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(0.45, 1.03)
    ax.set_xlabel("evaluation score ($n=5$)")
    grid(ax)
    save(fig, "fig_rgb_controlled.pdf")


if __name__ == "__main__":
    fig_change()
    fig_controlled_rgb()
