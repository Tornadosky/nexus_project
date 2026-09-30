"""Redraw the older paper PDFs in the shared style.

fig2  sealed core learning curves (episode return, flat / neural / NeSy)
fig3  Go1-rough seed spread plus action-noise means
fig4  language study: per-seed success and the refinement trajectory
fig5  single-skill removal, success retained versus the native policy

The previous PDFs had no generator. Numbers come from the sealed curve
table, the logged robustness CSVs, Anja's per-seed table, and the skill
probe files.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIGURES, PACK, ZIP
from plot_style import COL, apply, grid, shade_blocks, spread

OUTS = [FIGURES]
ENV_COLOR = {
    "Cartpole": COL["neural"],
    "Cheetah": COL["symbolic"],
    "Walker": COL["nesy"],
    "Hopper": COL["ppo"],
    "Go1": COL["hpqn"],
}


def save(fig, name):
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_curves():
    """Mean episode return of the sealed core runs, with a min--max band."""
    z = zipfile.ZipFile(ZIP)
    raw = z.read("nexus_matrix_review/curves.csv").decode("utf-8").splitlines()
    series = defaultdict(lambda: defaultdict(list))
    for line in raw[1:]:
        rid, step, metric, val = line.split(",", 3)
        if metric != "episode_return_mean":
            continue
        _block, task, method, seed = rid.split("__")
        # Step 0 is the untrained log and sits far above the Go1 curve.
        if int(step) == 0 or method not in ("flat", "neural", "nesy"):
            continue
        series[(task, method)][int(step)].append(float(val))

    apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.05), gridspec_kw={"wspace": .28})
    for ax, task, title in ((axes[0], "go1", "Go1"), (axes[1], "hopper", "Hopper")):
        for method in ("flat", "neural", "nesy"):
            steps = sorted(series[(task, method)])
            xs = [s / 1e6 for s in steps]
            lo = [min(series[(task, method)][s]) for s in steps]
            hi = [max(series[(task, method)][s]) for s in steps]
            mid = [sum(series[(task, method)][s]) / len(series[(task, method)][s])
                   for s in steps]
            ax.fill_between(xs, lo, hi, color=COL[method], alpha=.15, linewidth=0)
            ax.plot(xs, mid, color=COL[method], lw=1.4, label=method, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps (M)")
        ax.set_xlim(left=0)
        grid(ax, axis="y")
    axes[0].set_ylabel("episode return")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=.2)
    save(fig, "fig2_core_curves.pdf")


def _noise(prefix):
    """{perturbation: [primary success]} from the robustness CSVs."""
    root = PACK / "04_local_gpu_and_viper" / "robustness"
    acc = defaultdict(list)
    for path in sorted(root.glob(prefix + "_s*.csv")):
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if row.get("mode") not in ("", "action_noise"):
                continue
            acc[float(row["perturbation"])].append(float(row["primary_success_rate"]))
    return acc


def fig_reliability():
    """Go1-rough seeds at zero noise, then mean success as noise increases."""
    apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05),
                             gridspec_kw={"width_ratios": [1.05, 1.15, 1.15], "wspace": .38})
    ax = axes[0]
    arms = [("flat", "flat"), ("nesy", "nesy"), ("ppo", "ppo")]
    for i, (prefix, name) in enumerate(arms):
        vals = _noise("go1rough_" + prefix)[0.0]
        spread(ax, len(arms) - 1 - i, vals, COL[name])
        print(f"go1 rough {name} n={len(vals)} {min(vals):.3f}--{max(vals):.3f}")
    ax.set_yticks(range(len(arms))[::-1])
    ax.set_yticklabels([a[1] for a in arms])
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("primary success")
    ax.set_title("Go1 rough", loc="left")
    grid(ax)

    for ax, prefix, title in (
            (axes[1], "go1rough", "Go1, noise"),
            (axes[2], "hopper", "Hopper, noise")):
        for name in ("flat", "neural", "nesy"):
            acc = _noise(f"{prefix}_{name}")
            xs = sorted(acc)
            ys = [sum(acc[x]) / len(acc[x]) for x in xs]
            ax.plot(xs, ys, color=COL[name], lw=1.4, marker="o", ms=3.2,
                    label=name, zorder=3)
            print(f"{prefix} {name} means " +
                  " ".join(f"{y:.3f}" for y in ys))
        ax.set_title(title, loc="left")
        ax.set_xlabel("action-noise scale")
        ax.set_ylim(-0.01, 0.36)
        grid(ax, axis="y")
    axes[1].set_ylabel("primary success")
    axes[2].legend(frameon=False, loc="upper right", handlelength=1.2, labelspacing=.2)
    save(fig, "fig3_reliability.pdf")


def _language():
    """Per-seed success and reward ratio along the refinement trajectory."""
    tables = PACK / "03_anja_koroveshi_llm_github" / "analysis_llm" / "tables"
    seeds = list(csv.DictReader((tables / "per_seed_results.csv").open(encoding="utf-8")))
    curve = list(csv.DictReader((tables / "refinement_curve.csv").open(encoding="utf-8")))
    success = defaultdict(list)
    reward = defaultdict(list)
    for row in seeds:
        success[(row["env"], row["condition"])].append(float(row["eval_success_rate"]))
        reward[(row["env"], row["condition"])].append(float(row["env_reward_mean"]))
    refined = {}
    ratio = defaultdict(list)
    for row in curve:
        env = row["env"]
        hand = sum(reward[(env, "hand_written")]) / len(reward[(env, "hand_written")])
        value = float(row["env_reward_mean"])
        ratio[env].append(0.0 if hand == 0 else value / hand)
        if int(row["iteration"]) == 3:
            refined[env] = [float(row["primary_success_rate"])]
    return success, refined, ratio


def fig_language():
    """Success and reward side by side. Drawn in make_extra_figs.fig_llm_bars."""
    from make_extra_figs import fig_llm_bars
    fig_llm_bars()


def _fig_language_retired():
    """Earlier refinement-ratio panel. Kept so the old drawing is still readable."""
    success, refined, ratio = _language()
    order = [
        ("CartpoleBalance", "Cartpole"),
        ("CheetahRun", "Cheetah"),
        ("WalkerWalk", "Walker"),
        ("HopperHop", "Hopper"),
        ("Go1JoystickFlatTerrain", "Go1"),
    ]
    arms = (("hand_written", "hand", 0.26), ("llm_initial", "initial", 0.0),
            ("refined", "refined", -0.26))
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), gridspec_kw={"wspace": .32})
    ax = axes[0]
    for i, (env, pretty) in enumerate(order):
        y = len(order) - 1 - i
        for cond, color, dy in arms:
            vals = refined[env] if cond == "refined" else success[(env, cond)]
            spread(ax, y + dy, vals, COL[color])
            print(f"{pretty} {cond} n={len(vals)} {min(vals):.3f}--{max(vals):.3f}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([p for _e, p in order][::-1])
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("task success")
    ax.set_title("Success", loc="left")
    grid(ax)
    handles = [plt.Line2D([], [], color=COL[c], lw=4, alpha=.55, label=lab)
               for lab, c in (("hand", "hand"), ("initial", "initial"), ("refined", "refined"))]
    ax.legend(handles=handles, frameon=False, loc="lower right",
              handlelength=1.2, labelspacing=.2)

    ax = axes[1]
    for _env, pretty in order:
        env = _env
        ys = ratio[env]
        ax.plot(range(len(ys)), ys, color=ENV_COLOR[pretty], lw=1.4,
                marker="o", ms=3.2, label=pretty, zorder=3)
        print(pretty, "reward ratio", " ".join(f"{v:.2f}" for v in ys))
    ax.axhline(1, color="#333333", lw=.7, ls="--", zorder=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("feedback iteration")
    ax.set_ylabel("reward / hand-written")
    ax.set_title("Refinement", loc="left")
    ax.set_ylim(-0.05, 1.15)
    grid(ax, axis="y")
    # Below the axis, so the five environment names do not sit on the lines.
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.30, wspace=.32)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.73, 0.01), handlelength=1.2, columnspacing=.8,
               fontsize=7)
    save(fig, "fig4_language_ab.pdf")


def fig_removal():
    """Retained success after removing one skill. Orange marks the failed probe."""
    probes = json.loads((PACK / "04_local_gpu_and_viper" / "probes"
                         / "skill_probes.json").read_text(encoding="utf-8"))
    passed = {(r["env"], r["skill"]): r["matches_name"] for r in probes}
    rows = json.loads((PACK / "04_local_gpu_and_viper" / "probes"
                       / "skill_ablation.json").read_text(encoding="utf-8"))
    held = defaultdict(list)
    for row in rows:
        base = row["intact_success"]
        if base <= 0:
            raise RuntimeError(f"zero intact success for {row['checkpoint']}")
        held[(row["env"], row["removed_skill"])].append(row["success"] / base)

    order = [
        ("Go1JoystickFlatTerrain", "stand", "Go1 stand"),
        ("Go1JoystickFlatTerrain", "track_velocity", "Go1 track"),
        ("Go1JoystickFlatTerrain", "turn", "Go1 turn"),
        ("Go1JoystickFlatTerrain", "recover", "Go1 recover"),
        ("HopperHop", "stand_recover", "Hop stand"),
        ("HopperHop", "hop_forward", "Hop forward"),
        ("HopperHop", "stabilize_landing", "Hop landing"),
        ("HopperHop", "energy_efficient", "Hop efficient"),
    ]
    apply()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    # Four Go1 skills, then four Hopper skills.
    shade_blocks(ax, [4, 4])
    ax.set_ylim(-0.55, len(order) - 0.45)
    for i, (env, skill, pretty) in enumerate(order):
        y = len(order) - 1 - i
        vals = held[(env, skill)]
        color = COL["nesy"] if passed[(env, skill)] else COL["symbolic"]
        spread(ax, y, vals, color)
        print(f"{pretty} pass={passed[(env, skill)]} "
              f"{min(vals):.2f}--{max(vals):.2f}")
    ax.axvline(1, color="#333333", lw=.7, ls="--", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([p for _e, _s, p in order][::-1])
    ax.set_xlabel("success retained")
    grid(ax)
    handles = [
        plt.Line2D([], [], color=COL["nesy"], lw=4, alpha=.55, label="probe passes"),
        plt.Line2D([], [], color=COL["symbolic"], lw=4, alpha=.55, label="probe fails"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right",
              handlelength=1.2, labelspacing=.2)
    save(fig, "fig5_removal_only.pdf")


if __name__ == "__main__":
    fig_curves()
    fig_reliability()
    fig_language()
    fig_removal()
