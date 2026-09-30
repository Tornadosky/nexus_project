"""Skill-usage PDFs and the six-environment training curves.

Skill names drop the leading index (``2_Optimal Locomotion`` becomes
``Optimal Locomotion``). Curves use each method's longest logged run.
Go1 and Hopper come from the sealed controlled matrix, which already
contains flat, NeSy, neural, symbolic, and PPO at one budget. The other
four environments come from the local training logs; PPO there is the
separate baseline progress file.
"""
from __future__ import annotations

import csv
import json
import re
import textwrap
import zipfile
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import FIGURES, PACK, ZIP
from plot_style import COL, apply, grid, spread

OUTS = [FIGURES]
METHODS = ("flat", "neural", "nesy", "symbolic", "ppo")
LOCAL_ENVS = ("CartpoleBalance", "CheetahRun", "PandaPickCube", "WalkerWalk")
PPO_PREFIX = {
    "CartpoleBalance": "cartpole_balance_ppo_s",
    "CheetahRun": "cheetah_run_ppo_s",
    "PandaPickCube": "panda_pick_cube_ppo_s",
    "WalkerWalk": "walker_walk_ppo_s",
}
CURVE_ENVS = (
    ("CartpoleBalance", "Cartpole"),
    ("CheetahRun", "Cheetah"),
    ("Go1JoystickFlatTerrain", "Go1"),
    ("HopperHop", "Hopper"),
    ("PandaPickCube", "Panda"),
    ("WalkerWalk", "Walker"),
)


def save(fig, name, tight=True):
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        if tight:
            fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
        else:
            fig.savefig(out / name)
    plt.close(fig)


def clean_skill(name):
    """Drop the ``N_`` index and turn underscores or CamelCase into words."""
    name = re.sub(r"^\d+_", "", name)
    name = name.replace("_", " ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def skill_index(name):
    return int(name.split("_", 1)[0])


def wrap_name(name, width=20):
    """Break a long skill name so a three-column row still has room for the bar."""
    return "\n".join(textwrap.wrap(name, width)) or name


def fig_skill_usage():
    """One PDF per environment, three columns: hand-written, initial, refined.

    The value is printed at the end of each bar. Width stays 7.1 in so the
    five PDFs stack at 0.98\\textwidth without rescaling the type.
    """
    path = (PACK / "03_anja_koroveshi_llm_github" / "analysis_llm" / "tables"
            / "skill_usage_all_conditions.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_env = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_env[row["env"]][row["condition"]].append(row)

    panels = (
        ("hand_written", "Hand-written", COL["hand"]),
        ("llm_initial", "Initial", COL["initial"]),
        ("llm_refined", "Refined", COL["refined"]),
    )
    envs = (
        ("CartpoleBalance", "Cartpole"),
        ("CheetahRun", "Cheetah"),
        ("WalkerWalk", "Walker"),
        ("HopperHop", "Hopper"),
        ("Go1JoystickFlatTerrain", "Go1"),
    )

    apply()
    for index, (env, pretty) in enumerate(envs):
        columns = []
        for cond, title, color in panels:
            skills = sorted(by_env[env][cond], key=lambda row: skill_index(row["skill"]))
            columns.append((
                title, color,
                [(clean_skill(row["skill"]), float(row["mean_usage_fraction"])) for row in skills],
            ))
        n = max(len(skills) for _title, _color, skills in columns)
        # Fixed canvas so every environment has the same width when stacked.
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 0.44 * n + 0.85), sharex=True)
        fig.subplots_adjust(left=0.16, right=0.99, bottom=0.14, top=0.80, wspace=1.05)
        for ax, (title, color, skills) in zip(axes, columns):
            labels = []
            for i, (label, val) in enumerate(skills):
                y = len(skills) - 1 - i
                ax.barh(y, val, height=0.62, color=color, zorder=3)
                text = f"{val:.2f}"
                # Long bars carry the number inside; short bars carry it just past the end.
                if val >= 0.78:
                    ax.text(val - 0.04, y, text, va="center", ha="right",
                            fontsize=6.5, color="white")
                else:
                    ax.text(val + 0.04, y, text, va="center", ha="left",
                            fontsize=6.5, color="#222222")
                labels.append(wrap_name(label))
                print(f"{pretty:10} {title:14} {label:40} {val:.3f}")
            ax.set_yticks(range(len(skills)))
            ax.set_yticklabels(labels[::-1])
            ax.set_xlim(0, 1.32)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_title(title, loc="left", color=color)
            grid(ax)
            ax.tick_params(axis="y", labelsize=6.5, pad=1)
        fig.text(0.01, 0.96, pretty, ha="left", va="top", fontsize=8)
        if index == len(envs) - 1:
            fig.text(0.55, 0.02, "usage fraction", ha="center", va="bottom", fontsize=8)
        save(fig, f"fig_skill_usage_{pretty}.pdf", tight=False)


def _traces(by_seed):
    """One sorted (step, value) trace per seed, including the step-0 evaluation.

    Dropping step 0 left a blank from the origin to the next checkpoint
    (about 3M steps on Go1, about 10M on the PPO logs).
    """
    traces = []
    for pts in by_seed.values():
        pts = sorted(pts)
        if len(pts) >= 2:
            traces.append(pts)
    return traces


def _band(ax, traces, method, xmax):
    """Draw a method through ``xmax`` steps. Seeds are aligned first.

    ``xmax`` is the shortest run in the environment. Every method is drawn
    up to that step, so a longer log is windowed and a shorter log is not
    trimmed further.
    """
    if not traces:
        print("  skip", method)
        return
    grid = np.linspace(min(t[0][0] for t in traces), xmax, 120)
    cols = []
    for pts in traces:
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        yi = np.interp(grid, xs, ys)
        # Keep the sample at this seed's own last step. A one-step slack
        # stops the endpoint from being dropped by rounding.
        yi[(grid < xs[0]) | (grid > xs[-1] + 1.0)] = np.nan
        cols.append(yi)
    arr = np.vstack(cols)
    mid = np.nanmean(arr, axis=0)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    xs = grid / 1e6
    ax.fill_between(xs, lo, hi, color=COL[method], alpha=.15, linewidth=0)
    ax.plot(xs, mid, color=COL[method], lw=1.3, label=method, zorder=3)
    finite = np.where(np.isfinite(mid))[0]
    print(f"  {method:9} n={len(traces)} "
          f"{xs[finite[0]]:.2f}M={mid[finite[0]]:.2f} -> "
          f"{xs[finite[-1]]:.1f}M={mid[finite[-1]]:.2f}")


def _longest_budgets():
    """Largest logged budget, at least 1M steps, for each local env and method."""
    path = PACK / "04_local_gpu_and_viper" / "curves_return.csv"
    best = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["env"] not in LOCAL_ENVS or row["metric"] != "env/returned_episode_returns":
                continue
            steps = int(row["total_timesteps"])
            if steps < 1_000_000:
                continue
            key = (row["env"], row["method"])
            best[key] = max(best.get(key, 0), steps)
    return best


def _local_curves(budgets):
    path = PACK / "04_local_gpu_and_viper" / "curves_return.csv"
    series = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["env"], row["method"])
            if budgets.get(key) != int(row["total_timesteps"]):
                continue
            if row["metric"] != "env/returned_episode_returns":
                continue
            series[key][row["seed"]].append((int(row["env_step"]), float(row["value"])))
    return series


def _ppo_curves():
    root = PACK / "04_local_gpu_and_viper" / "ppo_baseline"
    series = {}
    for env, prefix in PPO_PREFIX.items():
        by_seed = defaultdict(list)
        for path in sorted(root.glob(prefix + "*.progress.json")):
            if "shipped" in path.name:
                continue
            for row in json.loads(path.read_text(encoding="utf-8")):
                by_seed[path.stem].append((int(row["step"]), float(row["eval/episode_reward"])))
        series[env] = by_seed
    return series


def _sealed_curves():
    z = zipfile.ZipFile(ZIP)
    raw = z.read("nexus_matrix_review/curves.csv").decode("utf-8").splitlines()
    series = defaultdict(lambda: defaultdict(list))
    for line in raw[1:]:
        rid, step, metric, val = line.split(",", 3)
        if metric != "episode_return_mean":
            continue
        _block, task, method, seed = rid.split("__")
        if method not in METHODS:
            continue
        env = {"go1": "Go1JoystickFlatTerrain", "hopper": "HopperHop"}[task]
        series[(env, method)][seed].append((int(step), float(val)))
    return series


def fig_all_curves():
    """Six environments. Each line is that method's longest logged run."""
    print("scanning budgets")
    budgets = _longest_budgets()
    for key, steps in sorted(budgets.items()):
        print(f"  budget {key[0]:24} {key[1]:9} {steps}")
    local = _local_curves(budgets)
    ppo = _ppo_curves()
    sealed = _sealed_curves()

    apply()
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.15),
                             gridspec_kw={"hspace": .48, "wspace": .28})
    for ax, (env, pretty) in zip(axes.ravel(), CURVE_ENVS):
        print(pretty)
        source = sealed if env in ("Go1JoystickFlatTerrain", "HopperHop") else local
        traces = {}
        for method in METHODS:
            raw = ppo[env] if method == "ppo" and env not in (
                "Go1JoystickFlatTerrain", "HopperHop") else source[(env, method)]
            traces[method] = _traces(raw)
        ends = [max(t[-1][0] for t in traces[m]) for m in METHODS if traces[m]]
        xmax = min(ends)
        print(f"  axis to {xmax / 1e6:.1f}M")
        for method in METHODS:
            _band(ax, traces[method], method, xmax)
        ax.set_title(pretty, loc="left")
        ax.set_xlabel("environment steps (M)")
        # A small pad keeps the last point inside the axes instead of on the spine.
        ax.set_xlim(0, (xmax / 1e6) * 1.03)
        grid(ax, axis="y")
    axes[0, 0].set_ylabel("episode return")
    axes[1, 0].set_ylabel("episode return")
    axes[2, 0].set_ylabel("episode return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.4, columnspacing=1.0)
    save(fig, "fig_all_curves.pdf")


def fig_llm_appendix():
    """Reward spread beside skill-set size. Same colors as the other language plots.

    Reward is min--max over seeds, with one dot per seed. Refined reward and
    every skill count are a single run. Skill counts are the size of the set,
    not a sample.
    """
    tables = PACK / "03_anja_koroveshi_llm_github" / "analysis_llm" / "tables"
    seeds = list(csv.DictReader((tables / "per_seed_results.csv").open(encoding="utf-8")))
    curve = list(csv.DictReader((tables / "refinement_curve.csv").open(encoding="utf-8")))
    sizes = list(csv.DictReader((tables / "skillset_sizes.csv").open(encoding="utf-8")))
    reward = defaultdict(list)
    for row in seeds:
        reward[(row["env"], row["condition"])].append(float(row["env_reward_mean"]))
    refined = {}
    for row in curve:
        if int(row["iteration"]) == 3:
            refined[row["env"]] = [float(row["env_reward_mean"])]
    count = {row["env"]: row for row in sizes}
    envs = (
        ("CartpoleBalance", "Cartpole"),
        ("CheetahRun", "Cheetah"),
        ("WalkerWalk", "Walker"),
        ("HopperHop", "Hopper"),
        ("Go1JoystickFlatTerrain", "Go1"),
    )
    arms = (
        ("hand_written", "hand_written_skills", "hand", "Hand-written", 0.26),
        ("llm_initial", "llm_initial_skills", "initial", "Initial", 0.0),
        ("refined", "llm_refined_skills", "refined", "Refined", -0.26),
    )
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45),
                             gridspec_kw={"width_ratios": [1.25, 1], "wspace": .32})
    ax = axes[0]
    for i, (env, _pretty) in enumerate(envs):
        y = len(envs) - 1 - i
        for cond, _col, color, _label, dy in arms:
            vals = refined[env] if cond == "refined" else reward[(env, cond)]
            spread(ax, y + dy, vals, COL[color])
            print(f"reward {env:24} {cond:14} n={len(vals)} {min(vals):.4f}--{max(vals):.4f}")
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([p for _e, p in envs][::-1])
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("reward per step")
    ax.set_title("Environment reward", loc="left")
    grid(ax)

    ax = axes[1]
    width = 0.24
    for j, (_cond, col, color, label, _dy) in enumerate(arms):
        xs = [i + (j - 1) * width for i in range(len(envs))]
        heights = [int(count[env][col]) for env, _pretty in envs]
        ax.bar(xs, heights, width=width, color=COL[color], label=label, zorder=3)
        for x, height in zip(xs, heights):
            print(f"skills {label:14} x={x:.2f} n={height}")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([p for _e, p in envs], rotation=25, ha="right")
    ax.set_ylabel("number of skills")
    ax.set_ylim(0, 6.4)
    ax.set_title("Skill count", loc="left")
    grid(ax, axis="y")
    handles = [plt.Line2D([], [], color=COL[c], lw=4, alpha=.55, label=lab)
               for _cond, _col, c, lab, _dy in arms]
    fig.subplots_adjust(top=0.82)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.98), handlelength=1.2, columnspacing=1.1)
    save(fig, "fig7_llm_appendix.pdf")


def fig_llm_bars():
    """Task success on the left, environment reward on the right.

    Dots are seeds and the bar is min--max. Hand-written and initial arms
    have five seeds. Refined is the last feedback iteration, one run.
    Alternating bands keep each environment's three arms together.
    The same figure replaces the older two-panel language plot.
    """
    tables = PACK / "03_anja_koroveshi_llm_github" / "analysis_llm" / "tables"
    seeds = list(csv.DictReader((tables / "per_seed_results.csv").open(encoding="utf-8")))
    curve = list(csv.DictReader((tables / "refinement_curve.csv").open(encoding="utf-8")))
    reward = defaultdict(list)
    success = defaultdict(list)
    for row in seeds:
        key = (row["env"], row["condition"])
        reward[key].append(float(row["env_reward_mean"]))
        success[key].append(float(row["eval_success_rate"]))
    refined_reward, refined_success = {}, {}
    for row in curve:
        if int(row["iteration"]) != 3:
            continue
        refined_reward[row["env"]] = [float(row["env_reward_mean"])]
        refined_success[row["env"]] = [float(row["primary_success_rate"])]

    envs = (
        ("CartpoleBalance", "Cartpole"),
        ("CheetahRun", "Cheetah"),
        ("Go1JoystickFlatTerrain", "Go1"),
        ("HopperHop", "Hopper"),
        ("WalkerWalk", "Walker"),
    )
    arms = (
        ("hand_written", "hand", "Hand-written", 0.26),
        ("llm_initial", "initial", "Initial", 0.0),
        ("refined", "refined", "Refined", -0.26),
    )
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True,
                             gridspec_kw={"wspace": .16})
    panels = (
        (axes[0], success, refined_success, "Task success", "primary success"),
        (axes[1], reward, refined_reward, "Environment reward", "reward per step"),
    )
    for ax, series, refined, title, xlabel in panels:
        for i, (env, _pretty) in enumerate(envs):
            y = len(envs) - 1 - i
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f0f0f0", zorder=0, linewidth=0)
            if i < len(envs) - 1:
                ax.axhline(y - 0.5, color="#c8c8c8", lw=.6, ls=(0, (2, 2)), zorder=1)
            for cond, color, _label, dy in arms:
                vals = refined[env] if cond == "refined" else series[(env, cond)]
                spread(ax, y + dy, vals, COL[color])
                print(f"{title:20} {env:24} {cond:12} n={len(vals)} "
                      f"{min(vals):.4f}--{max(vals):.4f}")
        ax.set_ylim(-0.55, len(envs) - 0.45)
        ax.set_xlim(-0.03, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        grid(ax)
    axes[0].set_yticks(range(len(envs)))
    axes[0].set_yticklabels([p for _e, p in envs][::-1])
    handles = [plt.Line2D([], [], color=COL[c], lw=4, alpha=.55, label=lab)
               for _cond, c, lab, _dy in arms]
    fig.subplots_adjust(top=0.82, wspace=0.16)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.98), handlelength=1.2, columnspacing=1.1)
    for name in ("fig_llm_bars.pdf", "fig4_language_ab.pdf"):
        for out in OUTS:
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    fig_skill_usage()
    fig_all_curves()
    fig_llm_bars()
