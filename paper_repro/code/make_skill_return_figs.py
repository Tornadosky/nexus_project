"""Per-skill episodic returns for the hand-written NeSy skills.

Each line is one skill. The return is that skill's own reward summed over
the episode, including steps where another skill was selected. The band is
the seed min--max. Local environments are the longest NeSy run in verify.
Panda is the v2 family only. Go1 and Hopper are the sealed NeSy evaluations.
"""
from __future__ import annotations

import csv
import zipfile
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_extra_figs import _traces, clean_skill
from paths import FIGURES, LOGS, ZIP
from plot_style import apply, grid

OUTS = (FIGURES,)

# Skill index order, not the method legend. Names in the legend say which skill.
SKILL_COLOR = ("#1f77b4", "#ff7f0e", "#2ca02c", "#b07aa1")
LOCAL = {
    "CartpoleBalance": "cartpole_balance_nesy_budget4x_s",
    "CheetahRun": "cheetah_run_nesy_budget4x_s",
    "PandaPickCube": "panda_pick_cube_nesy_v2_s",
    "WalkerWalk": "walker_walk_nesy_dm_s",
}
ENVS = (
    ("CartpoleBalance", "Cartpole"),
    ("CheetahRun", "Cheetah"),
    ("Go1JoystickFlatTerrain", "Go1"),
    ("HopperHop", "Hopper"),
    ("PandaPickCube", "Panda"),
    ("WalkerWalk", "Walker"),
)


def _draw(ax, traces, color, label, xmax):
    """Mean line and seed min--max, aligned onto one step grid and cut at ``xmax``."""
    grid_x = np.linspace(min(t[0][0] for t in traces), xmax, 160)
    cols = []
    for pts in traces:
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        yi = np.interp(grid_x, xs, ys)
        yi[(grid_x < xs[0]) | (grid_x > xs[-1] + 1.0)] = np.nan
        cols.append(yi)
    arr = np.vstack(cols)
    mid = np.nanmean(arr, axis=0)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    xs = grid_x / 1e6
    ax.fill_between(xs, lo, hi, color=color, alpha=.15, linewidth=0)
    ax.plot(xs, mid, color=color, lw=1.3, label=label, zorder=3)
    finite = np.where(np.isfinite(mid))[0]
    print(f"    {label:22} n={len(traces)} {xs[finite[0]]:.2f}M={mid[finite[0]]:.1f}"
          f" -> {xs[finite[-1]]:.1f}M={mid[finite[-1]]:.1f}")


def _local():
    """skill -> seed -> [(step, return)] for the four local NeSy runs.

    The series are the ``skill_return/*`` logs from the verify checkpoints,
    written out as data/logs/skill_returns_local.csv. Stems are the checkpoint
    names: cartpole and cheetah budget4x, panda v2, walker dm.
    """
    out = {}
    by_skill = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    path = LOGS / "skill_returns_local.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_skill[row["env"]][row["skill"]][row["stem"]].append(
                (int(row["step"]), float(row["value"]))
            )
    for env in LOCAL:
        out[env] = by_skill[env]
        print(env, "seeds", len({s for sk in by_skill[env].values() for s in sk}),
              "skills", len(by_skill[env]))
    return out


def _sealed():
    """Go1 and Hopper NeSy eval curves from the sealed matrix."""
    raw = zipfile.ZipFile(ZIP).read("nexus_matrix_review/curves.csv").decode("utf-8").splitlines()
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    names = {"go1": "Go1JoystickFlatTerrain", "hopper": "HopperHop"}
    for line in raw[1:]:
        rid, step, metric, val = line.split(",", 3)
        if not metric.startswith("common_skill_return/"):
            continue
        _block, task, method, seed = rid.split("__")
        if method != "nesy":
            continue
        skill = metric.split("/", 1)[1]
        out[names[task]][skill][seed].append((int(step), float(val)))
    return out


def _sealed_methods(methods):
    """Go1 and Hopper eval curves for the requested methods.

    ``common_skill_return`` is the task's hand-written skill reward times
    episode length, scored on that policy's behavior. Flat and PPO have no
    skill actors; the lines are still the NeSy skill objectives.
    """
    raw = zipfile.ZipFile(ZIP).read("nexus_matrix_review/curves.csv").decode("utf-8").splitlines()
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    names = {"go1": "Go1JoystickFlatTerrain", "hopper": "HopperHop"}
    for line in raw[1:]:
        rid, step, metric, val = line.split(",", 3)
        if not metric.startswith("common_skill_return/"):
            continue
        _block, task, method, seed = rid.split("__")
        if method not in methods:
            continue
        skill = metric.split("/", 1)[1]
        out[names[task]][method][skill][seed].append((int(step), float(val)))
    return out


def _sealed_grid(methods, titles, name):
    """Go1 and Hopper, one column per method, scored on the NeSy skill rewards."""
    pretty = {"Go1JoystickFlatTerrain": "Go1", "HopperHop": "Hopper"}
    sealed = _sealed_methods(methods)
    apply()
    # Room on the right of each row for one shared skill legend. Colors follow
    # skill index, so the three panels in a row use the same legend.
    fig, axes = plt.subplots(2, 3, figsize=(6.80, 3.55), sharey="row")
    fig.subplots_adjust(left=0.08, right=0.84, top=0.92, bottom=0.11,
                        wspace=0.22, hspace=0.48)
    for row, env in enumerate(("Go1JoystickFlatTerrain", "HopperHop")):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            source = sealed[env][method]
            print(pretty[env], method)
            traces = {skill: _traces(by_seed) for skill, by_seed in source.items()}
            ends = [max(t[-1][0] for t in rows) for rows in traces.values() if rows]
            xmax = min(ends)
            print(f"  axis to {xmax / 1e6:.1f}M")
            for i, skill in enumerate(sorted(traces, key=lambda name: int(name.split("_", 1)[0]))):
                if not traces[skill]:
                    continue
                _draw(ax, traces[skill], SKILL_COLOR[i % len(SKILL_COLOR)], clean_skill(skill), xmax)
            ax.set_title(f"{pretty[env]}  {titles[method]}", loc="left", pad=2)
            ax.set_xlim(0, (xmax / 1e6) * 1.03)
            grid(ax, axis="y")
        axes[row, -1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5),
                             fontsize=6.0, handlelength=1.1, labelspacing=0.18,
                             handletextpad=0.3, borderaxespad=0.0)
    for ax in axes[1, :]:
        ax.set_xlabel("steps (M)")
    axes[0, 0].set_ylabel("skill return")
    axes[1, 0].set_ylabel("skill return")
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_skill_returns_baselines():
    """Go1 and Hopper. Flat, HPQN, and PPO scored on the NeSy skill rewards.

    Cartpole, Cheetah, Panda, and Walker are absent: their flat logs contain
    only ``skill_return/0_flat_actor``, and PPO logs episode reward, not these
    skill objectives. The sealed matrix is the only place the same quantity
    exists for a non-NeSy policy.
    """
    _sealed_grid(("flat", "hpqn", "ppo"),
                 {"flat": "flat", "hpqn": "HPQN", "ppo": "PPO"},
                 "fig_skill_returns_baselines.pdf")


def fig_skill_returns_variants():
    """Go1 and Hopper. Neural, symbolic, and NeSy on the same skill rewards.

    Same sealed ``common_skill_return`` series as the baselines figure. NeSy is
    repeated from ``fig_skill_returns.pdf`` so the three hierarchical variants
    sit side by side on one axis per row.
    """
    _sealed_grid(("neural", "symbolic", "nesy"),
                 {"neural": "neural", "symbolic": "symbolic", "nesy": "NeSy"},
                 "fig_skill_returns_variants.pdf")


def fig_skill_returns():
    """Six environments. NeSy only, one line per hand-written skill."""
    local = _local()
    sealed = _sealed()
    apply()
    # Two rows by three columns. Width matches a figure* at 0.95\textwidth.
    # Legends sit inside the empty corner of each panel.
    fig, axes = plt.subplots(2, 3, figsize=(6.80, 3.40))
    fig.subplots_adjust(left=0.09, right=0.988, top=0.94, bottom=0.11,
                        wspace=0.38, hspace=0.46)
    # Cheetah and Walker rise through the upper right. The other panels
    # leave the lower right open.
    legend_loc = {
        "CheetahRun": "upper left",
        "WalkerWalk": "upper left",
    }
    for ax, (env, pretty) in zip(axes.ravel(), ENVS):
        source = sealed[env] if env in sealed else local[env]
        print(pretty)
        traces = {skill: _traces(by_seed) for skill, by_seed in source.items()}
        ends = [max(t[-1][0] for t in rows) for rows in traces.values() if rows]
        xmax = min(ends)
        print(f"  axis to {xmax / 1e6:.1f}M")
        for i, skill in enumerate(sorted(traces, key=lambda name: int(name.split("_", 1)[0]))):
            if not traces[skill]:
                continue
            _draw(ax, traces[skill], SKILL_COLOR[i % len(SKILL_COLOR)], clean_skill(skill), xmax)
        ax.set_title(pretty, loc="left", pad=2)
        ax.set_xlim(0, (xmax / 1e6) * 1.03)
        grid(ax, axis="y")
        ax.legend(frameon=True, loc=legend_loc.get(env, "lower right"),
                  fontsize=6.0, handlelength=1.1, labelspacing=0.12,
                  handletextpad=0.3, borderaxespad=0.15,
                  framealpha=0.92, edgecolor="none")
    for ax in axes[1, :]:
        ax.set_xlabel("steps (M)")
    axes[0, 0].set_ylabel("skill return")
    axes[1, 0].set_ylabel("skill return")
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig_skill_returns.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    fig_skill_returns()
    fig_skill_returns_baselines()
    fig_skill_returns_variants()
