"""Figures for the controlled (sealed 142-cell) study.

Reads the review pack directly so the numbers are traceable:
    nexus_continuous_control/runs/nexus_matrix_review_2026-09-08_18-28.zip

Emits into figures/:
    fig_controlled_llm.pdf     cheetah / walker seed spread
    fig_controlled_budget.pdf  pilot 13.1M vs final 52.4M, per family (budget-MISMATCHED)
    fig_controlled_rgb.pdf     state / pixels / constant primary success + pixel-drop probe

Also prints the dominant-skill fractions used in the LaTeX tables.
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

from paths import FIGURES, ZIP
from plot_style import COL, apply, grid, spread

OUT = FIGURES

apply()
HAND, INIT, REFI, RESA = COL["hand"], COL["initial"], COL["refined"], COL["resample"]


def read_pack():
    z = zipfile.ZipFile(ZIP)

    def rows(name):
        return list(csv.DictReader(io.StringIO(z.read(f"nexus_matrix_review/{name}").decode("utf-8"))))

    v2 = collections.defaultdict(dict)
    for r in rows("llm_v2_metrics.csv"):
        v2[r["run_id"]][r["metric"]] = float(r["value"])
    core = collections.defaultdict(dict)
    for r in rows("final_metrics.csv"):
        core[r["run_id"]][r["metric"]] = float(r["value"])
    rgb = rows("rgb_conditions.csv")
    abl = rows("rgb_ablation.csv")
    llm = json.loads(z.read("nexus_matrix_review/llm.json").decode("utf-8"))
    return v2, core, rgb, abl, llm


V2, CORE, RGBC, RGBA, LLM = read_pack()
SUCC = "primary_success_rate"


def hand(env):
    return [CORE[k][SUCC] for k in sorted(CORE) if k.startswith(f"llm_reference__{env}__")]


def cells(env, cond):
    """[(family, seed, success)] for one llm_final condition."""
    out = []
    for k in sorted(V2):
        if k.startswith(f"llm_final__{env}__{cond}__"):
            _, _, _, g, s = k.split("__")
            out.append((g, s, V2[k][SUCC]))
    return out


def pilots(env):
    return {k.split("__")[3]: V2[k][SUCC] for k in sorted(V2) if k.startswith(f"llm_pilot__{env}__")}


def spread_row(ax, y, vals, color, label=None):
    """min-max bar with one gold dot per seed."""
    spread(ax, y, vals, color)


def panel_env(ax, env, title, label_rows):
    """One environment. Rows are hand, initial, refined, resample."""
    conds = [("Hand", hand(env), HAND),
             ("Initial", [v for _, _, v in cells(env, "initial")], INIT),
             ("Refined", [v for _, _, v in cells(env, "refined")], REFI),
             ("Resample", [v for _, _, v in cells(env, "resample")], RESA)]
    for i, (name, vals, col) in enumerate(conds):
        spread_row(ax, len(conds) - 1 - i, vals, col)
        print(f"{title:8} {name:10} n={len(vals)} {min(vals):.3f}--{max(vals):.3f}")
    ax.set_xlim(-0.03, 1.03)
    ax.set_title(title, loc="left")
    grid(ax)
    if label_rows:
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([c[0] for c in conds][::-1])
    else:
        ax.tick_params(axis="y", left=False)


def fig_llm():
    """Column width, so an include at \\columnwidth keeps the 8 pt type."""
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True,
                             gridspec_kw={"wspace": .16})
    panel_env(axes[0], "cheetah", "Cheetah", True)
    panel_env(axes[1], "walker", "Walker", False)
    fig.supxlabel("primary success", fontsize=8)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.22, top=0.88, wspace=0.16)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_controlled_llm.pdf")
    plt.close(fig)


def fig_budget():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85), sharey=True,
                             gridspec_kw={"wspace": .12})
    for ax, env, title in ((axes[0], "cheetah", "Cheetah"), (axes[1], "walker", "Walker")):
        pil = pilots(env)
        for i, g in enumerate(sorted(pil)):
            fin = [v for gg, _s, v in cells(env, "initial") if gg == g]
            ax.plot([0, 1], [pil[g], sum(fin) / len(fin)], color="#bbbbbb", lw=1.0, zorder=1)
            ax.scatter([0], [pil[g]], s=18, color=COL["flat"], zorder=3)
            ax.scatter([1] * len(fin), fin, s=18, color=INIT, edgecolor="white",
                       linewidth=.4, zorder=3)
            if env == "walker":
                ax.annotate(g, (1, sum(fin) / len(fin)), textcoords="offset points",
                            xytext=(6, -2), fontsize=6.6, color="#444444")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pilot", "final"])
        ax.set_xlim(-.28, 1.55)
        ax.set_ylim(-0.06, 1.1)
        ax.set_title(title, loc="left")
        grid(ax, axis="y")
    axes[0].set_ylabel("primary success rate")
    fig.savefig(OUT / "fig_controlled_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rgb():
    """Kept as a one-panel copy. final_paper uses fig_rgb_controlled.pdf."""
    prim = collections.defaultdict(list)
    for r in RGBC:
        if r["condition"] != "intact":
            continue
        _, env, cond, _s = r["run_id"].split("__")
        prim[(env, cond)].append(float(r["primary_metric_value"]))

    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    cols = {"state": COL["state"], "pixels": COL["pixels"], "constant": COL["constant"]}
    rows_ = [("cartpole", "state"), ("cartpole", "pixels"), ("cartpole", "constant"),
             ("walker", "state"), ("walker", "pixels"), ("walker", "constant")]
    pretty = {"state": "state", "pixels": "state+RGB", "constant": "constant",
              "cartpole": "Cartpole", "walker": "Walker"}
    for y, (env, cond) in enumerate(reversed(rows_)):
        spread_row(ax, y, prim[(env, cond)], cols[cond])
    ax.set_yticks(range(len(rows_)))
    ax.set_yticklabels([f"{pretty[e]}, {pretty[c]}" for e, c in reversed(rows_)])
    ax.set_xlim(.45, 1.03)
    ax.set_xlabel("evaluation score ($n=5$)")
    grid(ax)
    fig.savefig(OUT / "fig_controlled_rgb.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def dominant_fractions():
    """max skill_usage per run -> the numbers quoted in the dominant-skill table."""
    def dom(metrics):
        u = [v for m, v in metrics.items() if m.startswith("skill_usage/")]
        return max(u) if u else None

    print("\ndominant-skill fraction (controlled study, Qwen3.8-27B, 52.4M)")
    for env in ("cheetah", "walker"):
        hv = [dom(CORE[k]) for k in sorted(CORE) if k.startswith(f"llm_reference__{env}__")]
        print(f"  {env:8s} hand      n={len(hv)} {min(hv):.2f}-{max(hv):.2f}")
        for cond in ("initial", "refined", "resample"):
            vs = [dom(V2[k]) for k in sorted(V2) if k.startswith(f"llm_final__{env}__{cond}__")]
            print(f"  {env:8s} {cond:9s} n={len(vs)} {min(vs):.2f}-{max(vs):.2f}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_llm()
    fig_budget()
    fig_rgb()
    dominant_fractions()
    print("\nwrote", OUT / "fig_controlled_llm.pdf", OUT / "fig_controlled_budget.pdf",
          OUT / "fig_controlled_rgb.pdf", sep="\n  ")
