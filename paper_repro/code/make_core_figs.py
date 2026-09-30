"""Core-evidence figures: the sealed 142-cell matrix plus the local/Viper corpus.

Sources, all under paper_repro/data/
  sealed      sealed/nexus_matrix_review_2026-09-08_18-28.zip
              (final_metrics.csv -> core block, probes.csv -> selector ablation)
  ladder      logs/hopper_ladder.csv
              Each success_tail is the mean of the last 10% of
              policy_diag/primary_success_rate from that checkpoint.
  OOS         logs/oos/rt_<condition>_<meta>_v2_s<seed>.csv

Emits into figures/:
  fig_core_sealed.pdf   matched-budget method comparison, 2 environments, n=5
  fig_selector.pdf      Q3: learned mask vs no mask vs symbolic-only selector
  fig_ladder.pdf        Hopper budget ladder 1x-32x, flat denominator from Viper
  fig_oos.pdf           rough-terrain command shifts, n=30

Prints every number quoted in the text, plus exact permutation p-values.
"""
from __future__ import annotations

import collections
import csv
import io
import math
import re
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIGURES, LOGS, ZIP
from plot_style import COL, apply, grid, shade_blocks, spread

OUT = FIGURES
SUCC = "primary_success_rate"

apply()


def sep_p(a, b):
    """One-sided exact permutation p when min(a) > max(b): 1 / C(na+nb, na)."""
    if min(a) <= max(b):
        return None
    return 1.0 / math.comb(len(a) + len(b), len(a))


def rng(v):
    return f"{min(v):.4f}-{max(v):.4f}"


# ------------------------------------------------------------------- sealed
def sealed():
    z = zipfile.ZipFile(ZIP)

    def rows(name):
        return list(csv.DictReader(io.StringIO(
            z.read(f"nexus_matrix_review/{name}").decode("utf-8"))))

    core = collections.defaultdict(list)
    for r in rows("final_metrics.csv"):
        if r["metric"] != SUCC or not r["run_id"].startswith("core__"):
            continue
        _, task, method, _s = r["run_id"].split("__")
        core[(task, method)].append(float(r["value"]))

    probes = collections.defaultdict(list)
    for r in rows("probes.csv"):
        if r["metric"] != SUCC:
            continue
        _, task, method, _s = r["run_id"].split("__")
        probes[(task, method, r["condition"])].append(float(r["value"]))
    return {k: sorted(v) for k, v in core.items()}, {k: sorted(v) for k, v in probes.items()}


CORE, PROBES = sealed()


# --------------------------------------------------------------- local+Viper
def ladder():
    """{(meta, budget_multiple): [success]} for the Hopper ladder.

    Each value is the mean of the last 10% of policy_diag/primary_success_rate
    from the matching hopper checkpoint. Stems and source folders are in
    data/logs/hopper_ladder.csv.
    """
    out = collections.defaultdict(list)
    with (LOGS / "hopper_ladder.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["meta"], int(row["budget_multiple"]))].append(float(row["success_tail"]))
    return {k: sorted(v) for k, v in out.items()}


LADDER = ladder()


# ----------------------------------------------------------------------- OOS
def oos():
    out = collections.defaultdict(list)
    for p in sorted((LOGS / "oos").glob("rt_*_v2_s*.csv")):
        m = re.match(r"rt_(\w+?)_(flat|nesy|neural)_v2_s\d+$", p.stem)
        if not m:
            continue
        r = list(csv.DictReader(open(p, encoding="utf-8")))
        if r:
            out[(m.group(1), m.group(2))].append(float(r[0][SUCC]))
    return {k: sorted(v) for k, v in out.items()}


OOS = oos()


def fig_core():
    order = ["flat", "hpqn", "symbolic", "nesy", "neural", "ppo"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.95), gridspec_kw={"wspace": .30})
    for ax, task, title in ((axes[0], "hopper", "Hopper"),
                            (axes[1], "go1", "Go1")):
        for i, meth in enumerate(order):
            v = CORE[(task, meth)]
            spread(ax, len(order) - 1 - i, v, COL[meth])
        ax.set_yticks(range(len(order))[::-1])
        ax.set_yticklabels(order)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("primary success rate ($n=5$)")
        ax.set_title(title, loc="left")
        grid(ax)
    fig.savefig(OUT / "fig_core_sealed.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_selector():
    """Hopper and Go1. The condition is the legend, so the y-axis only names the environment."""
    envs = (("hopper", "Hopper"), ("go1", "Go1"))
    conds = (
        ("native", "as trained", "nesy", 0.26),
        ("selector_unmasked", "mask removed", "constant", 0.0),
        ("selector_symbolic", "symbolic only", "symbolic", -0.26),
    )
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    shade_blocks(ax, [1, 1])
    ax.set_ylim(-0.55, len(envs) - 0.45)
    for i, (task, pretty) in enumerate(envs):
        y = len(envs) - 1 - i
        for key, _label, color, dy in conds:
            vals = PROBES[(task, "nesy", key)]
            spread(ax, y + dy, vals, COL[color])
            print(f"{pretty} {key} n={len(vals)} {min(vals):.3f}--{max(vals):.3f}")
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([pretty for _task, pretty in envs][::-1])
    ax.set_xlim(-0.03, .62)
    ax.set_xlabel("primary success")
    grid(ax)
    handles = [plt.Line2D([], [], color=COL[color], lw=4, alpha=.55, label=label)
               for _key, label, color, _dy in conds]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 0.02), handlelength=1.0, columnspacing=.6,
               fontsize=7)
    fig.subplots_adjust(bottom=0.34, left=0.16, right=0.98, top=0.96)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_selector.pdf")
    plt.close(fig)


def fig_ladder():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    mult = [1, 2, 4, 8, 16, 32]
    for meta, dx in (("flat", -0.10), ("nesy", 0.0), ("neural", 0.10)):
        xs, ys = [], []
        for b in mult:
            v = LADDER.get((meta, b))
            if not v:
                continue
            x = math.log2(b) + dx
            spread(ax, x, v, COL[meta], horiz=False)
            xs.append(x - dx)
            ys.append(sum(v) / len(v))
        ax.plot([x + dx for x in xs], ys, color=COL[meta], lw=1.1, zorder=5, label=meta)
    for b in mult:
        v = LADDER.get(("flat", b))
        if v:
            ax.annotate(f"{len(v)}", (math.log2(b) - 0.10, -0.045), ha="center",
                        fontsize=7, color="#666666")
    ax.set_xticks([math.log2(b) for b in mult])
    ax.set_xticklabels([f"{b}$\\times$" for b in mult])
    ax.set_xlabel("training budget (1$\\times$ = 26.2M steps)")
    ax.set_ylabel("primary success rate")
    ax.set_ylim(-0.08, 0.62)
    grid(ax, axis="y")
    ax.legend(loc="upper left", frameon=False, handletextpad=.4, labelspacing=.25)
    fig.savefig(OUT / "fig_ladder.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_oos():
    conds = [("indist", "train\ndistr."), ("cmd00", "stop"),
             ("cmd15", "1.5 m/s"), ("cmd20", "2.0 m/s"), ("flat0", "flat")]
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    shade_blocks(ax, [1] * len(conds), vertical=True)
    ax.set_xlim(-0.55, len(conds) - 0.45)
    for i, (c, lab) in enumerate(conds):
        for meta, dx in (("flat", -0.24), ("nesy", 0.0), ("neural", 0.24)):
            v = OOS.get((c, meta))
            if not v:
                continue
            spread(ax, i + dx, v, COL[meta], horiz=False)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([l for _, l in conds])
    ax.set_ylabel("primary success rate ($n=30$)")
    ax.set_ylim(-0.03, 1.03)
    grid(ax, axis="y")
    h = [plt.Line2D([], [], color=COL[m], lw=4, alpha=.55, label=m) for m in ("flat", "nesy", "neural")]
    ax.legend(handles=h, loc="upper right", frameon=False, handletextpad=.4, labelspacing=.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_oos.pdf", bbox_inches="tight")
    plt.close(fig)


def report():
    print("=== sealed core block ===")
    for task in ("hopper", "go1"):
        print(f" {task}")
        for meth in ("flat", "hpqn", "symbolic", "nesy", "neural", "ppo"):
            v = CORE[(task, meth)]
            print(f"   {meth:9s} n={len(v)} {rng(v)}")
        for h in ("nesy", "neural"):
            p = sep_p(CORE[(task, h)], CORE[(task, "flat")])
            print(f"   separation {h} > flat: " + (f"p={p:.4g}" if p else "OVERLAP"))
        p = sep_p(CORE[(task, "ppo")], CORE[(task, "neural")])
        print(f"   separation ppo > neural: " + (f"p={p:.4g}" if p else "OVERLAP"))

    print("\n=== selector ablation (nesy) ===")
    for task in ("hopper", "go1"):
        for c in ("native", "selector_unmasked", "selector_symbolic"):
            v = PROBES[(task, "nesy", c)]
            print(f" {task:7s} {c:18s} n={len(v)} {rng(v)}")
        p = sep_p(PROBES[(task, "nesy", "native")], PROBES[(task, "nesy", "selector_symbolic")])
        print(f" {task:7s} native > symbolic-only: " + (f"p={p:.4g}" if p else "OVERLAP"))

    print("\n=== hopper ladder ===")
    for meta in ("flat", "nesy", "neural"):
        for b in (1, 2, 4, 8, 16, 32):
            v = LADDER.get((meta, b))
            if v:
                print(f" {meta:7s} {b:2d}x n={len(v):2d} {rng(v)} mean={sum(v)/len(v):.4f}")
    flat_hi = sum([LADDER.get(("flat", b), []) for b in (8, 16, 32)], [])
    for b in (4, 8):
        v = LADDER.get(("neural", b), [])
        if v:
            p = sep_p(v, flat_hi)
            print(f" neural {b}x (n={len(v)}) vs flat 8-32x (n={len(flat_hi)}): "
                  + (f"p={p:.3g}" if p else "OVERLAP"))
    v = LADDER.get(("nesy", 8), [])
    if v:
        p = sep_p(v, LADDER.get(("flat", 8), []))
        print(f" nesy 8x vs flat 8x: " + (f"p={p:.3g}" if p else "OVERLAP"))

    print("\n=== OOS rough-terrain shifts ===")
    for c in ("indist", "cmd00", "cmd15", "cmd20", "flat0"):
        line = f" {c:8s}"
        for meta in ("flat", "nesy", "neural"):
            v = OOS.get((c, meta), [])
            line += f"  {meta}: n={len(v)} {rng(v) if v else '-'}"
        print(line)
        for h in ("nesy", "neural"):
            p = sep_p(OOS.get((c, h), [0]), OOS.get((c, "flat"), [1]))
            if p:
                print(f"          {h} > flat separated, p={p:.3g}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_core()
    fig_selector()
    fig_ladder()
    fig_oos()
    report()
