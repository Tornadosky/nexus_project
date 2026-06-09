#!/usr/bin/env python3
"""Render publication-quality (ICLR/NeurIPS-style) figures from a W&B project.

Pulls per-seed run histories, aggregates seeds into mean +/- std, and writes clean
matplotlib PNG/PDF figures. This is the offline, paper-grade counterpart to the live
W&B report -- it is what you would actually drop into a paper.

Usage:
  python tools/paper_figures_from_wandb.py --entity ENT --project PROJ --out runs/paper_figs
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# ----- paper style -------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

# Variant display order, labels, and a colorblind-friendly palette.
VARIANTS = ["nesy", "neural", "flat"]
LABEL = {"nesy": "NEXUS NeSy", "neural": "NEXUS neural", "flat": "Flat AC-PQN"}
COLOR = {"nesy": "#0072B2", "neural": "#D55E00", "flat": "#009E73"}

RETURN = "env/returned_episode_returns"
SUCCESS = "policy_diag/primary_success_rate"


def _ema(y: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def _seed_series(entity: str, project: str, metric: str):
    """Return {variant: list[(x, y)]} of per-seed histories."""
    runs = list(wandb.Api().runs(f"{entity}/{project}"))
    by_variant: dict[str, list[tuple[np.ndarray, np.ndarray]]] = collections.defaultdict(list)
    for r in runs:
        v = r.config.get("META_POLICY_TYPE")
        if v not in VARIANTS:
            continue
        h = r.history(keys=["env_step", metric], pandas=True).dropna(subset=["env_step", metric])
        if len(h) < 5:
            continue
        x = h["env_step"].to_numpy(float)
        y = h[metric].to_numpy(float)
        order = np.argsort(x)
        by_variant[v].append((x[order], y[order]))
    return by_variant


def _collect(entity: str, project: str, metric: str, n_grid: int = 240):
    """Return {variant: (grid, mean, sem, n_seeds)} on a common env_step grid.

    Band is the standard error of the mean across seeds (std / sqrt(n)) -- the
    uncertainty of the mean curve, the standard RL-paper choice.
    """
    by_variant = _seed_series(entity, project, metric)
    lo = min((x.min() for s in by_variant.values() for x, _ in s), default=0.0)
    hi = max((x.max() for s in by_variant.values() for x, _ in s), default=1.0)
    grid = np.linspace(lo, hi, n_grid)
    out = {}
    for v in VARIANTS:
        seeds = by_variant.get(v, [])
        if not seeds:
            continue
        resampled = np.vstack([_ema(np.interp(grid, x, y)) for x, y in seeds])
        n = len(seeds)
        sem = resampled.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(n_grid)
        out[v] = (grid, resampled.mean(0), sem, n)
    return out


def _final_window_stats(entity: str, project: str, metric: str, frac: float = 0.1):
    """Per-seed mean over the last ``frac`` of training, aggregated to mean +/- sem.

    Using a final window (not a single noisy last point) is what makes bimodal
    metrics like success rate stable enough for a bar chart.
    """
    by_variant = _seed_series(entity, project, metric)
    out = {}
    for v in VARIANTS:
        seeds = by_variant.get(v, [])
        if not seeds:
            continue
        per_seed = []
        for _x, y in seeds:
            k = max(1, int(len(y) * frac))
            per_seed.append(float(np.mean(y[-k:])))
        per_seed = np.asarray(per_seed)
        n = len(per_seed)
        sem = per_seed.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        out[v] = (float(per_seed.mean()), float(sem), n)
    return out


def fig_training_return(entity, project, out: Path, env_title: str) -> Path:
    data = _collect(entity, project, RETURN)
    n_seeds = max((d[3] for d in data.values()), default=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for v in VARIANTS:
        if v not in data:
            continue
        grid, mean, sem, n = data[v]
        c = COLOR[v]
        ax.plot(grid, mean, color=c, label=LABEL[v])
        ax.fill_between(grid, mean - sem, mean + sem, color=c, alpha=0.20, linewidth=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title(f"{env_title}: training return")
    ax.margins(x=0.01)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", title=f"mean $\\pm$ s.e.m. ({n_seeds} seeds)", title_fontsize=9)
    fig.tight_layout()
    path = out / "training_return.png"
    fig.savefig(path)
    fig.savefig(out / "training_return.pdf")
    plt.close(fig)
    return path


def _final_stats(entity, project, metric):
    runs = list(wandb.Api().runs(f"{entity}/{project}"))
    vals = collections.defaultdict(list)
    for r in runs:
        v = r.config.get("META_POLICY_TYPE")
        if v in VARIANTS and r.summary.get(metric) is not None:
            vals[v].append(float(r.summary[metric]))
    return {v: (np.mean(vals[v]), np.std(vals[v]), len(vals[v])) for v in VARIANTS if vals.get(v)}


def fig_final_bars(entity, project, out: Path, env_title: str) -> Path:
    ret = _final_window_stats(entity, project, RETURN)
    suc = _final_window_stats(entity, project, SUCCESS)
    n_seeds = max((s[2] for s in {**ret, **suc}.values()), default=0)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, stats, ylabel, title, ylim in [
        (axes[0], ret, "Final episode return", "Final return", None),
        (axes[1], suc, "Primary success rate", "Final success", (0, 1.05)),
    ]:
        xs = [v for v in VARIANTS if v in stats]
        means = [stats[v][0] for v in xs]
        sems = [stats[v][1] for v in xs]
        colors = [COLOR[v] for v in xs]
        bars = ax.bar(range(len(xs)), means, yerr=sems, capsize=5, color=colors,
                      width=0.62, edgecolor="black", linewidth=0.7,
                      error_kw=dict(elinewidth=1.1, capthick=1.1))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([LABEL[v] for v in xs], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        top = max((m + e for m, e in zip(means, sems)), default=1.0)
        ax.set_ylim(*(ylim if ylim else (0, top * 1.18)))
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        for b, m, e in zip(bars, means, sems):
            ax.annotate(f"{m:.2f}" if m < 10 else f"{m:.0f}",
                        (b.get_x() + b.get_width() / 2, b.get_height() + e),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 4), textcoords="offset points")
    fig.suptitle(f"{env_title}: final performance (last-10% window, error bars = s.e.m. over {n_seeds} seeds)",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    path = out / "final_performance.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(out / "final_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env-title", default="CartpoleBalance")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    p1 = fig_training_return(args.entity, args.project, args.out, args.env_title)
    p2 = fig_final_bars(args.entity, args.project, args.out, args.env_title)
    print("WROTE", p1)
    print("WROTE", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
