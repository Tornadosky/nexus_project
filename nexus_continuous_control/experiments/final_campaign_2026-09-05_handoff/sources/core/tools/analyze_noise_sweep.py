"""Analyse the exploration-schedule sweep and recommend a schedule for the V2 matrix.

Reads the 2x2 cartpole factorial (two cells from the V1.2 diagnostic, two from the sweep) plus
the Go1 transfer check, and reports:

  * the main effect of NOISE_START, the main effect of NOISE_FINISH, and their interaction;
  * whether the winning schedule transfers to Go1;
  * a bar chart with per-seed points, so a two-seed difference cannot masquerade as a result.

Usage
-----
    python tools/analyze_noise_sweep.py --out runs/sweep/noise
"""

from __future__ import annotations

import argparse
import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np


def finals(paths: list[Path], key: str = "rollout/episode_return") -> list[float]:
    out = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            ck = pickle.load(fh)
        metrics = ck["metrics"]
        if key not in metrics:
            continue
        r = np.asarray(metrics[key]).reshape(-1)
        out.append(float(np.mean(r[-max(1, len(r) // 10):])))
    return out


def collect(sweep: Path, verify: Path, diag: Path) -> dict:
    """The four factorial cells, wherever their runs happen to live."""
    return {
        # (NOISE_START, NOISE_FINISH) -> finals
        (0.30, 0.02): finals([verify / f"cartpole_balance_flat_s{s}.pkl" for s in (0, 1, 2)]),
        (0.30, 0.15): finals([sweep / f"cp_finish_only_s{s}.pkl" for s in (0, 1, 2)]),
        (1.00, 0.02): finals([sweep / f"cp_start_only_s{s}.pkl" for s in (0, 1, 2)]),
        (1.00, 0.15): finals([diag / f"flat_noise_only_s{s}.pkl" for s in (0, 1, 2)]),
    }


def plot(cells: dict, go1: dict, out: Path, go1_success: dict | None = None) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    go1_success = go1_success or {}
    have_go1 = any(v for v in go1.values())
    have_succ = any(v for v in go1_success.values())
    npanel = 1 + int(have_go1) + int(have_succ)
    fig, axes = plt.subplots(1, npanel, figsize=(5.5 * npanel, 4), dpi=160)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    labels, means, pts = [], [], []
    for start, fin in sorted(cells):
        vals = cells[(start, fin)]
        if not vals:
            continue
        labels.append(f"{start:g}→{fin:g}")
        means.append(np.mean(vals))
        pts.append(vals)
    x = np.arange(len(labels))
    colors = ["#8792a2" if lb == "0.3→0.02" else "#343C96" for lb in labels]
    colors[int(np.argmax(means))] = "#1B6B45"
    ax.bar(x, means, color=colors, width=0.62)
    for i, vals in enumerate(pts):
        ax.scatter([i] * len(vals), vals, color="#151A23", s=16, zorder=3, alpha=0.8)
    ax.axhline(787.4, ls="--", lw=1.2, color="#8A3227")
    ax.text(len(labels) - 0.45, 800, "upstream 787", color="#8A3227", fontsize=8, ha="right")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_xlabel("NOISE_START → NOISE_FINISH")
    ax.set_ylabel("final episode return")
    ax.set_title("CartpoleBalance flat — exploration factorial (dots = seeds)", fontsize=10, loc="left")
    ax.grid(alpha=0.18, lw=0.6, axis="y")

    def go1_panel(ax2, data: dict, ylabel: str, title: str) -> None:
        glabels, gmeans, gpts = [], [], []
        for k, v in data.items():
            if v:
                glabels.append(k)
                gmeans.append(np.mean(v))
                gpts.append(v)
        gx = np.arange(len(glabels))
        # Baseline grey, the rest coloured by whether they beat it — a losing arm must not
        # be able to render in the "winner" colour.
        base = gmeans[0] if gmeans else 0.0
        # Colour on SEED SEPARATION, not on the mean. With n=2 a mean difference is
        # routinely smaller than the seed range, and a green bar reads as a result. An arm
        # only earns a verdict colour when its seeds do not overlap the baseline's at all;
        # otherwise it stays neutral and the dots carry the story.
        base_pts = gpts[0] if gpts else []
        cols = ["#8792a2"]
        for vals in gpts[1:]:
            if base_pts and min(vals) > max(base_pts):
                cols.append("#1B6B45")
            elif base_pts and max(vals) < min(base_pts):
                cols.append("#8A3227")
            else:
                cols.append("#B0B7C3")  # overlapping seeds — inconclusive
        ax2.bar(gx, gmeans, color=cols[: len(glabels)], width=0.55)
        for i, vals in enumerate(gpts):
            ax2.scatter([i] * len(vals), vals, color="#151A23", s=16, zorder=3, alpha=0.8)
        if gmeans:
            ax2.axhline(base, ls="--", lw=1.0, color="#8792a2")
        ax2.set_xticks(gx, glabels, fontsize=8)
        ax2.set_ylabel(ylabel)
        ax2.set_title(title, fontsize=10, loc="left")
        ax2.grid(alpha=0.18, lw=0.6, axis="y")

    idx = 1
    if have_go1:
        go1_panel(
            axes[idx],
            go1,
            "final episode return",
            "Go1 flat — return (gameable; see success panel)",
        )
        idx += 1
    if have_succ:
        go1_panel(
            axes[idx],
            go1_success,
            "final tracking success rate",
            "Go1 flat — primary success (joystick tracking)",
        )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="runs/sweep/noise")
    ap.add_argument("--verify", default="runs/verify")
    ap.add_argument("--diag", default="runs/parity/diagnose")
    ap.add_argument("--out", default="runs/sweep/noise")
    args = ap.parse_args(argv)

    sweep, out = Path(args.sweep), Path(args.out)
    cells = collect(sweep, Path(args.verify), Path(args.diag))
    # Stage 3 adds the full upstream LOCOMOTION recipe (gamma 0.95 / lambda 0.99 /
    # noise 1.0->0.2), not just the noise knob. Reported on two metrics because Go1's
    # return is not the thing we care about: `primary_success_rate` is the joystick
    # tracking rate, and the campaign has already shown return can move without it.
    go1_runs = {
        "shipped 0.25→0.02": [sweep / f"go1_shipped_s{s}.pkl" for s in (0, 1)],
        "explore 1.0→0.15": [sweep / f"go1_explore_s{s}.pkl" for s in (0, 1)],
        "loco recipe": [sweep / f"go1_locorecipe_s{s}.pkl" for s in (0, 1)],
    }
    go1 = {k: finals(v) for k, v in go1_runs.items()}
    go1_success = {k: finals(v, "policy_diag/primary_success_rate") for k, v in go1_runs.items()}

    print("=" * 74)
    print("EXPLORATION SWEEP — CartpoleBalance flat, 2x2 factorial")
    print("=" * 74)
    print(f"{'START':>7} {'FINISH':>7} {'n':>3} {'mean':>9}   seeds")
    for start, fin in sorted(cells):
        v = cells[(start, fin)]
        if v:
            print(f"{start:7.2f} {fin:7.2f} {len(v):3d} {np.mean(v):9.1f}   {[round(x, 1) for x in v]}")

    complete = all(cells[k] for k in cells)
    summary = {"cells": {f"{a}_{b}": v for (a, b), v in cells.items()}, "go1": go1}
    if complete:
        m = {k: float(np.mean(v)) for k, v in cells.items()}
        eff_start = (m[(1.0, 0.02)] + m[(1.0, 0.15)]) / 2 - (m[(0.3, 0.02)] + m[(0.3, 0.15)]) / 2
        eff_fin = (m[(0.3, 0.15)] + m[(1.0, 0.15)]) / 2 - (m[(0.3, 0.02)] + m[(1.0, 0.02)]) / 2
        inter = (m[(1.0, 0.15)] - m[(1.0, 0.02)]) - (m[(0.3, 0.15)] - m[(0.3, 0.02)])
        print()
        print(f"  main effect of NOISE_START   {eff_start:+9.1f}")
        print(f"  main effect of NOISE_FINISH  {eff_fin:+9.1f}")
        print(f"  interaction                  {inter:+9.1f}")
        best = max(m, key=m.get)
        print(f"\n  best cell: START={best[0]:g} FINISH={best[1]:g}  ->  {m[best]:.1f}")
        summary |= {
            "effect_start": eff_start,
            "effect_finish": eff_fin,
            "interaction": inter,
            "best": list(best),
        }

    if any(go1.values()):
        print("\n" + "=" * 74)
        print("GO1 TRANSFER — return, and the tracking rate that actually matters")
        print("=" * 74)
        print(f"  {'arm':<22} {'n':>2}  {'return':>8}  seeds                {'success':>8}  seeds")
        for k, v in go1.items():
            if not v:
                continue
            s = go1_success.get(k, [])
            s_txt = f"{np.mean(s):8.4f}  {[round(x, 4) for x in s]}" if s else " " * 8 + "  (none)"
            print(f"  {k:<22} {len(v):2d}  {np.mean(v):8.2f}  {str([round(x, 2) for x in v]):<20} {s_txt}")

    summary["go1_success"] = go1_success
    plot(cells, go1, out / "noise_sweep.png", go1_success)
    (out / "noise_sweep.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out / 'noise_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
