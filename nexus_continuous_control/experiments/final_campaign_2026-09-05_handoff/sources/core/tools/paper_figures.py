#!/usr/bin/env python3
"""Paper-comparable figures: the Fig. 3 and Fig. 7 analogues from `docs/9847_From_Objects_to_Skills_In.pdf`.

Two figures, deliberately laid out the way the paper lays them out so a reader can put them
side by side.

**Fig. 3 analogue — per-skill returns, ONE PANEL PER SKILL, methods overlaid.**
The paper plots a panel for each skill (Rescue Divers, Shoot Enemies, Surface...) and overlays
every method, with a shaded seed band. That orientation answers the Q1 question directly: "does
this method learn *this* skill?" An earlier version of our plot was the transpose (one panel per
arm, one line per skill), which shows the same numbers but makes the method comparison — the
actual claim — impossible to read.

**HPQN is deliberately EXCLUDED from this figure, and that is a correction to a first draft.**
It was initially plotted as the paper plots its baselines, but the y-values are not the same
quantity: NEXUS's "stand_recover return" accumulates the hand-written *stand* reward, whereas
HPQN's accumulates the *env* reward (that is precisely what `SHARED_SKILL_REWARD` does). Drawing
them on one axis invites a comparison of two different measurements and would make HPQN look
like it "fails to learn the skill" when it was never optimising that reward at all.

Making HPQN comparable needs a trainer change — log `policy_module.skill_rewards` for
diagnostics while still *training* on the shared env reward — and a re-run. Until then the
figure states only what it can support: **both NEXUS variants learn every skill** (each skill's
own return rises from initialisation), which is the paper's Q1 claim for NEXUS itself. `flat`
has one actor and no per-skill decomposition, so it cannot appear here either.

**Fig. 7 analogue — game reward vs aligned goal.**
The paper's point is that return and the task's real objective can disagree, and that baselines
reward-hack. We measured that directly: on Go1, `flat` earns ~2x the environment return of
`neural` while tracking the commanded velocity ~1/10 as well. Left axis = episode return
(the gameable quantity), right axis = primary success (the behaviour we actually want), exactly
as the paper pairs Game Reward with Divers Rescued / Level Completion.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Method colours, chosen to echo the paper's legend (baselines grey/dark, NEXUS variants in
# blue / orange / teal).
METHOD_STYLE = {
    "HPQN (baseline)": ("#7A7A7A", "--"),
    "NEXUS (neural)": ("#1F77B4", "-"),
    "NEXUS (symbolic)": ("#F59F00", "-"),
    "NEXUS (nesy)": ("#0CA678", "-"),
}
FLAT_STYLE = ("#3D3D3D", ":")


def _curves(paths, key_prefix="skill_return/"):
    """-> (skill_names, array [seeds, skills, updates]) or None."""
    per_seed, names = [], None
    for p in paths:
        with open(p, "rb") as fh:
            m = (pickle.load(fh).get("metrics") or {})
        keys = sorted(k for k in m if k.startswith(key_prefix))
        if not keys:
            continue
        nm = [k.split("/", 1)[1] for k in keys]
        arr = np.stack([np.asarray(m[k]).reshape(-1) for k in keys])
        if names is None:
            names = nm
        elif nm != names:
            continue
        per_seed.append(arr)
    if not per_seed or names is None:
        return None
    n = min(a.shape[1] for a in per_seed)
    return names, np.stack([a[:, :n] for a in per_seed])


def fig3(env_specs, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for env_label, arms in env_specs:
        loaded = {}
        for method, pattern in arms.items():
            paths = sorted(Path().glob(pattern))
            c = _curves(paths)
            if c:
                loaded[method] = c
        if loaded:
            rows.append((env_label, loaded))
    if not rows:
        print("fig3: no per-skill data found")
        return

    ncol = max(len(next(iter(d.values()))[0]) for _, d in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.3 * ncol, 3.0 * len(rows)),
                             squeeze=False, dpi=150)
    for r, (env_label, loaded) in enumerate(rows):
        names = next(iter(loaded.values()))[0]
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(names):
                ax.axis("off")
                continue
            for method, (nm, arr) in loaded.items():
                col, ls = METHOD_STYLE.get(method, FLAT_STYLE)
                y = arr[:, c, :]                      # [seeds, updates]
                mu, sd = y.mean(0), y.std(0)
                x = np.arange(y.shape[1])
                ax.plot(x, mu, color=col, ls=ls, lw=1.5, label=method)
                if y.shape[0] > 1:
                    ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            ax.set_title(names[c].replace("_", " "), fontsize=10)
            ax.grid(alpha=0.18, lw=0.6)
            ax.set_xlabel("update")
            if c == 0:
                ax.set_ylabel(f"{env_label}\nSkill Returns", fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(
        "Per-skill returns — every skill's own reward rises from initialisation  "
        "(paper Fig. 3 analogue; HPQN omitted, see module docstring)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def _tail(pkl: Path, key: str):
    with open(pkl, "rb") as fh:
        m = (pickle.load(fh).get("metrics") or {})
    if key not in m:
        return None
    a = np.asarray(m[key]).reshape(-1)
    return float(a[-max(1, a.size // 10):].mean())


def fig7(env_specs, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = []
    for env_label, arms, goal_name in env_specs:
        rows = []
        for method, pattern in arms.items():
            paths = sorted(Path().glob(pattern))
            rets = [v for v in (_tail(p, "rollout/episode_return") for p in paths) if v is not None]
            sucs = [v for v in (_tail(p, "policy_diag/primary_success_rate") for p in paths)
                    if v is not None]
            if rets and sucs:
                rows.append((method, np.mean(rets), np.std(rets), np.mean(sucs), np.std(sucs)))
        if rows:
            envs.append((env_label, rows, goal_name))
    if not envs:
        print("fig7: no data")
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(5.6 * len(envs), 4.4), squeeze=False, dpi=150)
    for i, (env_label, rows, goal_name) in enumerate(envs):
        ax = axes[0][i]
        ax2 = ax.twinx()
        x = np.arange(len(rows))
        w = 0.38
        ax.bar(x - w / 2, [r[1] for r in rows], w, yerr=[r[2] for r in rows],
               color="#F3C9A0", edgecolor="#B5762F", capsize=3, label="episode return")
        ax2.bar(x + w / 2, [r[3] for r in rows], w, yerr=[r[4] for r in rows],
                color="#E58606", edgecolor="#8A4B04", capsize=3, label=goal_name)
        ax.set_xticks(x, [r[0].replace("NEXUS ", "NEXUS\n") for r in rows], fontsize=8)
        ax.set_ylabel("episode return (gameable)", color="#B5762F")
        ax2.set_ylabel(f"{goal_name} (the actual goal)", color="#8A4B04")
        ax.set_title(env_label, fontsize=11, loc="left")
        ax.grid(alpha=0.15, lw=0.6, axis="y")
        # Pin BOTH axes to zero. Seed error bars on a near-zero success rate (Go1 flat is
        # 0.07 +/- 0.11) autoscale the right axis below zero, which offsets its origin from
        # the left axis and makes the success bars look like they float above the baseline.
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
    # Build the legend from explicit patches. Calling twinx() again here stamped a SECOND
    # right-hand axis onto the first panel, which drew a duplicate set of tick labels on top of
    # the real ones and made the goal-metric scale unreadable.
    from matplotlib.patches import Patch  # noqa: PLC0415

    fig.legend(
        handles=[
            Patch(facecolor="#F3C9A0", edgecolor="#B5762F", label="episode return (gameable)"),
            Patch(facecolor="#E58606", edgecolor="#8A4B04", label="primary success (the goal)"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Return disagrees with the actual goal — return is not a behaviour gate  "
        "(paper Fig. 7 analogue)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/v3")
    args = ap.parse_args()
    out = Path(args.out)

    fig3(
        [
            ("HopperHop", {
                "NEXUS (neural)": "runs/verify/hopper_hop_neural_v2_s*.pkl",
                "NEXUS (nesy)": "runs/verify/hopper_hop_nesy_v2_s*.pkl",
            }),
            ("Go1", {
                "NEXUS (neural)": "runs/verify/go1_joystick_neural_v2_s*.pkl",
                "NEXUS (nesy)": "runs/verify/go1_joystick_nesy_v2_s*.pkl",
            }),
        ],
        out / "fig3_skill_returns.png",
    )

    fig7(
        [
            ("Go1JoystickFlatTerrain", {
                "flat (baseline)": "runs/verify/go1_joystick_flat_v2_s*.pkl",
                "NEXUS (neural)": "runs/verify/go1_joystick_neural_v2_s*.pkl",
                "NEXUS (nesy)": "runs/verify/go1_joystick_nesy_v2_s*.pkl",
            }, "joystick tracking success"),
            ("HopperHop", {
                "flat (baseline)": "runs/viper/hopper_hop_flat_s*.pkl",
                "NEXUS (neural)": "runs/verify/hopper_hop_neural_v2_s*.pkl",
                "NEXUS (nesy)": "runs/verify/hopper_hop_nesy_v2_s*.pkl",
            }, "hop success"),
        ],
        out / "fig7_return_vs_goal.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
