"""Quick single-checkpoint training plots (CPU-friendly, no GPU/render needed).

Reads one ``.pkl`` saved by ``train_nexus_playground`` and writes a 2x2 figure:
environment reward, per-skill rewards, skill-usage-over-time (stacked), and
losses; plus a printed deterministic-eval summary (if EVAL_AFTER_TRAIN was set).

Usage:
    python tools/plot_run.py runs/local_smoke.pkl
    python tools/plot_run.py runs/local_smoke.pkl --out my_plot.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a file, no display needed
import matplotlib.pyplot as plt
import numpy as np


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "/"))
        else:
            out[key] = v
    return out


def _curve(v):
    a = np.asarray(v, dtype=float).squeeze()
    if a.ndim == 0:
        return None
    if a.ndim == 2:
        a = a.mean(axis=0)            # average over seeds if present
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1]).mean(0)
    return a


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to a .pkl saved by train_nexus_playground")
    p.add_argument("--out", default=None, help="PNG output path (default next to the .pkl)")
    args = p.parse_args(argv)

    ck = pickle.load(open(args.checkpoint, "rb"))
    cfg = ck.get("config", {})
    title = f"{cfg.get('ENV_NAME', '?')} | {cfg.get('META_POLICY_TYPE', '?')} | CRITIC_AGG={cfg.get('CRITIC_AGG', 'mean')}"
    M = _flat(ck.get("metrics"))
    keys = sorted(M)
    grp = lambda pre: sorted(k for k in keys if k.startswith(pre))

    ev = _flat(ck.get("eval_metrics"))
    if ev:
        print("Deterministic-eval summary:")
        for k in sorted(ev):
            try:
                print(f"  {k}: {float(np.asarray(ev[k]).mean()):.4g}")
            except Exception:
                pass
    else:
        print("(no eval_metrics — train with --override EVAL_AFTER_TRAIN=true to get them)")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for k in [x for x in keys if "env_reward" in x or "original_reward" in x]:
        c = _curve(M[k])
        if c is not None:
            ax[0, 0].plot(c, label=k)
    ax[0, 0].set_title("Environment reward"); ax[0, 0].set_xlabel("update"); ax[0, 0].legend(fontsize=7)

    for k in grp("skill_reward/"):
        c = _curve(M[k])
        if c is not None:
            ax[0, 1].plot(c, label=k.split("/")[-1])
    ax[0, 1].set_title("Per-skill reward"); ax[0, 1].set_xlabel("update"); ax[0, 1].legend(fontsize=8)

    uk = grp("skill_usage/")
    cs = [c for c in (_curve(M[k]) for k in uk) if c is not None]
    if cs:
        L = min(len(c) for c in cs); cs = [c[:L] for c in cs]
        ax[1, 0].stackplot(range(L), *cs, labels=[k.split("/")[-1] for k in uk])
        ax[1, 0].set_title("Skill usage (share of steps)"); ax[1, 0].set_xlabel("update")
        ax[1, 0].set_ylim(0, 1); ax[1, 0].legend(fontsize=8, loc="upper right")

    for k in grp("loss/"):
        c = _curve(M[k])
        if c is not None:
            ax[1, 1].plot(c, label=k.split("/")[-1])
    ax[1, 1].set_title("Losses"); ax[1, 1].set_xlabel("update"); ax[1, 1].legend(fontsize=8)

    plt.suptitle(title, fontsize=13); plt.tight_layout()
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".curves.png")
    plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
