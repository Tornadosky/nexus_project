"""CLI for continuous-control NEXUS on MuJoCo Playground."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import serialization

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.policies.registry import load_policy_module, list_policies
from nexus_continuous.tracking import log_training_run
from nexus_continuous.utils import load_config, save_pickle_checkpoint


def _summarize_metrics(metrics: Any) -> dict[str, float]:
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        out = {}
        for key, value in metrics.items():
            try:
                arr = np.asarray(jax.device_get(value))
                out[key] = float(np.nanmean(arr))
            except Exception:
                pass
        return out
    # NexusTrainOutput under vmap stores metrics as a pytree in output.metrics.
    return _summarize_metrics(metrics)


def _commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=False, default="configs/cartpole_balance_nesy.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config value, e.g. ENV_NAME=CartpoleBalance")
    parser.add_argument("--save", type=str, default=None, help="Optional pickle checkpoint path.")
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases live tracking (the offline CSV pipeline is unaffected).",
    )
    args = parser.parse_args(argv)

    if args.list_policies:
        print(json.dumps(list_policies(), indent=2))
        return

    cfg = load_config(args.config, args.override)
    policy = load_policy_module(cfg.get("POLICY", cfg.get("ENV_NAME", "cartpole_balance")))
    print("Loaded config:")
    print(json.dumps(cfg, indent=2, default=str))
    print(f"Policy skills: {', '.join(policy.SKILL_NAMES)}")
    if hasattr(policy, "explain_policy"):
        print(policy.explain_policy())

    output = run_training(cfg)
    print("Training finished.")

    # Coexist: live W&B tracking is best-effort and never authoritative. The
    # offline pickle -> collect_nexus_results.py -> validator path below/elsewhere
    # remains the source of truth for research gates.
    commit_hash = _commit_hash()
    try:
        run_ids = log_training_run(cfg, output, commit_hash=commit_hash, cli_disable=args.no_wandb)
        if run_ids:
            print(f"Logged {len(run_ids)} W&B run(s): {', '.join(run_ids)}")
    except Exception as exc:  # never let tracking break a training job
        print(f"[wandb] live logging failed but training is unaffected: {exc!r}")

    metrics = output.metrics if hasattr(output, "metrics") else None
    summary = _summarize_metrics(metrics)
    if summary:
        print("Metric summary:")
        for key in sorted(summary):
            print(f"  {key}: {summary[key]:.6g}")
    eval_summary = _summarize_metrics(getattr(output, "eval_metrics", None))
    if eval_summary:
        print("Deterministic eval summary:")
        for key in sorted(eval_summary):
            print(f"  {key}: {eval_summary[key]:.6g}")

    save_path = args.save or cfg.get("SAVE_PATH")
    if save_path:
        payload = {
            "config": cfg,
            "runner_state": serialization.to_state_dict(output.runner_state),
            "metrics": output.metrics,
            "eval_metrics": output.eval_metrics,
            "eval_episode_table": output.eval_episode_table,
            "normalization_stats": output.normalization_stats,
            "commit_hash": commit_hash,
        }
        save_pickle_checkpoint(Path(save_path), payload)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
