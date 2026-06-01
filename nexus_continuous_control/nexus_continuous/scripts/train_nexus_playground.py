"""CLI for continuous-control NEXUS on MuJoCo Playground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.policies.registry import load_policy_module, list_policies
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=False, default="configs/cartpole_balance_nesy.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config value, e.g. ENV_NAME=CartpoleBalance")
    parser.add_argument("--save", type=str, default=None, help="Optional pickle checkpoint path.")
    parser.add_argument("--list-policies", action="store_true")
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
    metrics = output.metrics if hasattr(output, "metrics") else None
    summary = _summarize_metrics(metrics)
    if summary:
        print("Metric summary:")
        for key in sorted(summary):
            print(f"  {key}: {summary[key]:.6g}")

    save_path = args.save or cfg.get("SAVE_PATH")
    if save_path:
        payload = {
            "config": cfg,
            "runner_state": output.runner_state,
            "metrics": output.metrics,
        }
        save_pickle_checkpoint(Path(save_path), payload)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
