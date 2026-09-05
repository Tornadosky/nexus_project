"""Weights & Biases sweep entry point for continuous-control NEXUS.

This is the hyperparameter-search counterpart to ``train_nexus_playground``. The
W&B agent starts a run and injects the swept hyperparameters into ``wandb.config``;
this script merges them onto a base YAML config, runs training, and replays the
metrics history into the *active* run (it does not start a new one).

Usage::

    wandb sweep configs/sweeps/go1_joystick_nesy.yaml      # prints a sweep id
    wandb agent <entity>/<project>/<sweep_id>

The sweep YAML's ``command`` should call this module, e.g.::

    program: -m
    command:
      - ${env}
      - python
      - -m
      - nexus_continuous.scripts.wandb_sweep_agent
      - --config
      - configs/go1_joystick_nesy_phase2.yaml

Swept keys in ``wandb.config`` are applied as config overrides (top-level keys, or
dotted paths like ``WANDB.tags``). Reserved bookkeeping keys are ignored.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
from nexus_continuous.policies.registry import load_policy_module
from nexus_continuous.tracking import replay_history_to_run
from nexus_continuous.utils import load_config

# wandb bookkeeping keys that are not training hyperparameters.
_RESERVED = {"_wandb", "wandb_version", "commit_hash", "seed"}


def _commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _apply_override(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Base YAML config to sweep over.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Static KEY=VALUE override applied before the swept hyperparameters.",
    )
    args = parser.parse_args(argv)

    import wandb

    # The agent has already provided the swept hyperparameters; pick them up.
    run = wandb.init()

    cfg = load_config(args.config, args.override)
    swept = {k: v for k, v in dict(run.config).items() if k not in _RESERVED}
    for key, value in swept.items():
        _apply_override(cfg, key, value)

    # Record the fully-resolved config back onto the run for reproducibility.
    run.config.update({"resolved_config": json.loads(json.dumps(cfg, default=str))}, allow_val_change=True)

    policy = load_policy_module(cfg.get("POLICY", cfg.get("ENV_NAME", "cartpole_balance")))
    print(f"Sweep run: policy={cfg.get('POLICY')} skills={', '.join(policy.SKILL_NAMES)}")
    print("Resolved config:")
    print(json.dumps(cfg, indent=2, default=str))

    output = run_training(cfg)
    print("Training finished; replaying history into the active sweep run.")

    base_seed = int(cfg.get("SEED", 0))
    replay_history_to_run(run, cfg, output, seed_index=None, seed_value=base_seed)
    run.summary["commit_hash"] = _commit_hash()
    run.finish()


if __name__ == "__main__":
    main()
