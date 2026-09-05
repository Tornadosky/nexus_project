"""Print the exact recipe of an existing checkpoint, and its diff against a yaml config.

Runbook §1 (recipe fidelity): a new seed of an existing cell must be the same experiment, and
the config file is not authoritative — `configs/hopper_hop_neural.yaml` ships
NOISE_START=0.35/NOISE_FINISH=0.03 while the `_v2` cells were actually trained at 1.0/0.15.
Launching from the yaml alone silently changes the exploration schedule.
"""

from __future__ import annotations

import argparse
import pickle

import yaml

KEYS = (
    "ALG_NAME", "ENV_NAME", "POLICY", "TASK_POLICY", "META_POLICY_TYPE", "TOTAL_TIMESTEPS",
    "NUM_ENVS", "NUM_STEPS", "NUM_EPOCHS", "NUM_MINIBATCHES", "NOISE_START", "NOISE_FINISH",
    "NOISE_DECAY", "META_EPS_START", "META_EPS_FINISH", "META_EPS_DECAY", "GAMMA", "LAMBDA",
    "SKILL_LAMBDA", "META_LAMBDA", "LR", "MAX_GRAD_NORM", "META_DECISION_INTERVAL", "SEED",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--config", default=None, help="yaml to diff against")
    args = ap.parse_args()

    ck = pickle.load(open(args.checkpoint, "rb"))["config"]
    print(f"== {args.checkpoint}")
    for k in KEYS:
        if k in ck:
            print(f"   {k} = {ck[k]}")

    if args.config:
        cfg = yaml.safe_load(open(args.config))
        print(f"\n== diff vs {args.config} (yaml -> checkpoint); these become --override flags")
        any_diff = False
        for k in sorted(set(ck) | set(cfg)):
            a, b = cfg.get(k, "<absent>"), ck.get(k, "<absent>")
            if str(a) != str(b):
                any_diff = True
                print(f"   {k}: {a}  ->  {b}")
        if not any_diff:
            print("   (identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
