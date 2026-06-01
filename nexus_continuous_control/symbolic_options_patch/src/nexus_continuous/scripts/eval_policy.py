"""Lightweight policy/policy-rule inspection CLI.

This script intentionally does not reconstruct full neural checkpoints. It is a
quick way to inspect the hand-written symbolic and NeSy rules and to sanity-check
skill choices on synthetic feature vectors before launching expensive training.
"""

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

from nexus_continuous.policies.registry import load_policy_module


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="cartpole_balance")
    parser.add_argument("--obs", default=None, help="JSON list for one observation, e.g. '[0,0.2,0,0]'" )
    args = parser.parse_args(argv)

    module = load_policy_module(args.policy)
    if args.obs is None:
        obs = jnp.zeros((1, 16))
    else:
        obs = jnp.asarray([json.loads(args.obs)], dtype=jnp.float32)
    skill = module.symbolic_meta_policy(obs)
    mask = module.skill_mask(obs)
    print(module.explain_policy() if hasattr(module, "explain_policy") else "")
    print("skills:", list(module.SKILL_NAMES))
    print("symbolic_skill_id:", [int(x) for x in skill])
    print("symbolic_skill_name:", [module.SKILL_NAMES[int(x)] for x in skill])
    print("nesy_mask:", mask.astype(int).tolist())


if __name__ == "__main__":
    main()
