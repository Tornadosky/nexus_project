"""G8 — train the SAVED LLM-generated skillsets at the real V2 budget.

`nexus_continuous/scripts/run_llm_experiment.py` on `origin/llm-extension` always regenerates a
skillset through the LLM client before training. That is the wrong entry point tonight for two
reasons: it puts a Vertex round-trip (and its failure modes) inside a GPU cell, and it would
produce a *different* skillset from the one the report's LLM section already discusses. This
driver loads the skillset json that was saved with those results and trains it, so the only thing
that changes versus the report is the budget.

Runs out of a worktree of `origin/llm-extension` (main has no USE_LLM_SKILLS) with the main
repo's venv, writing checkpoints back into the main repo's runs/llm_budget/.

    PYTHONPATH=<worktree> python tools/_llm_budget_train.py \
        --env HopperHop --config configs/hopper_hop_nesy.yaml \
        --skillset docs/reports/llm_generated_skills/HopperHop.json \
        --seed 0 --out runs/llm_budget/hopper_hop_llm_s0.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from nexus_continuous.envs.env_registry import ENV_REGISTRY
from nexus_continuous.utils import load_config
from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skillset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", action="append", default=[],
                    help="KEY=VALUE, applied after the yaml (same semantics as the trainer).")
    args = ap.parse_args()

    if args.env not in ENV_REGISTRY:
        raise SystemExit(f"unknown env {args.env}; have {sorted(ENV_REGISTRY)}")

    skillset = json.loads(Path(args.skillset).read_text())

    cfg = load_config(args.config)
    cfg["ENV_NAME"] = args.env
    cfg["SEED"] = args.seed
    for kv in args.override:
        k, _, v = kv.partition("=")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    cfg["USE_LLM_SKILLS"] = True
    cfg["LLM_SKILLSET"] = skillset
    cfg["OBS_FIELDS"] = tuple(ENV_REGISTRY[args.env]["fields"])

    print(f"[g8] env={args.env} seed={args.seed} skills={len(skillset.get('skills', []))} "
          f"TOTAL_TIMESTEPS={cfg.get('TOTAL_TIMESTEPS')}", flush=True)

    result = run_training(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"config": cfg, "skills": skillset,
                     "metrics": result.metrics,
                     "eval_metrics": result.eval_metrics}, f)
    print(f"[g8] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
