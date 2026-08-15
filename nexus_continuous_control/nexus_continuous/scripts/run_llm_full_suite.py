"""Run the full LLM-extension result matrix: hand-written baseline, an
LLM-generated skillset (per backend, e.g. hf + vertex/gemini), and the
interactive refinement loop (per backend) -- across a set of environments.

For each environment, the hand-written baseline is trained ONCE(shared accross 
backends) so identical compute isn't wasted training the same baseline twice.
The pipeline LLM generation + training is run per backend. 

Results on: `output/<env>/...`.
`output/manifest.json` and `output/<env>_summary.json` are produced for 
collect_llm_results.py and help it turn the results into a report.
"""

from __future__ import annotations 
import argparse 
import json 
import pickle 
import time 
from dataclasses import asdict 
from pathlib import Path 
from typing import Any
import numpy as np 

from nexus_continuous.envs.env_registry import ENV_REGISTRY 
from nexus_continuous.policies.registry import canonicalize_policy_name
from nexus_continuous.llm.pipeline import generate_skillset, save_skillset
from nexus_continuous.utils import load_config
from nexus_continuous.algorithms import hierarchical_ac_pqn_playground as algo
from nexus_continuous.scripts.run_llm_refinement import build_client, run_refinement, _scalar_metrics 

DEFAULT_ENV_CONFIGS = {
    "CartpoleBalance": "configs/cartpole_balance_nesy.yaml",
    "CheetahRun": "configs/cheetah_run_nesy.yaml",
    "WalkerWalk": "configs/walker_walk_nesy.yaml",
    "HopperHop": "configs/hopper_hop_nesy.yaml",
    "PandaPickCube": "configs/panda_pick_cube_nesy.yaml",
    "Go1JoystickFlatTerrain": "configs/go1_joystick_nesy.yaml",
} 

def _run_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Mean/std across seeds for the headline reward + success/goal metrics."""
 
    def _collect(key: str, source: str) -> list[float]:
        vals = []
        for r in runs:
            v = r.get(source, {}).get(key)
            if v is not None and np.isfinite(v):
                vals.append(v)
        return vals
 
    out: dict[str, float] = {}
    er = _collect("returns/env_reward_mean", "metrics")
    if er:
        out["env_reward_mean"] = float(np.mean(er))
        out["env_reward_std"] = float(np.std(er))
    sr = _collect("primary_success_rate", "eval_metrics")
    if sr:
        out["success_rate_mean"] = float(np.mean(sr))
        out["success_rate_std"] = float(np.std(sr))
    gm = _collect("primary_goal_metric", "eval_metrics")
    if gm:
        out["goal_metric_mean"] = float(np.mean(gm))
        out["goal_metric_std"] = float(np.std(gm))
    
    return out



def train_hand_written(env_name: str, config_path: str, seeds: list[int],
                        overrides: list[str] | None = None) -> list[dict[str, Any]]:
    runs = []
    for seed in seeds:
        cfg = load_config(config_path, overrides or [])
        cfg["ENV_NAME"] = env_name
        cfg["SEED"] = seed
        cfg["POLICY"] = canonicalize_policy_name(env_name)
        cfg["TASK_POLICY"] = cfg["POLICY"]
        
        t0 = time.time()
        out = algo.run_training(cfg)
        
        runs.append({
            "seed": seed,
            "metrics": _scalar_metrics(out.metrics),
            "eval_metrics": _scalar_metrics(out.eval_metrics) if out.eval_metrics else {},
            "wall_s": time.time() - t0,
        })
        print(f"  [hand-written seed={seed}] env_reward_mean="
              f"{runs[-1]['metrics'].get('returns/env_reward_mean'):.3f}  "
              f"({runs[-1]['wall_s']:.0f}s)")
    
    return runs



def train_llm(env_name: str, config_path: str, backend: str, seeds: list[int],
              output: str, overrides: list[str] | None = None):
    """Generate ONE skillset for (env, backend), then train it across all seeds."""
    
    meta = ENV_REGISTRY[env_name]
    client = build_client(backend, env_name, seed=seeds[0])
    skillset = generate_skillset(
        env_name=env_name,
        observation_schema="\n".join(meta["fields"]),
        task_description=meta["task"],
        client=client,
        allowed_fields=set(meta["fields"]),
    )
    out_dir = Path(output) / env_name / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    save_skillset(skillset, str(out_dir / "llm_skillset.json"))
    print(f"  [llm/{backend}] skills: {[s.name for s in skillset.skills]}")
 
    obs_fields = tuple(meta["fields"])
    runs = []
    for seed in seeds:
        cfg = load_config(config_path, overrides or [])
        cfg["ENV_NAME"] = env_name
        cfg["SEED"] = seed
        cfg["USE_LLM_SKILLS"] = True
        cfg["LLM_SKILLSET"] = asdict(skillset)
        cfg["OBS_FIELDS"] = obs_fields
        
        t0 = time.time()
        out = algo.run_training(cfg)
        
        runs.append({
            "seed": seed,
            "metrics": _scalar_metrics(out.metrics),
            "eval_metrics": _scalar_metrics(out.eval_metrics) if out.eval_metrics else {},
            "wall_s": time.time() - t0,
        })
        print(f"  [llm/{backend} seed={seed}] env_reward_mean="
              f"{runs[-1]['metrics'].get('returns/env_reward_mean'):.3f}  "
              f"({runs[-1]['wall_s']:.0f}s)")
    
    return runs, skillset, client


def run_suite(
    envs: list[str],
    backends: list[str],
    env_configs: dict[str, str] | None = None,
    seeds: list[int] = (0, 1, 2),
    refine_iterations: int = 4,
    output: str = "results",
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    
    env_configs = env_configs or DEFAULT_ENV_CONFIGS
    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"envs": {}, "seeds": list(seeds), "backends": backends}
 
    for env_name in envs:
        config_path = env_configs[env_name]
        print(f"\n{'=' * 70}\n{env_name}: hand-written baseline ({len(seeds)} seed(s))\n{'=' * 70}")
        hand_runs = train_hand_written(env_name, config_path, list(seeds), overrides)
        hand_summary = _run_summary(hand_runs)
        print("  summary:", hand_summary)
 
        env_entry: dict[str, Any] = {
            "config": config_path,
            "hand_written": {"runs": hand_runs, "summary": hand_summary},
            "backends": {},
        }
 
        for backend in backends:
            print(f"\n{'-' * 70}\n{env_name} / {backend}: LLM skillset ({len(seeds)} seed(s))\n{'-' * 70}")
            llm_runs, skillset, client = train_llm(env_name, config_path, backend,
                                                     list(seeds), output, overrides)
            llm_summary = _run_summary(llm_runs)
            print("  summary:", llm_summary)
 
            print(f"\n{'-' * 70}\n{env_name} / {backend}: refinement loop "
                  f"({refine_iterations} iterations)\n{'-' * 70}")
            refine_result = run_refinement(
                env_name, config_path, backend, seed=seeds[0],
                num_iterations=refine_iterations, output=output,
                overrides=overrides, client=client,
            )
 
            env_entry["backends"][backend] = {
                "llm": {
                    "runs": llm_runs,
                    "summary": llm_summary,
                    "skillset": asdict(skillset),
                },
                "refinement": {
                    "curve": refine_result["history_serialized"],
                    "stopped_early": refine_result["stopped_early"],
                    "plot_path": refine_result["plot_path"],
                    "final_skillset": (
                        asdict(refine_result["final_skillset"])
                        if hasattr(refine_result["final_skillset"], "__dataclass_fields__")
                        else refine_result["final_skillset"]
                    ),
                },
            }
 
        manifest["envs"][env_name] = env_entry
        with open(out_root / f"{env_name}_summary.json", "w") as f:
            json.dump(env_entry, f, indent=2, default=str)
 
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(out_root / "manifest.pkl", "wb") as f:
        pickle.dump(manifest, f)
 
    print(f"\nDone. Manifest: {out_root / 'manifest.json'}")
    print(f"Next: python -m nexus_continuous.scripts.collect_llm_results --results {output}")
    
    return manifest 


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=list(DEFAULT_ENV_CONFIGS),
                     choices=list(DEFAULT_ENV_CONFIGS))
    ap.add_argument("--backends", nargs="+", default=["hf", "vertex"],
                     choices=["mock", "hf", "vertex", "gemini", "openai"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--refine-iterations", type=int, default=4)
    ap.add_argument("--output", default="results")
    ap.add_argument(
        "--override", action="append", default=[],
        help="Training config override applied to every run in the suite, "
             "e.g. --override EVAL_AFTER_TRAIN=True (repeatable).",
    )
    args = ap.parse_args(argv)
    run_suite(
        args.envs, args.backends, seeds=args.seeds,
        refine_iterations=args.refine_iterations, output=args.output,
        overrides=args.override,
    )
 
 
if __name__ == "__main__":
    main()