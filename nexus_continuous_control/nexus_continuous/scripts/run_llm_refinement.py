""" Run interactive LLM refinement loop. 
Propose -> train -> summarize metrics -> feed back to the LLM -> revise.
Save every iteration's skillset JSON + metrics on `output/<env>/<backend>/refinement."""

from __future__ import annotations 
import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
 
from nexus_continuous.envs.env_registry import ENV_REGISTRY
from nexus_continuous.llm.client import LLMClient, LLMConfig, MockSkillGenerator
from nexus_continuous.llm.pipeline import LLMSkillPipeline
from nexus_continuous.llm.refinement_loop import (LLMRefinementLoop, RefinementConfig, summarize_metrics)
from nexus_continuous.llm.plot import plot_refinement
from nexus_continuous.utils import load_config
from nexus_continuous.algorithms import hierarchical_ac_pqn_playground as algo
 
 
def build_client(backend: str, env_name: str, seed: int = 0) -> LLMClient:
    """Maps a --backend string to an LLMClient, shared with
    run_llm_full_suite.py so both scripts build clients identically."""
    
    backend = backend.lower()
    if backend == "mock":
        fields = ENV_REGISTRY[env_name]["fields"]
        return LLMClient(LLMConfig(backend="mock", seed=seed),
                          mock_generator=MockSkillGenerator(fields, seed=seed))
    if backend == "hf":
        return LLMClient(LLMConfig(backend="hf"))
    if backend in ("vertex", "gemini"):
        return LLMClient(LLMConfig(backend="vertex"))
    if backend == "openai":
        return LLMClient(LLMConfig(backend="openai"))
    
    raise ValueError(f"Unknown backend {backend!r} (expected mock/hf/vertex/openai)")

def _scalar_metrics(metrics_dict, tail: int = 5) -> dict[str, float]:
    """Reduce every [T]-shaped metrics array to a scalar (mean of the last
    `tail` updates), keeping every key (skill_usage, policy_diag, mask_available) 
    so behavioural diagnostics survive."""
    
    import jax
    metrics_dict = jax.device_get(metrics_dict)
    out = {}
    
    for k, v in metrics_dict.items():
        arr = np.asarray(v)
        try:
            out[k] = float(arr) if arr.ndim == 0 else float(np.nanmean(arr[-tail:]))
        except (TypeError, ValueError):
            continue
    
    return out 

def make_train_fn(env_name: str, config_path: str, seed: int, overrides: list[str] | None):
    """Return train_fn(skillset) -> metrics."""
    
    obs_fields = tuple(ENV_REGISTRY[env_name]["fields"])
    overrides = overrides or []
 
    def train_fn(skillset):
        ss_dict = asdict(skillset) if hasattr(skillset, "__dataclass_fields__") else skillset
        
        cfg = load_config(config_path, overrides)
        cfg["ENV_NAME"] = env_name
        cfg["SEED"] = seed
        cfg["USE_LLM_SKILLS"] = True
        cfg["LLM_SKILLSET"] = ss_dict
        cfg["OBS_FIELDS"] = obs_fields
        
        out = algo.run_training(cfg)
        return _scalar_metrics(out.metrics)
 
    return train_fn


def run_refinement(
    env_name: str,
    config: str,
    backend: str,
    seed: int = 0,
    num_iterations: int = 4,
    output: str = "results",
    overrides: list[str] | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    
    meta = ENV_REGISTRY[env_name]
    client = client or build_client(backend, env_name, seed=seed)
    pipeline = LLMSkillPipeline(client)
    loop = LLMRefinementLoop(pipeline, client)
    cfg = RefinementConfig(
        env_name=env_name,
        observation_schema="\n".join(meta["fields"]),
        task_description=meta["task"],
        num_iterations=num_iterations,
        allowed_fields=set(meta["fields"]),
    )
    train_fn = make_train_fn(env_name, config, seed, overrides)
 
    result = loop.run(cfg, train_fn)
 
    out_dir = Path(output) / env_name / backend / "refinement"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    history_serialized = []
    for rec in result.history:
        ss = rec.skillset
        ss_dict = asdict(ss) if hasattr(ss, "__dataclass_fields__") else ss
        
        with open(out_dir / f"iter_{rec.iteration}_skillset.json", "w") as f:
            json.dump(ss_dict, f, indent=2)
        
        history_serialized.append({
            "iteration": rec.iteration,
            "metrics": rec.metrics,
            "skillset": ss_dict,
            "refinement_ok": rec.refinement_ok,
            "refinement_error": rec.refinement_error,
        })
        
        msg = f"[{env_name}/{backend}] iter {rec.iteration}: {summarize_metrics(rec.metrics)}"
        if not rec.refinement_ok:
            msg += f"  [refinement failed: {rec.refinement_error}]"
        print(msg)
 
    with open(out_dir / "history.json", "w") as f:
        json.dump(history_serialized, f, indent=2, default=str)
    with open(out_dir / "history.pkl", "wb") as f:
        pickle.dump(result, f)
 
    plot_path = None
    if result.history:
        plot_path = plot_refinement(
            result.history, str(out_dir / "refinement_curve.png"),
            title=f"{env_name} refinement loop ({backend})",
        )
 
    final_ss = result.final_skillset
    final_ss_dict = asdict(final_ss) if hasattr(final_ss, "__dataclass_fields__") else final_ss
    with open(out_dir / "final_skillset.json", "w") as f:
        json.dump(final_ss_dict, f, indent=2)
 
    return {
        "env": env_name,
        "backend": backend,
        "final_skillset": final_ss,
        "history": result.history,
        "history_serialized": history_serialized,
        "stopped_early": result.stopped_early,
        "plot_path": plot_path,
        "out_dir": str(out_dir),
    }
    


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default="hf", choices=["mock", "hf", "vertex", "gemini", "openai"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--output", default="results")
    ap.add_argument(
        "--override", action="append", default=[],
        help="Training config override applied to every refinement iteration's "
             "training run, e.g. --override TOTAL_TIMESTEPS=2000000 to shorten "
             "each iteration for a faster refinement loop.",
    )
    args = ap.parse_args(argv)
    run_refinement(
        args.env, args.config, args.backend, seed=args.seed,
        num_iterations=args.iterations, output=args.output, overrides=args.override,
    )
 
 
if __name__ == "__main__":
    main()