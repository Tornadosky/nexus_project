"""Isolate original trainer from snapshot instrumentation at production batch size."""
import json, os, sys, time
from pathlib import Path
import yaml, jax, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources/core'))
from nexus_continuous.algorithms.hierarchical_ac_pqn_playground import run_training
assert os.environ.get('SLURM_JOB_ID') and jax.default_backend()=='gpu'
work=Path('/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/capacity/native_no_callbacks')
work.mkdir(parents=True,exist_ok=False)
cfg=yaml.safe_load((ROOT/'plan/configs/core__go1__nesy__s0.yaml').read_text())
cfg.update(SEED=7000,TOTAL_TIMESTEPS=1310720,EVAL_AFTER_TRAIN=False)
(work/'config.json').write_text(json.dumps(cfg,indent=2))
print('NATIVE_BEGIN',jax.__version__,jax.devices(),flush=True)
t=time.monotonic();result=run_training(cfg);jax.effects_barrier()
leaves=jax.tree_util.tree_leaves(result.runner_state[0].actor.params)
assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
record=dict(seconds=time.monotonic()-t,steps=cfg['TOTAL_TIMESTEPS'],num_envs=2048,snapshots_injected=False,scientific_sample=False,memory_stats=jax.devices()[0].memory_stats())
(work/'PASS.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
