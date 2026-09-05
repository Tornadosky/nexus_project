"""Diagnostic: unchanged update equations, host-orchestrated JIT updates."""
import os, sys, inspect, time, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'sources/core'))
import yaml, jax, numpy as np
from chunked_env import install
install(32)
from nexus_continuous.algorithms import hierarchical_ac_pqn_playground as trainer
if not os.environ.get('SLURM_JOB_ID'): raise SystemExit('Allocated APU required')
work=Path('/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/capacity/host_update_nesy')
work.mkdir(parents=True,exist_ok=False)
cfg=yaml.safe_load((ROOT/'plan/configs/core__go1__nesy__s0.yaml').read_text())
cfg.update(SEED=7000,TOTAL_TIMESTEPS=1310720,PRINT_EVERY=0,EVAL_AFTER_TRAIN=False)
source=inspect.getsource(trainer.make_train)
anchor='        runner_state, metrics = jax.lax.scan(\n            _update_step, runner_state, None, config["NUM_UPDATES"]\n        )'
assert source.count(anchor)==1
source=source.replace(anchor,'        return runner_state, _update_step, _run_deterministic_evaluation')
ns=dict(trainer.__dict__)
exec(compile(source,'<host-init-and-update>','exec'),ns)
print('MAKE_TRAIN',flush=True)
factory=ns['make_train'](cfg)
print('INITIALIZE',flush=True)
t0=time.monotonic()
runner,update,evaluate=factory(jax.random.split(jax.random.PRNGKey(7000),1)[0])
jax.block_until_ready(runner)
print('INIT_PASS',time.monotonic()-t0,flush=True)
update_jit=jax.jit(lambda state:update(state,None))
records=[]
for i in range(10):
    runner,metrics=update_jit(runner)
    jax.block_until_ready(runner)
    vals=jax.device_get(metrics)
    assert all(np.isfinite(np.asarray(v)).all() for v in vals.values())
    records.append({k:np.asarray(v).tolist() for k,v in vals.items()})
    print('UPDATE_PASS',i+1,time.monotonic()-t0,flush=True)
from flax import serialization
from common import write_pickle
stats=trainer._normalization_stats_from_state(runner[1],runner[2])
write_pickle(work/'final.pkl',dict(config=cfg,actual_steps=1310720,
    runner_state=serialization.to_state_dict(runner),
    normalization_stats=jax.device_get(stats),checkpoint_kind='full_runner'))
report=dict(scientific_sample=False,method='nesy',env_chunk=32,
    execution='host loop, same jitted optimizer update',num_envs=2048,
    steps=1310720,seconds=time.monotonic()-t0,memory_stats=jax.devices()[0].memory_stats())
(work/'PASS.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
