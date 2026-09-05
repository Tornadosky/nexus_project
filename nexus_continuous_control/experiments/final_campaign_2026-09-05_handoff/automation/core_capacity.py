"""Production-shape capacity checks. These are not primary experiment runs."""
import json, os, runpy, sys, time
from pathlib import Path
import yaml
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Requires one allocated APU')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
if os.environ.get('CAPACITY_CHUNK_ENV'):
    from chunked_env import install
    install(int(os.environ['CAPACITY_CHUNK_ENV']))
method = sys.argv[1]
assert method in ('nesy','ppo')
work = Path('/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/capacity')/os.environ.get('CAPACITY_LABEL',method).format(method=method)
work.mkdir(parents=True, exist_ok=False)
cfg = yaml.safe_load((ROOT/f'plan/configs/core__go1__{method}__s0.yaml').read_text())
cfg.update(SEED=7000, CAMPAIGN_CAPACITY=True,
    CAMPAIGN_ID=f'capacity_only__go1__{method}',
    TOTAL_TIMESTEPS=1638400 if method=='ppo' else 1310720)
config = work/'capacity_config.yaml'
config.write_text(yaml.safe_dump(cfg))
worker = ROOT/'scripts'/('train_ppo.py' if method=='ppo' else 'train_state.py')
sys.argv = [str(worker),'--repo',str(ROOT/'sources/core'),
            '--config',str(config),'--out',str(work/'run')]
t0 = time.monotonic()
runpy.run_path(str(worker), run_name='__main__')
import jax
record = dict(method=method, env='Go1JoystickFlatTerrain', xla_flags=os.environ.get('XLA_FLAGS'), env_chunk=os.environ.get('CAPACITY_CHUNK_ENV'),
    num_envs=8192 if method=='ppo' else 2048, scientific_sample=False,
    steps=cfg['TOTAL_TIMESTEPS'], seconds=time.monotonic()-t0,
    memory_stats=jax.devices()[0].memory_stats())
(work/'PASS.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2), flush=True)
