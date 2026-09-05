"""Measure the exact production shapes on NVIDIA; never a paper training run."""
import json, os, runpy, sys, time
from pathlib import Path
import yaml, jax
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
assert jax.default_backend()=='gpu' and len(jax.devices())==1
method=sys.argv[1]; task=sys.argv[2] if len(sys.argv)>2 else 'go1'
assert method in ('nesy','ppo') and task in ('go1','hopper')
work=Path('/home/smirn/nexus_campaign_verified_outputs/nvidia_capacity')/(task+'_'+method)
work.mkdir(parents=True,exist_ok=False)
cfg=yaml.safe_load((ROOT/f'plan/configs/core__{task}__{method}__s0.yaml').read_text())
budget=1310720 if method=='nesy' else (1638400 if task=='go1' else 9830400)
cfg.update(SEED=7000,CAMPAIGN_CAPACITY=True,CAMPAIGN_ID=f'capacity_only__{task}__{method}',TOTAL_TIMESTEPS=budget)
config=work/'capacity_config.yaml';config.write_text(yaml.safe_dump(cfg))
worker=ROOT/'scripts'/('train_ppo.py' if method=='ppo' else 'train_state.py')
sys.argv=[str(worker),'--repo',str(ROOT/'sources/core'),'--config',str(config),'--out',str(work/'run')]
t=time.monotonic();runpy.run_path(str(worker),run_name='__main__')
record=dict(task=task,method=method,seconds=time.monotonic()-t,steps=budget,scientific_sample=False,device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind,memory_stats=jax.devices()[0].memory_stats())
(work/'PASS.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
