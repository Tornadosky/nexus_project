"""Budget-exact PPO reference, using shipped hyperparameters and explicit overrides."""
from __future__ import annotations
import argparse, functools, inspect, time
from pathlib import Path
import yaml
from common import add_repo,write_json,write_pickle,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--config',required=True)
    p.add_argument('--out',required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    add_repo(a.repo); cfg=yaml.safe_load(Path(a.config).read_text())
    import jax
    from train_ppo_baseline import ppo_config_for,_shim_device_put_replicated
    from brax.training.agents.ppo import train as ppo,networks
    from mujoco_playground import registry,wrapper
    _shim_device_put_replicated()
    if len(jax.local_devices())!=1: raise ValueError('Expose exactly one accelerator to this process')
    kw=dict(ppo_config_for(cfg['ENV_NAME'])); net=dict(kw.pop('network_factory',{}) or {})
    kw.update(num_timesteps=int(cfg['TOTAL_TIMESTEPS']),num_evals=11,num_resets_per_eval=1,
              num_eval_envs=64,clipping_epsilon=.3,gae_lambda=.95,normalize_advantage=True)
    # These values are part of the frozen protocol, not a hyperparameter sweep.
    if cfg['ENV_NAME']=='HopperHop':
        kw.update(num_envs=2048,batch_size=1024,num_minibatches=32,unroll_length=30,
                  num_updates_per_batch=16,learning_rate=.001,discounting=.995,
                  entropy_cost=.01,reward_scaling=10.,action_repeat=1,episode_length=1000,
                  normalize_observations=True)
    elif cfg['ENV_NAME']=='Go1JoystickFlatTerrain':
        kw.update(num_envs=8192,batch_size=256,num_minibatches=32,unroll_length=20,
                  num_updates_per_batch=4,learning_rate=.0003,discounting=.97,
                  entropy_cost=.01,reward_scaling=1.,action_repeat=1,episode_length=1000,
                  max_grad_norm=1.,normalize_observations=True)
        net=dict(policy_hidden_layer_sizes=(512,256,128),value_hidden_layer_sizes=(512,256,128),
                 policy_obs_key='state',value_obs_key='privileged_state')
    else: raise ValueError('Only the two declared core PPO tasks are in scope')
    if a.smoke:
        kw.update(num_envs=32,batch_size=32,num_minibatches=1,unroll_length=5,
                  num_updates_per_batch=1,num_evals=2,num_timesteps=320,num_eval_envs=4)
        cfg['CAMPAIGN_SMOKE']=True; cfg['TOTAL_TIMESTEPS']=320
    unit=kw['batch_size']*kw['num_minibatches']*kw['unroll_length']*kw['action_repeat']
    if kw['num_timesteps']%(unit*(kw['num_evals']-1)):
        raise ValueError('Requested PPO budget would overshoot; do not run')
    sig=inspect.signature(ppo.train)
    if 'policy_params_fn' not in sig.parameters: raise RuntimeError('Pinned Brax lacks policy_params_fn')
    if set(kw)-set(sig.parameters): raise RuntimeError(f'Unsupported pinned Brax kwargs: {set(kw)-set(sig.parameters)}')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg); write_json(out/'effective_ppo_config.json',{**kw,'network_factory':net,
        'pinned_function_defaults':{name:repr(param.default) for name,param in sig.parameters.items()
                                    if param.default is not inspect.Parameter.empty}})
    write_json(out/'source.json',dict(ppo_sha256=digest(Path(ppo.__file__)),jax=jax.__version__,
                                    devices=[str(d) for d in jax.devices()]))
    ec=registry.get_default_config(cfg['ENV_NAME'])
    if 'impl' in ec: ec.impl='jax'
    env=registry.load(cfg['ENV_NAME'],config=ec); ev=registry.load(cfg['ENV_NAME'],config=ec)
    t0=time.monotonic(); progress=[]; saved=set()
    effective={**kw,'network_factory':net}
    def payload(step,params):
        return dict(config=cfg,params=jax.device_get(params),ppo_config=effective,
                    actual_steps=int(step),env_name=cfg['ENV_NAME'],seed=cfg['SEED'],checkpoint_kind='ppo')
    def checkpoint(step,make_policy,params):
        n=int(step)
        if n in saved: return
        write_pickle(out/'snapshots'/f'step_{n:012d}.pkl',payload(n,params)); saved.add(n)
    def report(step,metrics):
        progress.append(dict(step=int(step),wall_seconds=time.monotonic()-t0,
                             **{k:float(v) for k,v in metrics.items() if getattr(v,'ndim',0)==0}))
        write_json(out/'progress.json',progress,replace=True); print(progress[-1],flush=True)
    _,params,_=ppo.train(environment=env,eval_env=ev,wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=functools.partial(networks.make_ppo_networks,**net),
        seed=int(cfg['SEED']),progress_fn=report,policy_params_fn=checkpoint,**kw)
    jax.effects_barrier()
    expected=int(kw['num_timesteps'])
    if max(saved,default=-1)!=expected:
        raise RuntimeError(f'Actual saved maximum {max(saved,default=-1)} != budget {expected}; keep files, do not label matched')
    write_pickle(out/'final.pkl',payload(expected,params))
    write_json(out/'COMPLETE.json',dict(actual_steps=expected,smoke=a.smoke,
        snapshots=len(saved),final_sha256=digest(out/'final.pkl'),wall_seconds=time.monotonic()-t0))
if __name__=='__main__': main()
