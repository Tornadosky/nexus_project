"""Add observational snapshots to the supplied trainer; no learning-equation edits.

The checked-in trainer writes only after training. This wrapper injects host callbacks
into a loaded copy, saving evaluation-only actor/meta/normalizer snapshots at 0..100%.
The final checkpoint includes the original complete runner state. Nothing in the
user's repository is edited. Intermediate snapshots cannot resume training.
"""
from __future__ import annotations
import argparse, inspect, time
from pathlib import Path
import yaml
from common import add_repo, replace_once, write_json, write_pickle, digest

def train(repo: str, config: str, out: str, specs: str|None=None, smoke: bool=False):
    repo=add_repo(repo); cfg=yaml.safe_load(Path(config).read_text())
    if int(cfg.get('NUM_SEEDS',1))!=1: raise ValueError('One seed and one accelerator per process required')
    if cfg.get('USE_RGB'): raise ValueError('Use rgb_run.py for the separate vision snapshot')
    if specs and cfg.get('CAMPAIGN_SPEC'):
        cfg['CAMPAIGN_SPEC_PAYLOAD']=__import__('json').loads((Path(specs)/cfg['CAMPAIGN_SPEC']).read_text())
    if cfg.get('CAMPAIGN_SPEC') and 'CAMPAIGN_SPEC_PAYLOAD' not in cfg:
        raise FileNotFoundError('A generated, validated specification is required; pass --specs')
    import jax, numpy as np
    from flax import serialization
    from nexus_continuous.algorithms import hierarchical_ac_pqn_playground as trainer
    from llm_specs import install_policy
    install_policy(cfg,trainer)
    if smoke:
        cfg.update(NUM_ENVS=32,NUM_STEPS=8,NUM_MINIBATCHES=4,NUM_EPOCHS=1,
                   TOTAL_TIMESTEPS=512,EVAL_AFTER_TRAIN=False,CAMPAIGN_SMOKE=True)
    quantum=cfg['NUM_ENVS']*cfg['NUM_STEPS']; updates=cfg['TOTAL_TIMESTEPS']//quantum
    if cfg['TOTAL_TIMESTEPS']%quantum: raise ValueError('Budget must be an exact update multiple')
    interval=1 if smoke else updates//10
    if not smoke and updates%10: raise ValueError('Need ten equal snapshot intervals')
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg)
    write_json(out/'source.json',{'trainer':str(Path(trainer.__file__).resolve()),
        'sha256':digest(Path(trainer.__file__)),'devices':[str(d) for d in jax.devices()],
        'jax':jax.__version__,'instrumentation':'evaluation-only snapshots; final full runner',
        'smoke':smoke})
    t0=time.monotonic()
    def snapshot(n_updates,actor_params,meta_params,stats):
        n=int(np.asarray(n_updates)); steps=n*quantum
        payload={'config':cfg,'actual_steps':steps,'checkpoint_kind':'evaluation_only',
                 'normalization_stats':jax.device_get(stats),
                 'runner_state':{'0':{'actor':{'params':serialization.to_state_dict(actor_params)},
                                      'meta':None if meta_params is None else {'params':serialization.to_state_dict(meta_params)}}}}
        write_pickle(out/'snapshots'/f'step_{steps:012d}.pkl',payload)
        print(f'SNAPSHOT {steps:,} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    def hook(state,stats):
        import jax.numpy as jnp
        def yes(_):
            jax.debug.callback(snapshot,state.actor.n_updates,state.actor.params,
                               None if state.meta is None else state.meta.params,stats,ordered=True)
            return jnp.int32(0)
        return jax.lax.cond(state.actor.n_updates%interval==0,yes,lambda _:jnp.int32(0),None)
    src=inspect.getsource(trainer.make_train)
    anchor='        runner_state, metrics = jax.lax.scan(\n            _update_step, runner_state, None, config["NUM_UPDATES"]\n        )'
    new='''        _campaign_hook(train_state, _normalization_stats_from_state(env_state, obs))
        def _campaign_update(carry, item):
            new_carry, ms = _update_step(carry, item)
            _campaign_hook(new_carry[0], _normalization_stats_from_state(new_carry[1], new_carry[2]))
            return new_carry, ms
        runner_state, metrics = jax.lax.scan(
            _campaign_update, runner_state, None, config["NUM_UPDATES"]
        )'''
    src=replace_once(src,anchor,new)
    ns=dict(trainer.__dict__); ns['_campaign_hook']=hook
    exec(compile(src,'<campaign-instrumented-make-train>','exec'),ns)
    trainer.make_train=ns['make_train']
    result=trainer.run_training(cfg); jax.effects_barrier()
    write_pickle(out/'final.pkl',dict(config=cfg,actual_steps=cfg['TOTAL_TIMESTEPS'],
        checkpoint_kind='full_runner',runner_state=serialization.to_state_dict(result.runner_state),
        normalization_stats=jax.device_get(result.normalization_stats),
        metrics=jax.device_get(result.metrics),eval_metrics=jax.device_get(result.eval_metrics)))
    write_json(out/'timing.json',dict(wall_seconds=time.monotonic()-t0,steps=cfg['TOTAL_TIMESTEPS']))
    write_json(out/'COMPLETE.json',dict(actual_steps=cfg['TOTAL_TIMESTEPS'],smoke=smoke,
        snapshots=len(list((out/'snapshots').glob('*.pkl'))),final_sha256=digest(out/'final.pkl')))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--config',required=True)
    p.add_argument('--out',required=True); p.add_argument('--specs'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); train(a.repo,a.config,a.out,a.specs,a.smoke)
if __name__=='__main__': main()
