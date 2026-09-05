"""Restore trusted legacy actor trees and execute neural forward passes on CPU.

This verifies actual weights, not just file names. It does NOT simulate, train,
measure task competence, or approve a legacy checkpoint for the new cohort.
"""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
from common import add_repo,write_json,digest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True)
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); add_repo(a.repo)
    import jax,jax.numpy as jnp,numpy as np
    import robustness_eval as re
    ck=pickle.loads(Path(a.checkpoint).read_bytes()); cfg=ck.get('config',{})
    if not cfg or not ck.get('normalization_stats'):
        raise ValueError('This structural test requires the legacy NEXUS config and normalization stats')
    policy=re.load_policy_module(cfg['POLICY']); n=policy.NUM_SKILLS
    rs=ck['runner_state']['0']; saved=rs['actor']['params']; norm=ck['normalization_stats']
    obs_dim=int(np.asarray(norm['actor_mean']).shape[-1])
    # The action-output head is the Dense layer with the highest numerical index.
    names=sorted((k for k in saved if k.startswith('Dense_')), key=lambda k:int(k.split('_')[-1]))
    if not names: raise ValueError('Unrecognized actor layout')
    action_dim=int(np.asarray(saved[names[-1]]['bias']).shape[-1])
    actor,meta=re._build_networks(cfg,n,action_dim,jnp.ones(action_dim),jnp.zeros(action_dim))
    dummy=jnp.zeros((2,obs_dim))
    initial=jax.vmap(lambda key:actor.init(key,dummy)['params'])(jax.random.split(jax.random.PRNGKey(0),n))
    params=re._restore_params(initial,saved)
    values=jax.vmap(lambda q:actor.apply({'params':q},dummy))(params)
    report={'checkpoint':str(Path(a.checkpoint).resolve()),'sha256':digest(Path(a.checkpoint)),
      'actor_shape':list(values.shape),'actor_finite':bool(np.isfinite(np.asarray(values)).all()),
      'source_policy':cfg['POLICY'],'env':cfg['ENV_NAME'],'seed':cfg.get('SEED'),
      'actual_steps':int(np.asarray(rs['actor']['timesteps'])),
      'normalizer_finite':all(np.isfinite(np.asarray(v)).all() for v in norm.values()),
      'training_executed':False,'environment_rollout_executed':False,'primary_reuse_approved':False}
    if cfg['META_POLICY_TYPE'] in ('nesy','neural'):
        mp=re._restore_params(meta.init(jax.random.PRNGKey(1),dummy)['params'],rs['meta']['params'])
        q=meta.apply({'params':mp},dummy)
        report.update(meta_shape=list(q.shape),meta_finite=bool(np.isfinite(np.asarray(q)).all()))
    report['pass']=report['actor_finite'] and bool(report['normalizer_finite']) and report.get('meta_finite',True)
    report['normalizer_finite']=bool(report['normalizer_finite'])
    write_json(Path(a.out),report); print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['pass'] else 2)
if __name__=='__main__': main()
