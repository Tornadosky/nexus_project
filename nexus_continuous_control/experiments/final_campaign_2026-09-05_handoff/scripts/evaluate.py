"""One evaluator for PPO, flat and hierarchical checkpoints; common hand skill rewards.

Extends the supplied first-episode evaluator with raw episode tables, per-episode
skill counts, and strict actor deletion. No training occurs here. Deletion's empty
mask fallback relaxes rules among REMAINING actors and is separately measured.
"""
from __future__ import annotations
import argparse, inspect, io, json, pickle
from pathlib import Path
from common import add_repo,replace_once,write_json,atomic,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True); p.add_argument('--episodes',type=int,default=256)
    p.add_argument('--num-envs',type=int,default=64); p.add_argument('--seed',type=int,default=30000)
    p.add_argument('--max-steps',type=int); p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--selector',choices=('native','unmasked','symbolic'),default='native')
    p.add_argument('--force',type=int); p.add_argument('--remove',type=int)
    p.add_argument('--env-name'); p.add_argument('--command-range',type=float,nargs=3)
    a=p.parse_args(); add_repo(a.repo)
    if a.episodes%a.num_envs: raise ValueError('Use complete evaluation batches')
    if a.force is not None and a.remove is not None: raise ValueError('Force and remove are separate experiments')
    ck=pickle.loads(Path(a.checkpoint).read_bytes()); cfg=dict(ck.get('config',{}))
    if not cfg or not cfg.get('TASK_POLICY'): raise ValueError('Explicit config and TASK_POLICY required; adapt legacy checkpoint first')
    if cfg.get('USE_RGB'): raise ValueError('Vision checkpoints use rgb_run.py')
    if cfg.get('META_DECISION_INTERVAL',1)!=1: raise ValueError('Campaign interventions are defined for per-step selection only')
    out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    import jax, jax.numpy as jnp, numpy as np
    import robustness_eval as re
    from nexus_continuous.envs.playground_adapter import build_playground_env,get_actor_obs,get_policy_obs
    from llm_specs import install_policy
    install_policy(cfg,re)
    task=re.load_policy_module(cfg['TASK_POLICY'])
    ec=dict(cfg,NORMALIZE_OBS=False,NORMALIZE_REWARD=False)
    if a.env_name: ec['ENV_NAME']=a.env_name
    if a.command_range is not None:
        ec['ENV_CONFIG_OVERRIDES']={**ec.get('ENV_CONFIG_OVERRIDES',{}),'command_config.a':a.command_range}
    bundle=build_playground_env(ec); low=jnp.asarray(bundle.action_low); high=jnp.asarray(bundle.action_high)
    scale=(high-low)/2.; raw,_=bundle.env.reset(jax.random.split(jax.random.PRNGKey(0),a.num_envs),bundle.env_params)
    if ck.get('checkpoint_kind')=='ppo' or 'ppo_config' in ck:
        if a.selector!='native' or a.force is not None or a.remove is not None:
            raise ValueError('PPO has no skill selector to intervene on')
        from brax.training.agents.ppo import networks
        from brax.training.acme import running_statistics
        from train_ppo_baseline import make_ppo_selector
        pc=ck['ppo_config']; params=jax.device_put(ck['params'])
        sizes=jax.tree_util.tree_map(lambda x:x.shape[-1],params[0].mean)
        nets=networks.make_ppo_networks(sizes,bundle.action_dim,
            preprocess_observations_fn=running_statistics.normalize,**dict(pc.get('network_factory',{}) or {}))
        make_policy=networks.make_inference_fn(nets)
        select=make_ppo_selector(make_policy,params,pc,low,high); normalize=lambda x:x
        count=0; names=[]; policy=None
    else:
        policy=re.load_policy_module(cfg['POLICY']); count=int(policy.NUM_SKILLS); names=list(policy.SKILL_NAMES)
        for i in (a.force,a.remove):
            if i is not None and not 0<=i<count: raise ValueError(f'Skill {i} outside 0..{count-1}')
        if a.remove is not None and count<2: raise ValueError('Cannot remove the only actor')
        actor,meta=re._build_networks(cfg,count,bundle.action_dim,scale,(high+low)/2.)
        dummy=get_actor_obs(raw); rs=ck['runner_state']['0']
        initial=jax.vmap(lambda k:actor.init(k,dummy)['params'])(jax.random.split(jax.random.PRNGKey(0),count))
        ap=re._restore_params(initial,rs['actor']['params'])
        typ=cfg['META_POLICY_TYPE']; mp=None
        if typ in ('neural','nesy'):
            mp=re._restore_params(meta.init(jax.random.PRNGKey(1),dummy)['params'],rs['meta']['params'])
        if a.selector=='unmasked' and mp is None: raise ValueError('This checkpoint has no trained meta-Q')
        if a.remove is not None and mp is None: raise ValueError('Deletion experiment requires a trained meta-Q')
        normalize=re._make_normalizer(ck.get('normalization_stats'),cfg.get('NORMALIZE_OBS',True))
        native=re._make_selector(actor,meta,ap,mp,policy,typ,low,high,1)
        def select(obs,hold=None):
            if a.selector=='native' and a.force is None and a.remove is None: return native(obs,hold)
            oa=get_actor_obs(obs); po=get_policy_obs(obs); E=oa.shape[0]
            acts=jnp.swapaxes(jax.vmap(lambda pp:actor.apply({'params':pp},oa))(ap),0,1)
            if a.force is not None: skill=jnp.full((E,),a.force,dtype=jnp.int32)
            elif a.selector=='symbolic': skill=policy.symbolic_meta_policy(po).astype(jnp.int32)
            else:
                q=meta.apply({'params':mp},oa)
                mask=policy.skill_mask(po).astype(bool) if typ=='nesy' and a.selector!='unmasked' else jnp.ones_like(q,dtype=bool)
                if a.remove is not None:
                    remaining=jnp.ones_like(mask).at[:,a.remove].set(False)
                    mask=mask&remaining
                    # Permanently exclude the removed skill, including empty-mask cases.
                    mask=jnp.where(jnp.any(mask,axis=-1,keepdims=True),mask,remaining)
                skill=jnp.argmax(jnp.where(mask,q,-jnp.inf),axis=-1).astype(jnp.int32)
            return jnp.clip(acts[jnp.arange(E),skill],low,high),skill,hold
    def metrics(prev,obs,action,reward,done,info):
        m=dict(task.task_metrics(prev,obs,action,reward,done,info))
        if hasattr(task,'diagnostics'): m.update(task.diagnostics(prev,obs,action,reward,done,info))
        rewards=task.skill_rewards(prev,obs,action,reward,done,info)
        for i,name in enumerate(task.SKILL_NAMES): m[f'common_skill_step/{i}_{name}']=rewards[...,i]
        m['action_square_mean']=jnp.mean(action**2,axis=-1)
        if policy is not None and count>1:
            mask=policy.skill_mask(prev).astype(bool)
            m['rules/eligible_count']=jnp.sum(mask,axis=-1).astype(jnp.float32)
            m['rules/overlap_fraction']=(jnp.sum(mask,axis=-1)>1).astype(jnp.float32)
            if a.remove is not None:
                remaining=mask.at[:,a.remove].set(False)
                m['rules/deletion_relaxation_fraction']=((~jnp.any(remaining,axis=-1)).astype(jnp.float32)
                    if typ=='nesy' and a.selector!='unmasked' else jnp.zeros_like(reward))
        if cfg['ENV_NAME'].startswith('Go1'):
            f=task._features(obs,info); h,r,p,vx,vy,yaw,cx,cy,cz=f
            m.update({'go1/orientation_upright_fraction':((jnp.abs(r)<.6)&(jnp.abs(p)<.6)).astype(jnp.float32),
                      'go1/linear_error_squared':(vx-cx)**2+(vy-cy)**2,
                      'go1/yaw_error_squared':(yaw-cz)**2,
                      'go1/command_xy_raw_norm':jnp.sqrt(cx**2+cy**2),
                      'go1/command_abs_yaw':jnp.abs(cz)})
        return m
    source=inspect.getsource(re.evaluate)
    source=replace_once(source,'    return summary','    return summary, flat')
    source=replace_once(source,'jnp.zeros((max(num_skills, 1),), jnp.float32)',
                        'jnp.zeros((num_envs, max(num_skills, 1)), jnp.float32)')
    source=replace_once(source,'''skill_ct = skill_ct + jnp.sum(
                    jax.nn.one_hot(_skill, num_skills) * active[:, None], axis=0
                )''','''skill_ct = skill_ct + jax.nn.one_hot(_skill, num_skills) * active[:, None]''')
    source=replace_once(source,'share = skill_ct / jnp.maximum(jnp.sum(skill_ct), 1.0)',
                        'share = skill_ct / jnp.maximum(jnp.sum(skill_ct, axis=-1, keepdims=True), 1.0)')
    source=replace_once(source,'jnp.full((num_envs,), share[i])','share[:, i]')
    ns=dict(re.__dict__); exec(compile(source,'<campaign-episode-evaluator>','exec'),ns)
    summary,table=ns['evaluate'](cfg,bundle.env,bundle.env_params,select,normalize,metrics,low,high,scale,
        a.num_envs,a.episodes,a.max_steps or int(cfg.get('EVAL_MAX_STEPS') or bundle.episode_length),
        action_noise=a.noise,seed=a.seed,action_dim=bundle.action_dim,decision_interval=1,num_skills=count,skill_names=names)
    table={k:np.asarray(v) for k,v in table.items()}
    for k,v in list(table.items()):
        if k.startswith('common_skill_step/'):
            total=k.replace('common_skill_step/','common_skill_return/')
            table[total]=v*table['episode_length']; summary[total]=float(table[total].mean())
    if 'go1/linear_error_squared' in table:
        summary['go1/linear_rmse']=float(np.sqrt(table['go1/linear_error_squared'].mean()))
        summary['go1/yaw_rmse']=float(np.sqrt(table['go1/yaw_error_squared'].mean()))
    if a.command_range is not None and all(v==0 for v in a.command_range):
        if abs(summary.get('go1/command_xy_raw_norm',float('inf')))>1e-6 or abs(summary.get('go1/command_abs_yaw',float('inf')))>1e-6:
            raise RuntimeError('The requested stop override did not produce zero commands. Check the pinned vendor sampler; do not report this as stop.')
    meta=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=digest(Path(a.checkpoint)),
              actual_steps=ck.get('actual_steps'),train_config=cfg,evaluation_config=ec,
              episodes=a.episodes,num_envs=a.num_envs,eval_seed=a.seed,noise=a.noise,
              selector=a.selector,force=a.force,remove=a.remove,first_episode_only=True)
    out.mkdir(parents=True); write_json(out/'summary.json',summary); write_json(out/'metadata.json',meta)
    buf=io.BytesIO(); np.savez_compressed(buf,**table); atomic(out/'episodes.npz',buf.getvalue())
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
