"""Build the fixed 142-run maximum matrix, without starting or changing training."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import yaml
from common import write_json, atomic

TASKS={
 'hopper':('HopperHop','hopper_hop',117964800),
 'go1':('Go1JoystickFlatTerrain','go1_joystick',32768000),
 'cheetah':('CheetahRun','cheetah_run',52428800),
 'walker':('WalkerWalk','walker_walk',52428800),
 'cartpole':('CartpoleBalance','cartpole_balance',2048000),
}
METHODS={'flat':('flat',False),'hpqn':('neural',True),'neural':('neural',False),
         'symbolic':('symbolic',False),'nesy':('nesy',False),'ppo':('ppo',False)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--core-repo',required=True)
    p.add_argument('--rgb-repo',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); rows=[]
    def add(group,task,method,seed,budget=None,family=None,spec=None):
        env,pol,B=TASKS[task]; B=B if budget is None else budget
        ident=f'{group}__{task}__{method}' + (f'__g{family}' if family is not None else '') + f'__s{seed}'
        if group=='rgb':
            cfg=yaml.safe_load((Path(a.rgb_repo)/'configs'/f'{pol}_nesy_state_plus_rgb.yaml').read_text())
            cfg.update(RGB_ACTOR=method!='state',RGB_SHARED_ENCODER=False,RGB_META_SEES_PIXELS=False,
                       CAMPAIGN_CONSTANT_PIXELS=method=='constant',NUM_ENVS=128,TOTAL_TIMESTEPS=2048000)
            engine='rgb'; repo='rgb'
        else:
            cfg=yaml.safe_load((Path(a.core_repo)/'configs'/f'{pol}_nesy.yaml').read_text())
            meta,shared=METHODS.get(method,('nesy',False))
            cfg.update(TOTAL_TIMESTEPS=B,NUM_ENVS=2048,NUM_STEPS=64,NUM_EPOCHS=4,NUM_MINIBATCHES=64,
                       META_POLICY_TYPE=meta,SHARED_SKILL_REWARD=shared,USE_RGB=False,
                       META_DECISION_INTERVAL=1,CRITIC_AGG='mean',ACTOR_UPDATE_MODE='all_states',
                       SCALE_CLIP_BY_SKILLS=False,LINSPACE_NOISE=False,ACTOR_INIT_SCALE=.01,
                       CRITIC_INIT_SCALE=1.,META_INIT_SCALE=1.,PLAYGROUND_IMPL='jax',
                       EVAL_AFTER_TRAIN=False)
            if method=='flat': cfg['POLICY']='flat_baseline'
            if spec: cfg.update(POLICY='llm_generated',CAMPAIGN_SPEC=spec)
            engine='ppo' if method=='ppo' else 'state'; repo='core'
        cfg.update(SEED=seed,NUM_SEEDS=1,TASK_POLICY=pol,PRINT_EVERY=0,SAVE_PATH=None,
                   EVAL_NUM_ENVS=64,EVAL_NUM_EPISODES=256,EVAL_MAX_STEPS=1000,
                   EVAL_SEED=30000,CAMPAIGN_ID=ident)
        if group=='rgb': cfg['EVAL_MAX_STEPS']=250
        cf=out/'configs'/f'{ident}.yaml'; cf.parent.mkdir(exist_ok=True)
        atomic(cf,yaml.safe_dump(cfg,sort_keys=True).encode())
        rows.append(dict(id=ident,group=group,task=task,env=env,method=method,seed=seed,
                         proposal_family=family,budget=int(cfg['TOTAL_TIMESTEPS']),
                         engine=engine,repo=repo,config=str(cf.relative_to(out)),spec=spec))
    for task in ('hopper','go1'):
        for method in METHODS:
            for seed in range(5): add('core',task,method,seed)
    for task in ('cheetah','walker'):
        for seed in range(5): add('llm_reference',task,'nesy',seed)
        for g in range(3):
            add('llm_pilot',task,'initial',900+g,budget=13107200,family=g,
                spec=f'{task}/g{g}/initial.json')
            for condition in ('initial','refined','resample'):
                for seed in (0,1):
                    add('llm_final',task,condition,seed,family=g,
                        spec=f'{task}/g{g}/{condition}.json')
    for task in ('cartpole','walker'):
        for method in ('state','pixels','constant'):
            for seed in range(5): add('rgb',task,method,seed)
    assert len(rows)==142
    write_json(out/'matrix.json',rows)
    with (out/'matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={g:sum(r['group']==g for r in rows) for g in sorted({r['group'] for r in rows})}
    write_json(out/'counts.json',summary)
    print(json.dumps(summary,indent=2)); print('TOTAL',len(rows),sum(r['budget'] for r in rows))
if __name__=='__main__': main()
