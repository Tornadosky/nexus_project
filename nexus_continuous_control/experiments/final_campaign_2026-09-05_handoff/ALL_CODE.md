# Complete campaign code and configurations

This is the portable handoff source. It is not proof of a completed remote deployment. The live audit and install/run sequence are in AUDIT_REPORT.md and docs/RUNBOOK.md. User source/weights are not embedded; freeze_sources.py captures sources from the authorized repository.

## scripts/agent.py

```python
"""Machine-specific, opt-in launcher for the audited final campaign.

Default action is a printed plan. No broad queue can start accidentally. Existing
runs are never deleted/reused silently. Production requires a completed smoke
run for the same task/method family on the same recorded environment.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path
from common import ROOT, digest, write_json


def load_profile(name: str) -> dict:
    p = json.loads((ROOT / f'deploy/{name}.json').read_text())
    for key in ('results','evidence','specs'):
        p[key] = os.path.expanduser(p[key])
    return p

def verify_sources() -> str:
    if (ROOT/'INSTALLING').exists():
        raise RuntimeError('Installation incomplete: run verify_installation.py after freezing sources. Production still requires runtime smokes.')
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    for relative, expected in manifest.items():
        p = ROOT / 'sources' / relative
        if not p.is_file() or digest(p) != expected:
            raise RuntimeError(f'Frozen source changed or missing: {p}; create an audited revision, not a silent edit')
    files = sorted((ROOT/'scripts').glob('*.py')) + sorted((ROOT/'plan/configs').glob('*.yaml'))
    h = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode())
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def smoke_key(row: dict) -> str:
    # LLM families share code, but their individual specifications are validated separately.
    return f"{row['engine']}__{row['task']}__{'llm' if row['spec'] else row['method']}"

def base_env(profile: dict) -> dict:
    e = os.environ.copy()
    for key,value in profile.get('environment',{}).items(): e[key] = str(value)
    e['WANDB_MODE'] = 'disabled'; e['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Only the audited site overlay is inherited. The source path is set by each worker.
    e['PYTHONPATH'] = profile.get('pythonpath_prefix','')
    e['JAX_COMPILATION_CACHE_DIR'] = str(Path(profile['results']).parent / 'jax_cache')
    return e

def runtime_id(profile: dict) -> str:
    program = "import json,sys,importlib.metadata as m; names=['jax','jaxlib','flax','optax','mujoco','mujoco-mjx','brax','playground']; print(json.dumps({'python':sys.version,'packages':{n:m.version(n) for n in names}},sort_keys=True))"
    return subprocess.check_output([profile['python'],'-c',program],env=base_env(profile),text=True).strip()

def checks(profile: dict, row: dict, execute: bool, smoke: bool) -> list[str]:
    issues=[]
    if not Path(profile['python']).is_file(): issues.append('configured Python does not exist')
    if row['group'] not in profile['allowed_groups']: issues.append('task is assigned to another machine profile')
    if row['spec']:
        path = Path(profile['specs'])/row['spec']
        if not path.is_file(): issues.append(f'missing validated specification: {path}')
        else:
            from llm_specs import validate
            try: validate(json.loads(path.read_text()),row['task'])
            except Exception as e: issues.append(f'invalid specification: {e}')
    if execute:
        if not smoke and not profile.get('production_allowed',True):
            issues.append('This profile is smoke-only: do not mix WSL core results into the Viper cohort')
        if profile.get('requires_slurm') and not os.environ.get('SLURM_JOB_ID'):
            issues.append('Viper execution requires an allocated Slurm compute job; not the login node')
        capacity=Path(profile.get('capacity_check_path',profile['results']))
        while not capacity.exists(): capacity=capacity.parent
        free=shutil.disk_usage(capacity).free/2**30
        minimum=2 if smoke else float(profile.get('min_free_gib',50))
        if free<minimum: issues.append(f'{capacity}: {free:.1f} GiB free < {minimum:g} GiB guard')
    return issues

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('status','plan','run'))
    p.add_argument('--profile',choices=('wsl_core','wsl_rgb','viper'),default='viper')
    p.add_argument('--id'); p.add_argument('--group'); p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--results',help='new isolated results root for an explicitly documented retry')
    a=p.parse_args(); profile=load_profile(a.profile)
    if a.results: profile['results']=str(Path(a.results).expanduser().resolve())
    fingerprint=verify_sources()
    rows=json.loads((ROOT/'plan/matrix.json').read_text())
    rows=[r for r in rows if (not a.id or r['id']==a.id) and (not a.group or r['group']==a.group)]
    if not rows: raise ValueError('No matching row')
    if a.action=='run' and (not a.id or len(rows)!=1):
        raise ValueError('Exactly one --id required. Bulk scheduling is explicit in the Slurm wrapper.')
    if a.execute and a.action!='run': raise ValueError('--execute is only valid with run')
    print('PACKAGE',ROOT,'FINGERPRINT',fingerprint,flush=True)
    if a.action=='status':
        summary={'profile':a.profile,'rows':len(rows),'source_integrity':'PASS',
                 'python':profile['python'],'results':profile['results'],
                 'audit_production_execution':'not_performed','issues':{}}
        for r in rows:
            for issue in checks(profile,r,False,a.smoke):
                summary['issues'][issue]=summary['issues'].get(issue,0)+1
        print(json.dumps(summary,indent=2)); return
    for row in rows:
        source=ROOT/'sources'/row['repo']
        issues=checks(profile,row,a.execute,a.smoke)
        command=[profile['python'],str(ROOT/'scripts/campaign.py'),'--matrix',str(ROOT/'plan/matrix.json'),
                 '--core-repo',str(ROOT/'sources/core'),'--rgb-repo',str(ROOT/'sources/rgb'),
                 '--results',profile['results'],'--specs',profile['specs'],
                 '--groups',row['group'],'--id',row['id']]
        if a.smoke: command+=['--smoke']
        print(row['id'], 'BLOCKED: '+ '; '.join(issues) if issues else 'PLANNED',flush=True)
        print(shlex.join(command),flush=True)
        if not a.execute: continue
        if issues: raise RuntimeError('; '.join(issues))
        env=base_env(profile)
        lock=None
        if profile.get('gpu_lock'):
            lock=open(profile['gpu_lock'],'a')
            # Same flock as the existing project queues. Never wait invisibly for hours.
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Local GPU lock is held by another job; nothing launched')
        environment=runtime_id(profile)
        receipt=Path(profile['results'])/'_gates'/f'{smoke_key(row)}.json'
        if not a.smoke:
            if not receipt.is_file(): raise RuntimeError(f'Run the task/method smoke first; receipt absent: {receipt}')
            gate=json.loads(receipt.read_text())
            if gate['source_fingerprint']!=fingerprint or gate['runtime']!=environment:
                raise RuntimeError('Smoke gate was produced by different source or environment')
            ck=Path(gate['checkpoint'])
            if not ck.is_file() or digest(ck)!=gate['checkpoint_sha256']:
                raise RuntimeError('Smoke checkpoint no longer matches its receipt')
        run_dir=Path(profile['results'])/(row['id']+('__smoke' if a.smoke else ''))
        # Identity is checked before campaign.py decides to skip a completed folder.
        identity=Path(profile['results'])/'_identities'/(run_dir.name+'.json')
        spec_hash=digest(Path(profile['specs'])/row['spec']) if row['spec'] else None
        ident={'source_fingerprint':fingerprint,'runtime':environment,'config_sha256':digest(ROOT/'plan'/row['config']),
               'spec_sha256':spec_hash,'smoke':a.smoke}
        if identity.exists() and json.loads(identity.read_text())!=ident:
            raise RuntimeError('Existing run identity differs; preserve it and use a new --results for a documented retry')
        if not identity.exists(): write_json(identity,ident)
        subprocess.run(command,check=True,env=env,cwd=source)
        if a.smoke:
            complete=json.loads((run_dir/'COMPLETE.json').read_text())
            expected=3 if row['engine']=='state' else None
            if not complete.get('smoke') or (expected and complete.get('snapshots')!=expected):
                raise RuntimeError('Smoke completion/snapshot count failed')
            # Receipt is a tested-code gate, NOT evidence of learning or a full-size memory test.
            item={'source_fingerprint':fingerprint,'runtime':environment,'run_id':row['id'],
                  'checkpoint':str(run_dir/'final.pkl'),'checkpoint_sha256':digest(run_dir/'final.pkl'),
                  'full_size_memory_tested':False}
            if not receipt.exists(): write_json(receipt,item)
        if lock is not None: lock.close()

if __name__=='__main__': main()

```

## scripts/campaign.py

```python
"""Non-destructive launcher. Groups and rows are frozen in matrix.json.

Use one process per accelerator. Existing runs are skipped ONLY after a verified
COMPLETE record. A crashed/partial directory is never silently treated as success.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from common import ROOT,digest,write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--matrix',required=True)
    p.add_argument('--core-repo'); p.add_argument('--rgb-repo'); p.add_argument('--results',required=True)
    p.add_argument('--specs'); p.add_argument('--groups',default='core')
    p.add_argument('--row',type=int); p.add_argument('--id'); p.add_argument('--list',action='store_true')
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shards',type=int,default=1)
    a=p.parse_args(); matrix=Path(a.matrix).resolve(); rows=json.loads(matrix.read_text())
    if not 0<=a.shard_index<a.shards: raise ValueError('Invalid shard')
    indexed=[(i,r) for i,r in enumerate(rows) if r['group'] in a.groups.split(',')]
    if a.row is not None: indexed=[(i,r) for i,r in indexed if i==a.row]
    if a.id is not None: indexed=[(i,r) for i,r in indexed if r['id']==a.id]
    indexed=[v for n,v in enumerate(indexed) if n%a.shards==a.shard_index]
    if not indexed: raise ValueError('No matching jobs')
    for i,row in indexed:
        if a.list: print(i,row['id'],row['budget']); continue
        repo=a.rgb_repo if row['repo']=='rgb' else a.core_repo
        if not repo: raise ValueError(f'Missing --{row["repo"]}-repo')
        folder=Path(a.results).resolve()/row['id']
        if a.smoke: folder=folder.with_name(folder.name+'__smoke')
        complete=folder/'COMPLETE.json'
        if folder.exists():
            if complete.exists():
                c=json.loads(complete.read_text())
                expected=c['actual_steps'] if a.smoke else row['budget']
                if c['actual_steps']!=expected or bool(c.get('smoke'))!=a.smoke or digest(folder/'final.pkl')!=c['final_sha256']:
                    raise RuntimeError(f'Completion verification failed: {folder}')
                print('VERIFIED COMPLETE',row['id']); continue
            raise RuntimeError(f'Partial directory {folder}. Keep it; use a separate results directory for a documented retry.')
        script={'state':'train_state.py','ppo':'train_ppo.py','rgb':'rgb_run.py'}[row['engine']]
        cmd=[sys.executable,str(ROOT/'scripts'/script),'--repo',str(Path(repo).resolve()),
             '--config',str(matrix.parent/row['config']),'--out',str(folder)]
        if row['spec']:
            if not a.specs: raise ValueError('This row requires --specs')
            spec=Path(a.specs)/row['spec']
            if not spec.exists(): raise FileNotFoundError(spec)
            cmd+=['--specs',str(Path(a.specs).resolve())]
        if a.smoke: cmd+=['--smoke']
        print('COMMAND',__import__('shlex').join(cmd),flush=True)
        if not a.dry_run:
            subprocess.run(cmd,check=True,cwd=Path(repo).resolve())
if __name__=='__main__': main()

```

## scripts/common.py

```python
"""Non-destructive utilities. Only load pickle checkpoints you trust."""
from __future__ import annotations
import hashlib, json, os, pickle, sys, tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
def atomic(path: Path, data: bytes, replace: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f'Refusing to overwrite {path}')
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        if replace: os.replace(temp, path)
        else: os.link(temp, path); os.unlink(temp)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if hasattr(x, 'tolist'): return x.tolist()
    if isinstance(x, Path): return str(x)
    return x

def write_json(path: Path, x: Any, replace: bool = False) -> None:
    atomic(path, json.dumps(jsonable(x), indent=2, allow_nan=False).encode(), replace)

def write_pickle(path: Path, x: Any) -> None:
    atomic(path, pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))

def add_repo(repo: str) -> Path:
    if (ROOT/'INSTALLING').exists():
        raise RuntimeError('Installation incomplete. Do not run worker scripts until verify_installation.py succeeds.')
    p=Path(repo).resolve()
    if not (p/'nexus_continuous').is_dir():
        raise FileNotFoundError(f'{p} must be the nexus_continuous_control package root')
    sys.path.insert(0,str(p)); sys.path.insert(0,str(p/'tools'))
    return p

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old)!=1:
        raise RuntimeError('Source snapshot differs: instrumentation anchor is not unique. '
                           'Stop before spending GPU time; do not guess a replacement.')
    return source.replace(old,new,1)

```

## scripts/eval_viper.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=nexus_final_eval
#SBATCH --partition=apu
#SBATCH --account=mage_apu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=108000
#SBATCH --time=04:00:00
#SBATCH --output=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/logs/%x_%A_%a.out
#SBATCH --error=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/logs/%x_%A_%a.err
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${EVAL_SUITE:?curves,probes,shifts,pilot,llm}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disable
export PYTHONPATH=/ptmp/akalenik/nexus/site
ROOT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs
srun /ptmp/akalenik/jaxrocm_venv/bin/python "$CAMPAIGN_ROOT/scripts/evaluate_campaign.py" \
 --matrix "$CAMPAIGN_ROOT/plan/matrix.json" --repo "$CAMPAIGN_ROOT/sources/core" \
 --results "$ROOT/results" --out "$ROOT/evidence" --suite "$EVAL_SUITE" \
 --shards "${EVAL_SHARDS:-8}" --shard-index "${SLURM_ARRAY_TASK_ID:?}"

```

## scripts/evaluate.py

```python
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
        if abs(summary.get('go1/command_xy_norm',float('inf')))>1e-6 or abs(summary.get('go1/command_abs_yaw',float('inf')))>1e-6:
            raise RuntimeError('The requested stop override did not produce zero commands. Check the pinned vendor sampler; do not report this as stop.')
    meta=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=digest(Path(a.checkpoint)),
              actual_steps=ck.get('actual_steps'),train_config=cfg,evaluation_config=ec,
              episodes=a.episodes,num_envs=a.num_envs,eval_seed=a.seed,noise=a.noise,
              selector=a.selector,force=a.force,remove=a.remove,first_episode_only=True)
    out.mkdir(parents=True); write_json(out/'summary.json',summary); write_json(out/'metadata.json',meta)
    buf=io.BytesIO(); np.savez_compressed(buf,**table); atomic(out/'episodes.npz',buf.getvalue())
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

```

## scripts/evaluate_campaign.py

```python
"""Execute the predeclared checkpoint-only suites; never trains or selects winners."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from common import ROOT,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--matrix',required=True); p.add_argument('--repo',required=True)
    p.add_argument('--results',required=True); p.add_argument('--out',required=True)
    p.add_argument('--suite',choices=('curves','probes','shifts','pilot','llm'),required=True)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0); a=p.parse_args()
    rows=json.loads(Path(a.matrix).read_text()); jobs=[]; results=Path(a.results).resolve()
    def job(row,label,ck,episodes=128,extra=(),seed=30000):
        jobs.append((row,label,ck,episodes,list(extra),seed))
    for r in rows:
        rd=results/r['id']; ck=rd/'final.pkl'
        if a.suite=='curves' and r['group']=='core':
            snapshots=sorted((rd/'snapshots').glob('step_*.pkl'))
            if len(snapshots)!=11: raise RuntimeError(f'{rd}: expected 11 snapshots, found {len(snapshots)}')
            for cp in snapshots:
                steps=int(cp.stem.split('_')[1]); job(r,cp.stem,cp,256 if steps==r['budget'] else 64)
        elif a.suite=='probes' and r['group']=='core' and r['method'] in ('hpqn','neural','symbolic','nesy'):
            job(r,'native',ck)
            for i in range(4): job(r,f'force_{i}',ck,extra=('--force',str(i)))
            if r['method'] in ('neural','nesy'):
                for i in range(4): job(r,f'remove_{i}',ck,extra=('--remove',str(i)))
            if r['method']=='nesy':
                for sel in ('unmasked','symbolic'): job(r,f'selector_{sel}',ck,extra=('--selector',sel))
        elif a.suite=='shifts' and r['group']=='core':
            for n in (0.,.05,.1,.2): job(r,f'noise_{n:g}',ck,extra=('--noise',str(n)))
            if r['task']=='go1':
                conditions={'stop':(0.,0.,0.),'half':(.75,.4,.6),'high':(2.25,1.2,1.8),
                            'forward':(1.5,0.,0.),'yaw':(0.,0.,1.2)}
                for name,vals in conditions.items(): job(r,name,ck,extra=('--command-range',*map(str,vals)))
                job(r,'rough',ck,extra=('--env-name','Go1JoystickRoughTerrain'))
        elif a.suite=='pilot' and r['group']=='llm_pilot': job(r,'validation',ck,64,seed=20000)
        elif a.suite=='llm' and r['group'] in ('llm_final','llm_reference'): job(r,'test',ck,256)
    if not 0<=a.shard_index<a.shards: raise ValueError('Invalid shard')
    jobs=[j for n,j in enumerate(jobs) if n%a.shards==a.shard_index]
    if not jobs: raise ValueError('No jobs')
    for row,label,ck,n,extras,seed in jobs:
        if not ck.exists(): raise FileNotFoundError(f'Missing required weights: {ck}. Do not replace with training metrics.')
        dest=Path(a.out).resolve()/a.suite/row['id']/label
        if dest.exists():
            m=dest/'metadata.json'; data=dest/'episodes.npz'; s=dest/'summary.json'
            if all(f.exists() for f in (m,data,s)) and json.loads(m.read_text())['checkpoint_sha256']==digest(ck):
                print('HAVE VERIFIED EVALUATION',dest); continue
            raise RuntimeError(f'Partial or stale evaluation at {dest}; preserve it and use a separate --out')
        cmd=[sys.executable,str(ROOT/'scripts/evaluate.py'),'--repo',str(Path(a.repo).resolve()),
             '--checkpoint',str(ck),'--out',str(dest),'--episodes',str(n),
             '--num-envs','64','--seed',str(seed),*extras]
        print(__import__('shlex').join(cmd),flush=True)
        if not a.dry_run: subprocess.run(cmd,check=True,cwd=Path(a.repo).resolve())
if __name__=='__main__': main()

```

## scripts/freeze_sources.py

```python
"""Create new, LF-normalized source snapshots without checking out or editing Git.

The core snapshot is the actual working files, including uncommitted fixes. RGB
is exported from the audited origin/main object. No checkpoints are copied.
"""
from __future__ import annotations
import argparse, hashlib, io, json, subprocess, tarfile
from pathlib import Path
from common import ROOT, write_json, atomic

DIRECTORIES = {'nexus_continuous', 'configs', 'tests', 'tools'}
SUFFIXES = {'.py', '.yaml', '.yml', '.toml'}
RGB_COMMIT = '7557d5d9b9c75fbe93091ead6ae525a1c377cdf6'

def selected(name: str) -> bool:
    p = Path(name)
    return (p.name == 'pyproject.toml' or
            (p.parts[0] in DIRECTORIES and p.suffix in SUFFIXES)) and '__pycache__' not in p.parts

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--package', default=str(ROOT))
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); package = Path(a.package).resolve()
    source = repo / 'nexus_continuous_control'; target = package / 'sources'
    if target.exists(): raise FileExistsError(f'Already frozen: {target}')
    if git(repo, 'rev-parse', 'origin/main') != RGB_COMMIT:
        raise RuntimeError('origin/main changed; inspect changes before updating the frozen RGB source')
    contract=json.loads((package/'audit/source_contract.json').read_text())
    if git(repo,'rev-parse','HEAD')!=contract['core_base_head']:
        raise RuntimeError('Working branch HEAD changed since the live audit. Review before freezing a revised campaign.')
    for rel,expected in contract['sha256_lf'].items():
        q=source/rel
        if not q.is_file() or hashlib.sha256(q.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise RuntimeError(f'Critical working source differs from the audit: {rel}. Preserve it; review the diff, not a blind pull.')
    manifest = {}; target.mkdir(parents=True)
    def put(kind: str, name: str, data: bytes) -> None:
        data = data.replace(b'\r\n', b'\n')
        rel = f'{kind}/{name}'
        atomic(target / rel, data)
        manifest[rel] = hashlib.sha256(data).hexdigest()
    for folder in sorted(DIRECTORIES):
        for p in sorted((source / folder).rglob('*')):
            name = p.relative_to(source).as_posix()
            if p.is_file() and selected(name): put('core', name, p.read_bytes())
    put('core', 'pyproject.toml', (source / 'pyproject.toml').read_bytes())
    data = subprocess.check_output(['git', 'archive', f'{RGB_COMMIT}:nexus_continuous_control'], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for member in tf:
            if member.isfile() and selected(member.name):
                if Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError('Unsafe archive path')
                if member.size > 5_000_000: raise ValueError('Unexpectedly large source file')
                put('rgb', member.name, tf.extractfile(member).read())
    write_json(package / 'source_manifest.json', manifest)
    write_json(package / 'audit/source_origins.json', {
        'core': {'kind': 'working-tree snapshot, NOT a clean commit', 'repo': str(repo),
                 'base_head': git(repo, 'rev-parse', 'HEAD'),
                 'semantic_diff': git(repo, 'diff', '--ignore-space-at-eol', '--stat')},
        'rgb': {'kind': 'Git object export', 'commit': RGB_COMMIT},
        'normalization': 'CRLF -> LF only; original files unchanged',
        'files': len(manifest), 'checkpoints_copied': 0})
    print(json.dumps({'files': len(manifest), 'path': str(target)}, indent=2))

if __name__ == '__main__': main()

```

## scripts/inventory.py

```python
"""Inventory source/dependencies and user-confirmed trusted checkpoints. No writes to inputs."""
from __future__ import annotations
import argparse,json,subprocess,sys,pickle
from pathlib import Path
from common import digest,write_json

def run(cmd,cwd=None):
    r=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return {'returncode':r.returncode,'output':r.stdout}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--out',required=True)
    p.add_argument('--checkpoint-root'); p.add_argument('--trusted-checkpoints',action='store_true')
    a=p.parse_args(); root=Path(a.repo).resolve()
    paths=sorted(set(root.glob('nexus_continuous/**/*.py'))|set(root.glob('configs/*.yaml'))|set(root.glob('tools/*.py')))
    out={'repo':str(root),'source_sha256':{str(f.relative_to(root)):digest(f) for f in paths},
         'git_head':run(['git','rev-parse','HEAD'],root),'git_status':run(['git','status','--porcelain'],root),
         'submodules':run(['git','submodule','status','--recursive'],root),
         'pip_freeze':run([sys.executable,'-m','pip','freeze']),'checkpoints':[]}
    if a.checkpoint_root:
        if not a.trusted_checkpoints: raise ValueError('Pickle can execute code; explicitly confirm these are your own trusted files')
        for path in sorted(Path(a.checkpoint_root).rglob('*.pkl')):
            row={'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest(path)}
            try:
                ck=pickle.loads(path.read_bytes()); cfg=ck.get('config',{})
                row.update(keys=list(ck),has_weights=any(k in ck for k in ('runner_state','params','actor_params')),
                           has_normalizer=('normalization_stats' in ck or 'params' in ck),config=cfg,
                           actual_steps=ck.get('actual_steps'),train_budget=cfg.get('TOTAL_TIMESTEPS'),
                           seed=cfg.get('SEED',ck.get('seed')))
                del ck
            except Exception as e: row['error']=f'{type(e).__name__}: {e}'
            out['checkpoints'].append(row)
    write_json(Path(a.out),out)
    print('Sources',len(paths),'checkpoints',len(out['checkpoints']))
if __name__=='__main__': main()

```

## scripts/llm_phases.sh

```bash
#!/bin/bash
# Generates specifications only. Never submits RL training; no implicit environment installation.
set -euo pipefail
: "${KIT:?Installed package}" "${LLM_PY:?Separate verified Torch/Transformers interpreter}"
: "${SPECS:?New specification directory}" "${EVIDENCE:?Directory containing pilot validation evidence}"
phase="${1:?Choose lock, initial, resample, or refine}"
# Same GPU mutex as the existing local project queues.
exec 9>"${TMPDIR:-/tmp}/nexus_local_gpu.lock"
flock -n 9 || { echo "GPU lock held; no generation started"; exit 2; }
mkdir -p "$SPECS"
lock="$SPECS/model_lock.json"
if [[ "$phase" == lock ]]; then
  "$LLM_PY" "$KIT/scripts/llm_specs.py" lock-model --out "$lock"
  exit
fi
[[ "$phase" == initial || "$phase" == resample || "$phase" == refine ]] || exit 2
condition="$phase"
[[ "$condition" != refine ]] || condition=refined
for task in cheetah walker; do
  for family in 0 1 2; do
    out="$SPECS/${task}/g${family}/${condition}.json"
    args=("$KIT/scripts/llm_specs.py" generate --model-lock "$lock" --task "$task" --family "$family" --condition "$condition" --out "$out")
    if [[ "$condition" == refined ]]; then
      # Look up the exact pilot ID from the locked matrix, not a guessed naming convention.
      pilot="$("$LLM_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(next(x["id"] for x in r if x["group"]=="llm_pilot" and x["task"]==sys.argv[2] and x["proposal_family"]==int(sys.argv[3])))' "$KIT/plan/matrix.json" "$task" "$family")"
      args+=(--initial "$SPECS/${task}/g${family}/initial.json" --feedback "$EVIDENCE/pilot/$pilot/validation/summary.json")
    fi
    "$LLM_PY" "${args[@]}"
  done
done

```

## scripts/llm_specs.py

```python
"""Strict bounded specification generation and a validated adapter to the supplied DSL.

No generated Python is executed. Invalid samples are retained as failures. The
small existing model is retained; this is not a comparison of model families.
"""
from __future__ import annotations
import argparse, ast, json, math
from pathlib import Path
from common import write_json, atomic

FIELDS={'cheetah':('forward_velocity','torso_pitch','joint_speed'),
        'walker':('height','torso_pitch','forward_velocity','joint_speed')}
COUNTS={'cheetah':3,'walker':4}
TYPES={'negative_distance','positive_velocity','target_height','binary_bonus','action_penalty','posture_penalty'}
ALLOWED=(ast.Expression,ast.BoolOp,ast.And,ast.Or,ast.UnaryOp,ast.Not,ast.USub,ast.UAdd,
         ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Compare,ast.Gt,ast.GtE,ast.Lt,ast.LtE,
         ast.Eq,ast.NotEq,ast.Call,ast.Name,ast.Load,ast.Constant)

def validate(spec:dict,task:str)->None:
    if task not in FIELDS: raise ValueError('Only cheetah and walker are in the frozen LLM study')
    skills=spec.get('skills')
    if not isinstance(skills,list) or len(skills)!=COUNTS[task]: raise ValueError(f'Exactly {COUNTS[task]} skills required')
    if len({s.get('name') for s in skills})!=len(skills): raise ValueError('Unique names required')
    always=False
    for s in skills:
        if not isinstance(s.get('name'),str) or not s['name'].strip(): raise ValueError('Missing skill name')
        rule=s.get('activation_rule')
        if not isinstance(rule,str) or not rule: raise ValueError('Missing activation_rule')
        tree=ast.parse(rule,mode='eval'); always|=isinstance(tree.body,ast.Constant) and tree.body.value is True
        for n in ast.walk(tree):
            if not isinstance(n,ALLOWED): raise ValueError(f'Unsupported rule syntax: {type(n).__name__}')
            if isinstance(n,ast.Name) and n.id not in (*FIELDS[task],'abs','min','max'):
                raise ValueError(f'Unknown rule field {n.id}')
            if isinstance(n,ast.Constant) and (not isinstance(n.value,(int,float,bool)) or not math.isfinite(float(n.value))):
                raise ValueError('Only finite numeric/bool rule constants')
            if isinstance(n,ast.Call):
                if not isinstance(n.func,ast.Name) or n.func.id not in ('abs','min','max') or n.keywords:
                    raise ValueError('Only abs/min/max calls')
                if len(n.args)!=(1 if n.func.id=='abs' else 2): raise ValueError('Wrong call arity')
        terms=s.get('reward_terms')
        if not isinstance(terms,list) or not 1<=len(terms)<=6: raise ValueError('One to six reward terms per skill')
        for t in terms:
            if set(t)-{'type','weight','lhs','rhs','threshold'}: raise ValueError('Unknown reward-term keys')
            if t.get('type') not in TYPES: raise ValueError(f'Unsupported reward type {t.get("type")}')
            w=t.get('weight',1.)
            if not isinstance(w,(int,float)) or isinstance(w,bool) or not 0<float(w)<=10:
                raise ValueError('Weights must be positive and <=10; zero is rejected, never changed into one')
            if t.get('type')!='action_penalty' and t.get('lhs') not in FIELDS[task]: raise ValueError('Known lhs required')
            if 'rhs' in t and t['rhs'] not in FIELDS[task]: raise ValueError('rhs must be a known field; constants use threshold')
            if 'threshold' in t and (not isinstance(t['threshold'],(float,int)) or not math.isfinite(t['threshold'])):
                raise ValueError('Finite numeric threshold required')
            if 'rhs' in t and 'threshold' in t: raise ValueError('threshold would be ignored when rhs is present')
            typ=t['type']
            if 'rhs' in t and typ!='negative_distance': raise ValueError('rhs is consumed only by negative_distance')
            if 'threshold' in t and typ not in ('negative_distance','target_height','binary_bonus'):
                raise ValueError('threshold would be ignored for this type')
            if typ in ('target_height','binary_bonus') and 'threshold' not in t: raise ValueError('Explicit threshold required')
            if typ=='target_height' and t['threshold']<=0: raise ValueError('target_height threshold must be positive')
            if typ=='action_penalty' and set(t)-{'type','weight'}: raise ValueError('action_penalty has no field arguments')
    if not always: raise ValueError('One skill must explicitly use activation_rule="True" to cover all states')

def install_policy(cfg,consumer):
    if 'CAMPAIGN_SPEC_PAYLOAD' not in cfg: return
    from nexus_continuous.policies.registry import load_policy_module
    from nexus_continuous.llm.interpreter import make_policy_module
    env=cfg['ENV_NAME']; task={'CheetahRun':'cheetah','WalkerWalk':'walker'}[env]
    spec=cfg['CAMPAIGN_SPEC_PAYLOAD']; validate(spec,task)
    hand=load_policy_module(cfg['TASK_POLICY'])
    def fields(obs,info=None):
        f=hand._features(obs,info)
        if task=='cheetah':
            vx,pitch,speed=f
            return dict(forward_velocity=vx,torso_pitch=pitch,joint_speed=speed)
        height,pitch,vx,speed=f
        return dict(height=height,torso_pitch=pitch,forward_velocity=vx,joint_speed=speed)
    module=make_policy_module(spec,FIELDS[task],task_metrics_fn=hand.task_metrics,
                              field_fn=fields,mask_mode='strict')
    original=consumer.load_policy_module
    consumer.load_policy_module=lambda name:module if name=='llm_generated' else original(name)

PROMPT='''Design a continuous-control NEXUS skillset. Return JSON only: {"skills": [
 {"name":"...", "activation_rule":"...", "reward_terms":[{"type":"...","weight":1.0,"lhs":"..."}]}]}.
Every scalar field is the same semantic feature supplied to the hand-designed controller.
Fields and units: forward_velocity in m/s, torso_pitch in radians, joint_speed mean absolute joint
angular speed in rad/s, and height in metres only when listed. No hidden fields or environmental
reward are available. Use EXACTLY the requested skill count and no extra term keys.
One skill must have activation_rule "True". Others may overlap. NeSy chooses max learned meta-Q
among allowed skills every step. Define simple complementary goals, not named copies.
Rules: numeric comparisons, and/or/not, + - *, abs(x), min(x,y), max(x,y). No division.
Rewards sum terms. All weights >0 and <=10; penalty types negate their weight internally.
Allowed vocabulary and EXACT executable semantics:
negative_distance: -weight*abs(lhs-rhs), or -weight*abs(lhs-threshold), or -weight*abs(lhs).
positive_velocity: weight*lhs.
target_height: weight*clip(lhs/threshold,0,1), requires positive threshold (not rhs).
binary_bonus: weight*(lhs>threshold), requires lhs and threshold.
action_penalty: -weight*sum(action**2), has neither lhs nor rhs nor threshold.
posture_penalty: -weight*abs(lhs).
rhs must name another available field, never a number. Constants use threshold when supported.
A terminal penalty of 1 is subtracted from every skill. Produce between one and six terms per skill.
'''

def main():
    from common import ROOT
    if (ROOT/'INSTALLING').exists():
        raise RuntimeError('Installation incomplete; no model download or generation started')
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    lock=sub.add_parser('lock-model'); lock.add_argument('--out',required=True)
    gen=sub.add_parser('generate'); gen.add_argument('--model-lock',required=True)
    gen.add_argument('--task',choices=FIELDS,required=True); gen.add_argument('--family',type=int,choices=range(3),required=True)
    gen.add_argument('--condition',choices=('initial','refined','resample'),required=True)
    gen.add_argument('--initial'); gen.add_argument('--feedback'); gen.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='lock-model':
        from huggingface_hub import HfApi,snapshot_download
        model='Qwen/Qwen2.5-1.5B-Instruct'; rev=HfApi().model_info(model).sha
        local=snapshot_download(model,revision=rev)
        write_json(Path(a.out),dict(model=model,revision=rev,local_path=local,
            temperature=.7,top_p=.9,max_new_tokens=4096,max_syntax_repairs=2)); return
    lock=json.loads(Path(a.model_lock).read_text()); out=Path(a.out)
    if out.exists() or out.with_suffix('.generation.json').exists(): raise FileExistsError(out)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    tok=AutoTokenizer.from_pretrained(lock['local_path'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(lock['local_path'],local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto').eval()
    seed={'initial':2000,'refined':4000,'resample':3000}[a.condition]+a.family
    task_text='Run forward quickly while maintaining posture.' if a.task=='cheetah' else 'Walk forward around 1 m/s while remaining upright near 1.2 m torso height.'
    messages=[dict(role='system',content=PROMPT),dict(role='user',content=
        f'Task: {a.task}. {task_text} Available fields: {FIELDS[a.task]}. Skill count: {COUNTS[a.task]}.')]
    if a.condition=='refined':
        if not a.initial or not a.feedback: raise ValueError('Refinement needs initial JSON and held-out PILOT validation summary')
        messages += [dict(role='assistant',content=Path(a.initial).read_text()),
            dict(role='user',content='Revise this SAME proposal once using these pilot validation metrics. '
                 'Do not add skills. All fields and reward constraints stay the same.\n'+Path(a.feedback).read_text())]
    attempts=[]
    for attempt in range(3):
        set_seed(seed+10000*attempt)
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tok(text,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            result=model.generate(**inputs,do_sample=True,temperature=.7,top_p=.9,
                                  max_new_tokens=4096,pad_token_id=tok.eos_token_id)
        raw=tok.decode(result[0,inputs.input_ids.shape[-1]:],skip_special_tokens=True)
        try:
            cleaned=raw.strip()
            if cleaned.startswith('```'):
                cleaned='\n'.join(cleaned.splitlines()[1:-1])
            spec=json.loads(cleaned); validate(spec,a.task); error=None
        except Exception as e: error=f'{type(e).__name__}: {e}'
        attempts.append(dict(seed=seed+10000*attempt,prompt=messages.copy(),raw=raw,error=error))
        if error is None:
            write_json(out,spec); break
        messages += [dict(role='assistant',content=raw),dict(role='user',content=
            'Syntax/type validation failed: '+error+'. Repair only schema/type errors; keep your proposed goals.')]
    write_json(out.with_suffix('.generation.json'),dict(model=lock,task=a.task,family=a.family,
        condition=a.condition,attempts=attempts,valid=error is None,
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__))
    if error is not None: raise SystemExit('Invalid proposal after two bounded repairs: recorded failure; no replacement sample')
if __name__=='__main__': main()

```

## scripts/make_matrix.py

```python
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

```

## scripts/probe_runtime.py

```python
"""Import/API probe. Does not train, reset a simulator, or allocate a Slurm job.

--device is an optional device-initialization test; it is never implied on a
Viper login node. RGB still requires an actual rendering smoke test afterward.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib, importlib.metadata as md
import inspect, json, os, sys
from pathlib import Path
from common import add_repo, write_json

PACKAGES = ('jax','jaxlib','flax','optax','brax','mujoco','mujoco-mjx','playground',
            'numpy','PyYAML','warp-lang','torch','transformers','huggingface-hub','accelerate')

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--repo', required=True)
    p.add_argument('--kind', choices=('state','ppo','rgb'), required=True)
    p.add_argument('--out', required=True); p.add_argument('--device', action='store_true')
    a = p.parse_args(); source = add_repo(a.repo)
    report = {'python': sys.version, 'executable': sys.executable, 'kind': a.kind,
              'source': str(source), 'packages': {}, 'errors': [],
              'training_executed': False, 'rendering_executed': False}
    for name in PACKAGES:
        try: report['packages'][name] = md.version(name)
        except md.PackageNotFoundError: report['packages'][name] = None
    try:
        import jax
        from nexus_continuous.algorithms import hierarchical_ac_pqn_playground as trainer
        report['trainer'] = str(Path(trainer.__file__).resolve())
        if source not in Path(trainer.__file__).resolve().parents:
            raise RuntimeError('Wrong trainer imported: editable install shadowed frozen source')
        if a.kind == 'state':
            text = inspect.getsource(trainer.make_train)
            anchor = '        runner_state, metrics = jax.lax.scan(\n            _update_step, runner_state, None, config["NUM_UPDATES"]\n        )'
            if text.count(anchor) != 1: raise RuntimeError('Snapshot injection anchor changed')
            if 'SHARED_SKILL_REWARD' not in text: raise RuntimeError('Core source lost the HPQN control')
            import robustness_eval as re
            ev = inspect.getsource(re.evaluate)
            for anchor in ('    return summary', 'jnp.zeros((max(num_skills, 1),), jnp.float32)'):
                if ev.count(anchor) != 1: raise RuntimeError('Common evaluator source anchor changed')
        if a.kind == 'ppo':
            from train_ppo_baseline import ppo_config_for, _shim_device_put_replicated
            from brax.training.agents.ppo import train as ppo
            sig = inspect.signature(ppo.train)
            report['ppo_signature'] = str(sig)
            for key in ('policy_params_fn','num_evals','num_resets_per_eval','wrap_env_fn',
                        'num_eval_envs','clipping_epsilon','gae_lambda','normalize_advantage'):
                if key not in sig.parameters: raise RuntimeError(f'PPO API lacks {key}')
            report['ppo_supported_tasks'] = {n: bool(ppo_config_for(n)) for n in ('HopperHop','Go1JoystickFlatTerrain')}
        if a.kind == 'rgb':
            from nexus_continuous.scripts import rgb_pixel_ablation as rgb
            import mujoco.mjx as mjx
            import warp
            report['rgb_harness'] = str(Path(rgb.__file__).resolve())
            report['mjx_render_api'] = {n: hasattr(mjx,n) for n in ('render','create_render_context','refit_bvh')}
            if not all(report['mjx_render_api'].values()):
                raise RuntimeError('Installed MuJoCo-MJX lacks the frozen RGB renderer API; use a separate verified RGB environment')
            report['requires_render_smoke'] = True
        from mujoco_playground import registry
        import mujoco_playground
        report['playground_module'] = str(Path(mujoco_playground.__file__).resolve())
        try: report['playground_direct_url'] = json.loads(md.distribution('playground').read_text('direct_url.json') or 'null')
        except md.PackageNotFoundError: pass
        if a.device:
            report['devices'] = [str(d) for d in jax.devices()]
            report['backend'] = jax.default_backend()
            if report['backend'] != 'gpu' or len(jax.local_devices()) != 1:
                raise RuntimeError('One exposed accelerator required')
    except Exception as e: report['errors'].append(f'{type(e).__name__}: {e}')
    report['import_api_pass'] = not report['errors']
    # No permissive NaN JSON; every probe has a unique destination.
    write_json(Path(a.out), report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['import_api_pass'] else 2)

if __name__ == '__main__': main()

```

## scripts/rgb_run.py

```python
"""Run the supplied controlled RGB campaign, plus a constant-input CNN control.

All three arms KEEP the vision environment. The constant arm zeroes the pixel
accessor before either trainer or evaluator is imported, so the same CNN receives
constant input during training AND testing. Report 250-step windows, not complete
episode returns: the supplied RGB harness does not stop at the first done.
"""
from __future__ import annotations
import argparse,json,pickle,time
from pathlib import Path
import yaml
from common import add_repo,write_json,write_pickle,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--config',required=True)
    p.add_argument('--out',required=True); p.add_argument('--load-policy'); p.add_argument('--reuse-proof'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--episodes',type=int,default=64); a=p.parse_args()
    add_repo(a.repo); cfg=yaml.safe_load(Path(a.config).read_text())
    constant=bool(cfg.get('CAMPAIGN_CONSTANT_PIXELS',False)); state=not cfg.get('RGB_ACTOR',True)
    if constant and state: raise ValueError('Constant-pixel control requires the same CNN as the informative-image arm')
    if not cfg.get('USE_RGB'): raise ValueError('All arms must keep the vision environment')
    if cfg.get('RGB_SHARED_ENCODER') or cfg.get('RGB_META_SEES_PIXELS'): raise ValueError('Outside frozen experiment scope')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    import jax,jax.numpy as jnp
    from nexus_continuous.envs import playground_adapter as adapter
    if constant:
        original=adapter.get_actor_pixels
        def zero_pixels(obs):
            value=original(obs)
            return None if value is None else jnp.zeros_like(value)
        adapter.get_actor_pixels=zero_pixels
    from nexus_continuous.scripts import rgb_pixel_ablation as rgb
    write_json(out/'config.json',cfg)
    write_json(out/'source.json',dict(rgb_harness=str(Path(rgb.__file__).resolve()),
        sha256=digest(Path(rgb.__file__)),constant_pixels=constant,
        devices=[str(x) for x in jax.devices()],scope='250-step windows, may include resets'))
    updates=2 if a.smoke else 250; envs=8 if a.smoke else 128
    # Keep the original training minibatch divisibility in a smoke run (8*64 /64=8).
    args=['--config',a.config,'--meta','nesy','--seed',str(cfg['SEED']),
          '--updates',str(updates),'--num-envs',str(envs),
          '--episodes',str(2 if a.smoke else a.episodes),'--eval-steps',str(20 if a.smoke else 250),
          '--out',str(out/'evaluation')]
    if state: args+=['--no-rgb']
    if a.load_policy:
        old=pickle.loads(Path(a.load_policy).read_bytes())
        if bool(old.get('constant_pixels',False))!=constant: raise ValueError('Constant-pixel metadata mismatch')
        proof=old
        if 'actual_steps' not in old or 'config' not in old:
            if not a.reuse_proof:
                raise ValueError('Legacy checkpoint lacks budget/config: pass a reuse proof built from its preserved run records')
            proof=json.loads(Path(a.reuse_proof).read_text())
            if proof.get('checkpoint_sha256')!=digest(Path(a.load_policy)):
                raise ValueError('Reuse proof does not match checkpoint bytes')
            if not proof.get('provenance_description'): raise ValueError('Reuse proof needs provenance_description')
        if int(proof['actual_steps'])!=updates*envs*64:
            raise ValueError('Old checkpoint budget does not match the declared experiment')
        pc=proof['config']
        ignored={'CAMPAIGN_ID','CAMPAIGN_CONSTANT_PIXELS','TASK_POLICY','SAVE_PATH','PRINT_EVERY',
                 'EVAL_NUM_ENVS','EVAL_NUM_EPISODES','EVAL_MAX_STEPS','EVAL_SEED'}
        mismatches={k:(pc.get(k),v) for k,v in cfg.items() if k not in ignored and pc.get(k)!=v}
        if mismatches: raise ValueError(f'Reuse configuration mismatch: {mismatches}')
        write_json(out/'reuse_proof.json',proof)
        args+=['--load-policy',str(Path(a.load_policy).resolve())]
    else: args+=['--save-policy',str(out/'policy.raw.pkl')]
    t0=time.monotonic(); rgb.main(args)
    src=Path(a.load_policy) if a.load_policy else out/'policy.raw.pkl'
    blob=pickle.loads(src.read_bytes()); blob.update(config=cfg,constant_pixels=constant,
        actual_steps=updates*envs*64,checkpoint_kind='rgb_policy',source_policy_sha256=digest(src))
    write_pickle(out/'final.pkl',blob)
    write_json(out/'COMPLETE.json',dict(smoke=a.smoke,actual_steps=updates*envs*64,
        final_sha256=digest(out/'final.pkl'),wall_seconds=time.monotonic()-t0,loaded=a.load_policy))
if __name__=='__main__': main()

```

## scripts/run_rgb_serial.sh

```bash
#!/bin/bash
# Explicit execution, not called by installation. One local process per GPU.
set -euo pipefail
: "${KIT:?Set installed package path}" "${PY:?Set verified local interpreter}"
mapfile -t ids < <("$PY" -c 'import json,sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1])) if r["group"]=="rgb"))' "$KIT/plan/matrix.json")
for id in "${ids[@]}"; do
  "$PY" "$KIT/scripts/agent.py" run --profile wsl_rgb --id "$id" --execute
done

```

## scripts/submit_group.sh

```bash
#!/bin/bash
# Explicit, bounded Slurm submission. Defaults to printing, not submission.
set -euo pipefail
: "${KIT:?Set installed Viper package path}"
group="${1:?core, llm_reference, llm_pilot, or llm_final}"
shift
case "$group" in core|llm_reference|llm_pilot|llm_final) ;; *) echo "Unknown group"; exit 2;; esac
smoke=0; submit=0
for arg in "$@"; do
 case "$arg" in --smoke) smoke=1;; --submit) submit=1;; *) echo "Unknown argument: $arg"; exit 2;; esac
done
OUT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs
suffix=""; [[ "$smoke" == 0 ]] || suffix=_smoke
RUN_LIST="$OUT/lists/${group}${suffix}.txt"
[[ -f "$RUN_LIST" ]] || { echo "Missing job list: $RUN_LIST"; exit 2; }
n="$(wc -l < "$RUN_LIST")"
(( n > 0 )) || exit 2
mkdir -p "$OUT/logs"
export CAMPAIGN_ROOT="$KIT" RUN_LIST SMOKE="$smoke"
limit=24:00:00; [[ "$smoke" == 0 ]] || limit=01:00:00
cmd=(sbatch --time="$limit" --array="0-$((n-1))%8" --export=ALL "$KIT/scripts/viper.sbatch")
printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$submit" == 1 ]]; then "${cmd[@]}"; fi

```

## scripts/summarize_inventory.py

```python
"""Summarize the remote metadata scan and explicitly reject unjustified run reuse."""
from __future__ import annotations
import argparse, collections, csv, io, json
from pathlib import Path
from common import ROOT, write_json, atomic

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metadata',required=True)
    ap.add_argument('--out',default=str(ROOT/'audit')); a=ap.parse_args()
    data=[json.loads(s) for s in Path(a.metadata).read_text().splitlines()]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    hashes=collections.Counter(r.get('sha256') for r in data)
    summary={'files':len(data),'bytes':sum(r['bytes'] for r in data),
      'decode_errors':[r['relative_path'] for r in data if r.get('error')],
      'unique_sha256':len(hashes),'duplicate_pairs':sum(n==2 for n in hashes.values()),
      'weight_files':sum(bool(r.get('has_weights')) for r in data),
      'weight_files_with_recorded_normalization':sum(bool(r.get('has_weights') and r.get('normalization_present')) for r in data),
      'metrics_only':[r['relative_path'] for r in data if not r.get('has_weights')],
      'unstable_during_read':[r['relative_path'] for r in data if not r.get('stable_during_read')],
      'exact_core_budget_candidates':[r['relative_path'] for r in data if r.get('actual_steps') in (117964800,32768000)],
      'rgb_name_candidates':[r['relative_path'] for r in data if any(w in r['relative_path'].lower() for w in ('rgb','pixel','camera'))],
      'approved_primary_run_reuse':0,
      'reuse_reason':'No matched core budget/snapshot cohorts; no RGB weights located; legacy LLM specifications and source/environment provenance differ. Budget alone is not a reuse proof.'}
    write_json(out/'checkpoint_summary.json',summary)
    buf=io.StringIO(); fields=['relative_path','bytes','sha256','env_name','seed','requested_steps','actual_steps','has_weights','normalization_present','commit_hash','reuse_class']
    w=csv.DictWriter(buf,fieldnames=fields); w.writeheader()
    for r in data:
        x={k:r.get(k) for k in fields}; x['reuse_class']='legacy_evaluation_candidate' if r.get('has_weights') and r.get('normalization_present') else 'needs_normalizer_provenance' if r.get('has_weights') else 'metrics_only_not_a_checkpoint'
        w.writerow(x)
    atomic(out/'checkpoint_index.csv',buf.getvalue().encode())
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

```

## scripts/train_ppo.py

```python
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

```

## scripts/train_state.py

```python
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
    def snapshot(state,stats):
        n=int(np.asarray(state.actor.n_updates)); steps=n*quantum
        payload={'config':cfg,'actual_steps':steps,'checkpoint_kind':'evaluation_only',
                 'normalization_stats':jax.device_get(stats),
                 'runner_state':{'0':{'actor':{'params':serialization.to_state_dict(state.actor.params)},
                                      'meta':{'params':serialization.to_state_dict(state.meta.params)}}}}
        write_pickle(out/'snapshots'/f'step_{steps:012d}.pkl',payload)
        print(f'SNAPSHOT {steps:,} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    def hook(state,stats):
        import jax.numpy as jnp
        def yes(_):
            jax.debug.callback(snapshot,state,stats,ordered=True)
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

```

## scripts/verify_installation.py

```python
"""Validate a freshly frozen package. No GPU imports, training, or Git writes.

The only state transition is renaming this package's INSTALLING guard after all
code/config checks pass. GPU/render/restore and production-size checks are separate.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
import yaml
from common import ROOT,digest,write_json,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--finish',action='store_true');a=p.parse_args()
    manifest_path=ROOT/'source_manifest.json'
    if not manifest_path.is_file():raise FileNotFoundError('Run freeze_sources.py into this fresh package first')
    manifest=json.loads(manifest_path.read_text())
    for rel,sha in manifest.items():
        path=ROOT/'sources'/rel
        if not path.is_file() or digest(path)!=sha:raise RuntimeError(f'Source integrity failure: {rel}')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    record=ROOT/'installation_checks'/stamp
    record.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu')
    commands=[
        [sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        [sys.executable,str(ROOT/'scripts/make_matrix.py'),'--core-repo',str(ROOT/'sources/core'),
         '--rgb-repo',str(ROOT/'sources/rgb'),'--out',str(record/'regenerated_plan')]]
    for i,cmd in enumerate(commands):
        result=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        atomic(record/f'check_{i}.log',result.stdout.encode())
        print(result.stdout)
        if result.returncode:raise RuntimeError(f'Installation check {i} failed; guard retained')
    expected=json.loads((ROOT/'plan/matrix.json').read_text())
    actual=json.loads((record/'regenerated_plan/matrix.json').read_text())
    if expected!=actual:raise RuntimeError('Regenerated matrix differs. Do not overwrite the frozen plan')
    for row in expected:
        one=yaml.safe_load((ROOT/'plan'/row['config']).read_text())
        two=yaml.safe_load((record/'regenerated_plan'/row['config']).read_text())
        if one!=two:raise RuntimeError(f'Regenerated configuration differs: {row["id"]}')
    for shell in sorted((ROOT/'scripts').glob('*.sh'))+sorted((ROOT/'scripts').glob('*.sbatch')):
        subprocess.run(['bash','-n',str(shell)],check=True)
    report={'source_files':len(manifest),'matrix_rows':len(expected),'source_manifest_sha256':digest(manifest_path),
            'static_checks':'PASS','training_executed':False,'rendering_executed':False,'model_restore_tested':False,
            'runtime_smokes_required':True,'production_authorized_by_this_check':False}
    write_json(record/'result.json',report)
    if a.finish:
        guard=ROOT/'INSTALLING';saved=ROOT/'INSTALLING.closed_after_static_checks'
        if not guard.is_file():raise RuntimeError('Expected own INSTALLING guard; no state changed')
        if saved.exists():raise FileExistsError(saved)
        guard.rename(saved)
        print('Static installation checks passed. Guard archived, not deleted. Runtime smokes remain mandatory.')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

```

## scripts/verify_legacy_weights.py

```python
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

```

## scripts/viper.sbatch

```bash
#!/bin/bash
#SBATCH --job-name=nexus_final
#SBATCH --partition=apu
#SBATCH --account=mage_apu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=108000
#SBATCH --time=24:00:00
#SBATCH --output=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/logs/%x_%A_%a.out
#SBATCH --error=/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/logs/%x_%A_%a.err
set -euo pipefail
: "${CAMPAIGN_ROOT:?Set installed package path}" "${RUN_LIST:?One matrix ID per line}"
index="${SLURM_ARRAY_TASK_ID:?Array index is required}"
RUN_ID="$(sed -n "$((index + 1))p" "$RUN_LIST")"
[ -n "$RUN_ID" ] || { echo "No run at array index $index"; exit 2; }
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disable
export PYTHONPATH=/ptmp/akalenik/nexus/site
args=("$CAMPAIGN_ROOT/scripts/agent.py" run --profile viper --id "$RUN_ID" --execute)
if [[ "${SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
# This uses exactly one allocated APU and the already installed ROCm interpreter.
# No retry loops: a failed/partial run remains visible and cannot be overwritten.
srun /ptmp/akalenik/jaxrocm_venv/bin/python "${args[@]}"

```

## scripts/write_job_lists.py

```python
"""Materialize immutable job lists; submitting them is a separate action."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common import ROOT,atomic,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    rows=json.loads((ROOT/'plan/matrix.json').read_text());out=Path(a.out)
    for group in ('core','llm_reference','llm_pilot','llm_final','rgb'):
        selected=[r for r in rows if r['group']==group]
        atomic(out/(group+'.txt'),(''.join(r['id']+'\n' for r in selected)).encode())
        seen=set();smokes=[]
        for r in selected:
            key=(r['task'],'llm' if r['spec'] else r['method'])
            if key not in seen:seen.add(key);smokes.append(r['id'])
        atomic(out/(group+'_smoke.txt'),(''.join(x+'\n' for x in smokes)).encode())
    print(out)
if __name__=='__main__':main()

```

## tests/test_static.py

```python
"""CPU tests: matrix arithmetic, schema rejection, non-destructive IO, source anchors.
These do NOT claim that MuJoCo/JAX GPU training has been executed.
"""
import ast,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import atomic,replace_once
from llm_specs import validate
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_code_parses(self):
        for path in (ROOT/'scripts').glob('*.py'): ast.parse(path.read_text(),filename=str(path))
    def test_matrix(self):
        rows=json.loads((ROOT/'plan/matrix.json').read_text()); self.assertEqual(len(rows),142)
        self.assertEqual(len({r['id'] for r in rows}),142)
        self.assertEqual(sum(r['budget'] for r in rows),7073792000)
        for r in rows:
            if r['engine']=='state': self.assertEqual(r['budget']%131072,0)
            if r['engine']=='ppo':
                unit=983040 if r['task']=='hopper' else 163840
                self.assertEqual(r['budget']%(10*unit),0)
    def test_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'keep'; atomic(path,b'original')
            with self.assertRaises(FileExistsError): atomic(path,b'changed')
            self.assertEqual(path.read_bytes(),b'original')
    def test_anchor(self):
        self.assertEqual(replace_once('abc','b','x'),'axc')
        with self.assertRaises(RuntimeError): replace_once('bb','b','x')
    def test_validator(self):
        def spec(): return {'skills':[{'name':f's{i}','activation_rule':'True' if i==0 else 'abs(torso_pitch) < 0.4',
            'reward_terms':[{'type':'positive_velocity','lhs':'forward_velocity','weight':1.}]} for i in range(3)]}
        validate(spec(),'cheetah')
        x=spec(); x['skills'][1]['reward_terms'][0]['type']='invented'
        with self.assertRaises(ValueError): validate(x,'cheetah')
        x=spec(); x['skills'][0]['reward_terms'][0]['weight']=0
        with self.assertRaises(ValueError): validate(x,'cheetah')
        x=spec(); x['skills'][1]['activation_rule']='__import__("os").system("false")'
        with self.assertRaises(ValueError): validate(x,'cheetah')
        x=spec(); x['skills'][0]['reward_terms'][0]['lhs']='unknown'
        with self.assertRaises(ValueError): validate(x,'cheetah')
if __name__=='__main__': unittest.main()

```

## deploy/viper.json

```json
{
  "python": "/ptmp/akalenik/jaxrocm_venv/bin/python",
  "results": "/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/results",
  "evidence": "/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/evidence",
  "specs": "/ptmp/akalenik/nexus/final_campaign_2026-09-05_outputs/specs",
  "capacity_check_path": "/ptmp/akalenik",
  "min_free_gib": 50,
  "pythonpath_prefix": "/ptmp/akalenik/nexus/site",
  "environment": {
    "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
    "MUJOCO_GL": "disable"
  },
  "allowed_groups": [
    "core",
    "llm_reference",
    "llm_pilot",
    "llm_final"
  ],
  "requires_slurm": true,
  "production_allowed": true
}

```

## deploy/wsl_core.json

```json
{
  "python": "/mnt/c/Users/smirn/VSCodeProjects/nexus_project/nexus_continuous_control/.venv-wsl312/bin/python",
  "results": "/home/smirn/nexus_campaign_results_2026-09-05",
  "evidence": "/home/smirn/nexus_campaign_evidence_2026-09-05",
  "specs": "/home/smirn/nexus_campaign_specs_2026-09-05",
  "gpu_lock": "/tmp/nexus_local_gpu.lock",
  "capacity_check_path": "/mnt/c",
  "min_free_gib": 50,
  "pythonpath_prefix": "",
  "environment": {},
  "requires_slurm": false,
  "allowed_groups": [
    "core",
    "llm_reference",
    "llm_pilot",
    "llm_final"
  ],
  "production_allowed": false
}

```

## deploy/wsl_rgb.json

```json
{
  "python": "/mnt/c/Users/smirn/VSCodeProjects/nexus_project/nexus_continuous_control/.venv-wsl312/bin/python",
  "results": "/home/smirn/nexus_campaign_results_2026-09-05",
  "evidence": "/home/smirn/nexus_campaign_evidence_2026-09-05",
  "specs": "/home/smirn/nexus_campaign_specs_2026-09-05",
  "gpu_lock": "/tmp/nexus_local_gpu.lock",
  "capacity_check_path": "/mnt/c",
  "min_free_gib": 50,
  "pythonpath_prefix": "",
  "environment": {
    "MUJOCO_GL": "egl"
  },
  "requires_slurm": false,
  "allowed_groups": [
    "rgb"
  ],
  "production_allowed": true
}

```

## plan/configs/core__go1__flat__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__flat__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__flat__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__flat__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__flat__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__flat__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__flat__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__flat__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__flat__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__flat__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__hpqn__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__hpqn__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__hpqn__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__hpqn__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__hpqn__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__hpqn__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__hpqn__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__hpqn__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__hpqn__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__hpqn__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__nesy__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__nesy__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__nesy__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__nesy__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__nesy__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__nesy__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__nesy__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__nesy__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__nesy__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__nesy__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__neural__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__neural__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__neural__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__neural__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__neural__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__neural__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__neural__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__neural__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__neural__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__neural__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__ppo__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__ppo__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__ppo__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__ppo__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__ppo__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__ppo__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__ppo__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__ppo__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__ppo__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__ppo__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__symbolic__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__symbolic__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__symbolic__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__symbolic__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__symbolic__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__symbolic__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__symbolic__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__symbolic__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__go1__symbolic__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__go1__symbolic__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: Go1JoystickFlatTerrain
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.25
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: go1_joystick
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: go1_joystick
TOTAL_TIMESTEPS: 32768000
USE_RGB: false

```

## plan/configs/core__hopper__flat__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__flat__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__flat__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__flat__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__flat__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__flat__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__flat__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__flat__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__flat__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__flat__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: flat
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: flat_baseline
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__hpqn__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__hpqn__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__hpqn__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__hpqn__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__hpqn__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__hpqn__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__hpqn__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__hpqn__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__hpqn__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__hpqn__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: true
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__nesy__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__nesy__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__nesy__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__nesy__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__nesy__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__nesy__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__nesy__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__nesy__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__nesy__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__nesy__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__neural__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__neural__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__neural__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__neural__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__neural__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__neural__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__neural__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__neural__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__neural__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__neural__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: neural
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__ppo__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__ppo__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__ppo__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__ppo__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__ppo__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__ppo__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__ppo__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__ppo__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__ppo__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__ppo__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: ppo
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__symbolic__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__symbolic__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__symbolic__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__symbolic__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__symbolic__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__symbolic__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__symbolic__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__symbolic__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/core__hopper__symbolic__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: core__hopper__symbolic__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: HopperHop
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.05
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: symbolic
NOISE_DECAY: 1.0
NOISE_FINISH: 0.05
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: hopper_hop
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: hopper_hop
TOTAL_TIMESTEPS: 117964800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__initial__g0__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__initial__g0__s0
CAMPAIGN_SPEC: cheetah/g0/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__initial__g0__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__initial__g0__s1
CAMPAIGN_SPEC: cheetah/g0/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__initial__g1__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__initial__g1__s0
CAMPAIGN_SPEC: cheetah/g1/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__initial__g1__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__initial__g1__s1
CAMPAIGN_SPEC: cheetah/g1/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__initial__g2__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__initial__g2__s0
CAMPAIGN_SPEC: cheetah/g2/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__initial__g2__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__initial__g2__s1
CAMPAIGN_SPEC: cheetah/g2/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__refined__g0__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__refined__g0__s0
CAMPAIGN_SPEC: cheetah/g0/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__refined__g0__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__refined__g0__s1
CAMPAIGN_SPEC: cheetah/g0/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__refined__g1__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__refined__g1__s0
CAMPAIGN_SPEC: cheetah/g1/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__refined__g1__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__refined__g1__s1
CAMPAIGN_SPEC: cheetah/g1/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__refined__g2__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__refined__g2__s0
CAMPAIGN_SPEC: cheetah/g2/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__refined__g2__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__refined__g2__s1
CAMPAIGN_SPEC: cheetah/g2/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__resample__g0__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__resample__g0__s0
CAMPAIGN_SPEC: cheetah/g0/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__resample__g0__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__resample__g0__s1
CAMPAIGN_SPEC: cheetah/g0/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__resample__g1__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__resample__g1__s0
CAMPAIGN_SPEC: cheetah/g1/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__resample__g1__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__resample__g1__s1
CAMPAIGN_SPEC: cheetah/g1/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__resample__g2__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__resample__g2__s0
CAMPAIGN_SPEC: cheetah/g2/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__cheetah__resample__g2__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__cheetah__resample__g2__s1
CAMPAIGN_SPEC: cheetah/g2/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__initial__g0__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__initial__g0__s0
CAMPAIGN_SPEC: walker/g0/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__initial__g0__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__initial__g0__s1
CAMPAIGN_SPEC: walker/g0/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__initial__g1__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__initial__g1__s0
CAMPAIGN_SPEC: walker/g1/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__initial__g1__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__initial__g1__s1
CAMPAIGN_SPEC: walker/g1/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__initial__g2__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__initial__g2__s0
CAMPAIGN_SPEC: walker/g2/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__initial__g2__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__initial__g2__s1
CAMPAIGN_SPEC: walker/g2/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__refined__g0__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__refined__g0__s0
CAMPAIGN_SPEC: walker/g0/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__refined__g0__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__refined__g0__s1
CAMPAIGN_SPEC: walker/g0/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__refined__g1__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__refined__g1__s0
CAMPAIGN_SPEC: walker/g1/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__refined__g1__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__refined__g1__s1
CAMPAIGN_SPEC: walker/g1/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__refined__g2__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__refined__g2__s0
CAMPAIGN_SPEC: walker/g2/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__refined__g2__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__refined__g2__s1
CAMPAIGN_SPEC: walker/g2/refined.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__resample__g0__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__resample__g0__s0
CAMPAIGN_SPEC: walker/g0/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__resample__g0__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__resample__g0__s1
CAMPAIGN_SPEC: walker/g0/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__resample__g1__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__resample__g1__s0
CAMPAIGN_SPEC: walker/g1/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__resample__g1__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__resample__g1__s1
CAMPAIGN_SPEC: walker/g1/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__resample__g2__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__resample__g2__s0
CAMPAIGN_SPEC: walker/g2/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_final__walker__resample__g2__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_final__walker__resample__g2__s1
CAMPAIGN_SPEC: walker/g2/resample.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_pilot__cheetah__initial__g0__s900.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_pilot__cheetah__initial__g0__s900
CAMPAIGN_SPEC: cheetah/g0/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 900
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 13107200
USE_RGB: false

```

## plan/configs/llm_pilot__cheetah__initial__g1__s901.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_pilot__cheetah__initial__g1__s901
CAMPAIGN_SPEC: cheetah/g1/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 901
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 13107200
USE_RGB: false

```

## plan/configs/llm_pilot__cheetah__initial__g2__s902.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_pilot__cheetah__initial__g2__s902
CAMPAIGN_SPEC: cheetah/g2/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 902
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 13107200
USE_RGB: false

```

## plan/configs/llm_pilot__walker__initial__g0__s900.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_pilot__walker__initial__g0__s900
CAMPAIGN_SPEC: walker/g0/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 900
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 13107200
USE_RGB: false

```

## plan/configs/llm_pilot__walker__initial__g1__s901.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_pilot__walker__initial__g1__s901
CAMPAIGN_SPEC: walker/g1/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 901
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 13107200
USE_RGB: false

```

## plan/configs/llm_pilot__walker__initial__g2__s902.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_pilot__walker__initial__g2__s902
CAMPAIGN_SPEC: walker/g2/initial.json
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: llm_generated
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 902
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 13107200
USE_RGB: false

```

## plan/configs/llm_reference__cheetah__nesy__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__cheetah__nesy__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: cheetah_run
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__cheetah__nesy__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__cheetah__nesy__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: cheetah_run
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__cheetah__nesy__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__cheetah__nesy__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: cheetah_run
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__cheetah__nesy__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__cheetah__nesy__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: cheetah_run
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__cheetah__nesy__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__cheetah__nesy__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: CheetahRun
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: cheetah_run
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: cheetah_run
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__walker__nesy__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__walker__nesy__s0
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 0
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__walker__nesy__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__walker__nesy__s1
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 1
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__walker__nesy__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__walker__nesy__s2
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 2
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__walker__nesy__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__walker__nesy__s3
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 3
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/llm_reference__walker__nesy__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_INIT_SCALE: 0.01
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_ID: llm_reference__walker__nesy__s4
CRITIC_AGG: mean
CRITIC_HIDDEN_SIZES:
- 256
- 256
CRITIC_INIT_SCALE: 1.0
ENV_NAME: WalkerWalk
EVAL_AFTER_TRAIN: false
EVAL_MAX_STEPS: 1000
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LINSPACE_NOISE: false
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_DECISION_INTERVAL: 1
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_INIT_SCALE: 1.0
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 2048
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
SAVE_PATH: null
SCALE_CLIP_BY_SKILLS: false
SEED: 4
SHARED_SKILL_REWARD: false
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 52428800
USE_RGB: false

```

## plan/configs/rgb__cartpole__constant__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__cartpole__constant__s0
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 0
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__constant__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__cartpole__constant__s1
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 1
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__constant__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__cartpole__constant__s2
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 2
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__constant__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__cartpole__constant__s3
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 3
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__constant__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__cartpole__constant__s4
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 4
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__pixels__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__pixels__s0
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 0
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__pixels__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__pixels__s1
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 1
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__pixels__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__pixels__s2
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 2
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__pixels__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__pixels__s3
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 3
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__pixels__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__pixels__s4
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 4
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__state__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__state__s0
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 0
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__state__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__state__s1
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 1
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__state__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__state__s2
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 2
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__state__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__state__s3
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 3
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__cartpole__state__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.001
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__cartpole__state__s4
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: CartpoleBalance
EVAL_AFTER_TRAIN: true
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0001
LR_DECAY: 1.0
LR_END: 2.0e-05
LR_START: 0.0001
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.02
NOISE_START: 0.3
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 8
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: warp
POLICY: cartpole_balance
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 4
SKILL_LAMBDA: 0.65
TASK_POLICY: cartpole_balance
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__constant__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__walker__constant__s0
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 0
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__constant__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__walker__constant__s1
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 1
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__constant__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__walker__constant__s2
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 2
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__constant__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__walker__constant__s3
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 3
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__constant__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: true
CAMPAIGN_ID: rgb__walker__constant__s4
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 4
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__pixels__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__pixels__s0
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 0
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__pixels__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__pixels__s1
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 1
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__pixels__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__pixels__s2
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 2
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__pixels__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__pixels__s3
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 3
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__pixels__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__pixels__s4
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: true
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 4
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__state__s0.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__state__s0
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 0
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__state__s1.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__state__s1
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 1
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__state__s2.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__state__s2
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 2
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__state__s3.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__state__s3
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 3
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```

## plan/configs/rgb__walker__state__s4.yaml

```yaml
ACTIVATION: relu
ACTOR_HIDDEN_SIZES:
- 256
- 256
ACTOR_UPDATE_MODE: all_states
ALG_NAME: nexus_ac_pqn
ANNEAL_LR: true
BEHAVIOR_PENALTY_COEFF: 0.0005
CAMPAIGN_CONSTANT_PIXELS: false
CAMPAIGN_ID: rgb__walker__state__s4
CRITIC_HIDDEN_SIZES:
- 256
- 256
ENV_NAME: WalkerWalk
EVAL_MAX_STEPS: 250
EVAL_NUM_ENVS: 64
EVAL_NUM_EPISODES: 256
EVAL_SEED: 30000
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.0003
LR_DECAY: 1.0
LR_END: 5.0e-05
LR_START: 0.0003
MAX_GRAD_NORM: 1.0
META_EPS_DECAY: 0.6
META_EPS_FINISH: 0.02
META_EPS_START: 1.0
META_HIDDEN_SIZES:
- 128
- 128
META_LAMBDA: 0.8
META_POLICY_TYPE: nesy
NOISE_DECAY: 0.8
NOISE_FINISH: 0.03
NOISE_START: 0.35
NORMALIZE_OBS: true
NORMALIZE_REWARD: false
NORM_TYPE: layer_norm
NUM_CRITICS: 2
NUM_ENVS: 128
NUM_EPOCHS: 4
NUM_MINIBATCHES: 64
NUM_SEEDS: 1
NUM_STEPS: 64
PLAYGROUND_IMPL: jax
POLICY: walker_walk
PRINT_EVERY: 0
RGB_ACTOR: false
RGB_AUGMENT: true
RGB_AUG_PAD: 4
RGB_EMBED_DIM: 128
RGB_EPISODE_LENGTH: 250
RGB_META_SEES_PIXELS: false
RGB_PROPRIO: full
RGB_SHARED_ENCODER: false
SAVE_PATH: null
SEED: 4
SKILL_LAMBDA: 0.65
TASK_POLICY: walker_walk
TOTAL_TIMESTEPS: 2048000
USE_RGB: true

```
