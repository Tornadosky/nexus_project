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
    program = "import json,sys,os,importlib.metadata as m; names=['jax','jaxlib','flax','optax','mujoco','mujoco-mjx','brax','playground','numpy','warp-lang']; print(json.dumps({'python':sys.version,'executable':sys.executable,'xla_flags':os.environ.get('XLA_FLAGS',''),'packages':{n:m.version(n) for n in names}},sort_keys=True))"
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
