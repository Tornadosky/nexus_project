"""Restore all six saved image-study smoke policies; no training occurs."""
import fcntl, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import agent
from common import digest
profile=agent.load_profile('wsl_rgb')
env=agent.base_env(profile);env.pop('XLA_FLAGS',None)
env['PYTHONUNBUFFERED']='1'
base=Path(profile['results'])
output=base.parent/'saved_policy_restore'
rows=json.loads((ROOT/'plan/matrix.json').read_text())
ids=[i for i in json.loads((ROOT/'plan/smoke_ids.json').read_text()) if i.startswith('rgb__')]
records=[]
for ident in ids:
    row=next(r for r in rows if r['id']==ident)
    checkpoint=base/(ident+'__smoke')/'final.pkl'
    dest=output/ident
    command=[profile['python'],str(ROOT/'scripts/rgb_run.py'),
        '--repo',str(ROOT/'sources/rgb'),'--config',str(ROOT/'plan'/row['config']),
        '--load-policy',str(checkpoint),'--smoke','--out',str(dest)]
    output.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with open(profile['gpu_lock'],'a') as lock, (output/(ident+'.log')).open('x') as log:
            fcntl.flock(lock,fcntl.LOCK_EX)
            subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    complete=json.loads((dest/'COMPLETE.json').read_text())
    assert complete['smoke'] and complete['loaded']==str(checkpoint)
    assert digest(dest/'final.pkl')==complete['final_sha256']
    evaluation=dest/'evaluation/pixel_ablation.json'
    json.loads(evaluation.read_text())
    records.append(dict(run_id=ident,checkpoint_sha256=digest(checkpoint),
        evaluation_sha256=digest(evaluation),saved_policy_restored=True,
        training_executed=False))
    print('SAVED_RGB_RESTORE_PASS',ident,flush=True)
report=dict(tests=records,source_fingerprint=agent.verify_sources(),
            runtime=json.loads(agent.runtime_id(profile)),training_executed=False)
receipt=ROOT/'verification/rgb_saved_restore.json'
if receipt.exists() and json.loads(receipt.read_text())!=report:
    raise RuntimeError('Different existing receipt must be preserved before re-verifying')
receipt.write_text(json.dumps(report,indent=2))
print('ALL_SIX_SAVED_RGB_POLICIES_RESTORED',flush=True)
