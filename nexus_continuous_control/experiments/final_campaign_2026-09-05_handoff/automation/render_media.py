"""Fixed-seed real simulation videos, frames and decision traces; no training."""
import argparse, fcntl, inspect, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import add_repo, digest, replace_once
import agent
parser=argparse.ArgumentParser()
parser.add_argument('--task',choices=('hopper','go1'),required=True)
parser.add_argument('--smoke',action='store_true')
a=parser.parse_args()
p=agent.load_profile('wsl_core')
base=Path(p['results']);ident=f'core__{a.task}__nesy__s0'
checkpoint=base/(ident+('__smoke' if a.smoke else ''))/'final.pkl'
assert checkpoint.is_file(),checkpoint
out=base.parent/('media_smoke' if a.smoke else 'media')/ident
expected=dict(checkpoint_sha256=digest(checkpoint),seed=41000,
              requested_steps=4 if a.smoke else 1000,smoke=a.smoke)
if (out/'COMPLETE.json').exists():
    assert json.loads((out/'COMPLETE.json').read_text())['identity']==expected
    print('PRESERVED_FIXED_SEED_MEDIA',out);raise SystemExit(0)
if out.exists():raise RuntimeError(f'Partial media preserved: {out}')
out.mkdir(parents=True)
lock=open(p['gpu_lock'],'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
add_repo(str(ROOT/'sources/core'))
import numpy as np
import render_rollout as renderer
from PIL import Image
records=[];saved=[]
def decision(step,obs,meta,params,policy):
    state=renderer.get_actor_obs(obs);raw=renderer.get_policy_obs(obs)
    q=np.asarray(meta.apply({'params':params},state))[0]
    eligible=np.asarray(policy.skill_mask(raw))[0].astype(bool)
    assert np.isfinite(q).all()
    records.append(dict(step=step,meta_q=q.tolist(),eligible=eligible.tolist(),
                        observation_timing='before selected action'))
def frame(step,pixels,skill):
    records[-1]['selected_skill']=skill
    ambiguous=sum(records[-1]['eligible'])>1
    if not saved or (ambiguous and step-saved[-1]>=60 and len(saved)<4):
        Image.fromarray(pixels).save(out/f'frame_{step:04d}.png')
        saved.append(step)
source=inspect.getsource(renderer.main)
source=replace_once(source,'        action, skill, hold = select(obs, hold)',
    '        _record_decision(t, obs, meta_q, meta_params, policy_module)\n        action, skill, hold = select(obs, hold)')
source=replace_once(source,'        skills.append(skill_i)',
    '        skills.append(skill_i)\n        _record_frame(t, frames[-1], skill_i)')
namespace=dict(renderer.__dict__)
namespace.update(_record_decision=decision,_record_frame=frame)
exec(compile(source,'<fixed-seed-decision-rendering>','exec'),namespace)
sys.argv=[str(ROOT/'sources/core/tools/render_rollout.py'),'--checkpoint',str(checkpoint),
    '--out',str(out/'rollout.mp4'),'--seed','41000','--strip',
    '--steps',str(expected['requested_steps'])]
assert namespace['main']()==0
assert records and saved and (out/'rollout.mp4').stat().st_size>0
(out/'decisions.json').write_text(json.dumps(records,indent=2))
receipt=dict(identity=expected,frames_recorded=len(records),selected_frame_steps=saved,
    image_selection='first frame, then first ambiguous decisions at least 60 steps apart, at most four',
    rendered_state_timing='after the recorded selected action',training_executed=False,
    files={f.name:digest(f) for f in out.iterdir() if f.is_file()})
(out/'COMPLETE.json').write_text(json.dumps(receipt,indent=2))
print('FIXED_SEED_MEDIA_PASS',a.task,len(records),flush=True)
lock.close()
