"""Collect actual completed tests and verify their checkpoint/source identities."""
import hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import agent
from common import digest
profile=sys.argv[1]
assert profile in ('viper','wsl_rgb')
p=agent.load_profile(profile)
base=Path(p['results']); fingerprint=agent.verify_sources()
runtime=agent.runtime_id(p)
rows=json.loads((ROOT/'plan/matrix.json').read_text())
ids=json.loads((ROOT/'plan/smoke_ids.json').read_text())
ids=[i for i in ids if i.startswith('rgb__') == (profile=='wsl_rgb')]
report=dict(profile=profile,source_fingerprint=fingerprint,runtime=json.loads(runtime),
            smoke_training=[],restore_tests=[],capacity=[])
for ident in ids:
 row=next(r for r in rows if r['id']==ident)
 folder=base/(ident+'__smoke')
 complete=json.loads((folder/'COMPLETE.json').read_text())
 gate=json.loads((base/'_gates'/(agent.smoke_key(row)+'.json')).read_text())
 assert complete['smoke'] and gate['source_fingerprint']==fingerprint
 assert gate['runtime']==runtime
 assert digest(Path(gate['checkpoint']))==gate['checkpoint_sha256']
 report['smoke_training'].append(dict(run_id=ident,complete=complete,gate=gate))
 if profile=='viper':
  evaluation=base.parent/'release_evaluations'/ident
  tests=json.loads((evaluation/'PASS.json').read_text())
  for item in tests['tests']:
   meta=json.loads((evaluation/item['test']/'metadata.json').read_text())
   assert digest(Path(meta['checkpoint']))==meta['checkpoint_sha256']
   assert item['pass_restore_and_rollout']
  report['restore_tests'].append(tests)
 else:
  data=json.loads((folder/'evaluation/pixel_ablation.json').read_text())
  report['restore_tests'].append(dict(run_id=ident,
      evaluation_sha256=digest(folder/'evaluation/pixel_ablation.json'),
      saved_checkpoint=gate['checkpoint_sha256'],evaluation_keys=sorted(data)))
capacity=(base.parent/'capacity' if profile=='viper' else
          Path('/home/smirn/nexus_campaign_verified_outputs/capacity'))
for variant in (('nesy','ppo') if profile=='viper' else ('cartpole','walker')):
 path=capacity/variant/'PASS.json'
 if not path.exists() and profile=='viper':
  path=capacity/(variant+'_retry')/'PASS.json'
 record=json.loads(path.read_text())
 record['evidence_path']=str(path)
 assert record['scientific_sample'] is False
 report['capacity'].append(record)
if profile=='wsl_rgb':
 model=Path('/home/smirn/nexus_finalize_2026-09-05_17-42/model_inference.json')
 report['model_inference']=json.loads(model.read_text())
 assert report['model_inference']['pass_inference']
report['all_required_checks_passed']=True
report['primary_matrix_training_started']=False
out=ROOT/'verification'/f'{profile}.json'
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists() and json.loads(out.read_text())!=report:
 raise RuntimeError('Existing verification changed; retain it before auditing again')
out.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(profile=profile,smokes=len(report['smoke_training']),
    restores=len(report['restore_tests']),capacity=len(report['capacity']),
    all_required_checks_passed=True),indent=2))
