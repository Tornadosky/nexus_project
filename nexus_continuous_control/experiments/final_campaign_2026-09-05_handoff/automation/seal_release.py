"""Seal the inspected NVIDIA release only from actual, matching test receipts."""
import ast, datetime, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import agent
fingerprint=agent.verify_sources()
reports={}
proofs={}
for name,count,capacity in [('wsl_core',16,4),('wsl_rgb',6,2)]:
    path=ROOT/'verification'/f'{name}.json'
    report=json.loads(path.read_text())
    assert report['all_required_checks_passed'] is True
    assert report['source_fingerprint']==fingerprint
    assert report['runtime']==json.loads(agent.runtime_id(agent.load_profile(name)))
    assert len(report['smoke_training'])==count and len(report['capacity'])==capacity
    assert report['primary_matrix_training_started'] is False
    reports[name]=report;proofs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
rgb_path=ROOT/'verification/rgb_saved_restore.json'
rgb=json.loads(rgb_path.read_text())
assert rgb['source_fingerprint']==fingerprint and len(rgb['tests'])==6
assert all(t['saved_policy_restored'] and not t['training_executed'] for t in rgb['tests'])
proofs[str(rgb_path.relative_to(ROOT))]=hashlib.sha256(rgb_path.read_bytes()).hexdigest()
media=[]
profile=agent.load_profile('wsl_core')
for task in ('hopper','go1'):
    folder=Path(profile['results']).parent/'media_smoke'/f'core__{task}__nesy__s0'
    record=json.loads((folder/'COMPLETE.json').read_text())
    assert record['identity']['smoke'] and record['frames_recorded']>0
    assert not record['training_executed']
    for name,value in record['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==value
    media.append(dict(task=task,evidence_path=str(folder),receipt=record))
rows=json.loads((ROOT/'plan/matrix.json').read_text())
assert len(rows)==142
for row in rows:
    base=Path(agent.load_profile('wsl_rgb' if row['group']=='rgb' else 'wsl_core')['results'])
    assert not (base/row['id']/'COMPLETE.json').exists(),'Primary training was already started'
for file in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'automation').glob('*.py')):
    ast.parse(file.read_text(),filename=str(file))
for file in sorted(ROOT.rglob('*')):
    if file.is_file() and file.suffix in ('.sh','.sbatch') and 'sources' not in file.parts:
        subprocess.run(['bash','-n',str(file)],check=True)
subprocess.run(['bash',str(ROOT/'RUN_ALL.sh')],check=True)
assert agent.load_profile('viper')['production_allowed'] is False
p=ROOT/'deploy/wsl_core.json';profile=json.loads(p.read_text())
profile['production_allowed']=True;p.write_text(json.dumps(profile,indent=2)+'\n')
media_path=ROOT/'verification/media_smoke.json'
media_path.write_text(json.dumps(dict(tests=media,training_executed=False),indent=2))
proofs[str(media_path.relative_to(ROOT))]=hashlib.sha256(media_path.read_bytes()).hexdigest()
ready=dict(production_ready=True,scope='Inspected NVIDIA machine; Viper primary training is disabled',
    source_fingerprint=fingerprint,primary_matrix_training_started=False,
    maximum_primary_training_runs=142,smoke_training_families=22,production_shape_capacity_tests=6,
    saved_core_evaluation_cases=sum(len(x['tests']) for x in reports['wsl_core']['restore_tests']),
    saved_rgb_policy_reload_tests=6,real_simulation_media_tests=2,
    proof_sha256=proofs,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path=ROOT/'READY.json'
if path.exists():
    previous=json.loads(path.read_text());ready['created_utc']=previous['created_utc'];assert ready==previous
else:path.write_text(json.dumps(ready,indent=2))
# A complete copyable source/configuration document; generated deterministically.
excluded={'__pycache__','installation_checks','.git'}
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts)]
parts=['# Complete executable release source and configurations\n\n',
       'The verified launch sequence is in `docs/RUNBOOK.md`. Scientific parameters are fixed in `plan/`.\n\n']
languages={'.py':'python','.sh':'bash','.sbatch':'bash','.yaml':'yaml','.yml':'yaml','.toml':'toml','.json':'json'}
for file in files:
    relative=file.relative_to(ROOT)
    if file.suffix not in languages:continue
    if file.suffix=='.json' and relative.parts[0]!='deploy':continue
    parts.extend([f'## `{relative.as_posix()}`\n\n```{languages[file.suffix]}\n',
                  file.read_text().rstrip()+'\n```\n\n'])
(ROOT/'ALL_CODE.md').write_text(''.join(parts))
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts)]
manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n'
                 for p in files if p.name!='SHA256SUMS.txt')
(ROOT/'SHA256SUMS.txt').write_text(manifest)
subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS.txt'],cwd=ROOT,check=True)
print(json.dumps(ready,indent=2))
print('RELEASE_SEALED',len(manifest.splitlines()),'files',flush=True)
