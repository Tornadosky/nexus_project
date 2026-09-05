"""Keep all planned rows; schedule only specifications passing bounded validation."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import digest
from llm_specs import validate
p=argparse.ArgumentParser()
p.add_argument('--group',choices=('llm_pilot','llm_final'),required=True)
p.add_argument('--specs',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
all_rows=json.loads((ROOT/'plan/matrix.json').read_text())
pin=json.loads((ROOT/'deploy/model_pin.json').read_text())
selected=[]; excluded=[]; provenance={}
for row in all_rows:
 if row['group']!=a.group:continue
 spec=a.specs/row['spec']; history=spec.with_suffix('.generation.json')
 skip=spec.with_suffix('.SKIPPED.json')
 if history.exists():
  record=json.loads(history.read_text())
  assert record['model']['revision']==pin['revision']
  assert (record['task'],record['family'],record['condition']) == (
      row['task'],row['proposal_family'],row['method'])
  provenance[str(history)]=digest(history)
  if record['valid']:
   validate(json.loads(spec.read_text()),row['task'])
   provenance[str(spec)]=digest(spec);selected.append(row)
  else:
   excluded.append(dict(id=row['id'],reason='bounded_validation_failed'))
 elif skip.exists():
  marker=json.loads(skip.read_text())
  assert row['method']=='refined' and marker['reason']=='initial_failed_bounded_validation'
  initial=spec.parent/'initial.generation.json'
  assert digest(initial)==marker['dependency_sha256']
  assert not json.loads(initial.read_text())['valid']
  provenance[str(skip)]=digest(skip)
  excluded.append(dict(id=row['id'],reason=marker['reason']))
 else:
  raise FileNotFoundError(f'Missing specification provenance, not a negative result: {spec}')
a.out.mkdir(parents=True,exist_ok=True)
def save(name,value):
 target=a.out/name
 text=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
 if target.exists():
  if target.read_text()!=text:raise RuntimeError(f'Selection changed: {target}')
 else:target.write_text(text)
save(a.group+'.txt',''.join(r['id']+'\n' for r in selected))
save(a.group+'.json',selected)
evaluation=selected+([r for r in all_rows if r['group']=='llm_reference']
                     if a.group=='llm_final' else [])
save(a.group+'_evaluation.json',evaluation)
save(a.group+'_disposition.json',dict(planned=len(selected)+len(excluded),
    executable=len(selected),excluded=excluded,provenance=provenance,
    replacement_samples_allowed=False))
print(json.dumps(dict(group=a.group,executable=len(selected),excluded=excluded)),flush=True)
