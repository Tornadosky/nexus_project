"""Create missing lists, verify existing lists, never overwrite a changed plan."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import atomic
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
rows=json.loads((ROOT/'plan/matrix.json').read_text())
for group in ('core','llm_reference','llm_pilot','llm_final','rgb'):
 selected=[r for r in rows if r['group']==group]
 seen=set();smokes=[]
 for r in selected:
  key=(r['task'],'llm' if r['spec'] else r['method'])
  if key not in seen:seen.add(key);smokes.append(r['id'])
 for suffix,ids in [('',[r['id'] for r in selected]),('_smoke',smokes)]:
  path=a.out/(group+suffix+'.txt');payload=''.join(i+'\n' for i in ids).encode()
  if path.exists():
   if path.read_bytes()!=payload:raise RuntimeError(f'Job list changed: {path}')
  else:atomic(path,payload)
print('ALL_JOB_LISTS_VERIFIED',a.out)
