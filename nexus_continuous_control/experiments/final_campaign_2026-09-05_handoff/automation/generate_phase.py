"""Run a fixed LLM phase; retain bounded validation failures without resampling."""
import fcntl, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import digest, write_json
from llm_specs import validate
mutex=open('/tmp/nexus_llm_generation.lock','a')
fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
phase=sys.argv[1]
assert phase in ('initial','resample','refined')
base=Path('/home/smirn/nexus_campaign_specs_2026-09-05')
evidence=Path('/mnt/d/nexus_final_campaign_2026-09-05/evidence')
lock=base/'model_lock.json'
model=json.loads(lock.read_text())
pin=json.loads((ROOT/'deploy/model_pin.json').read_text())
assert model['revision']==pin['revision'] and model['model']==pin['model']
rows=json.loads((ROOT/'plan/matrix.json').read_text())
for task in ('cheetah','walker'):
 for family in range(3):
  out=base/task/f'g{family}'/(phase+'.json')
  history=out.with_suffix('.generation.json')
  skipped=out.with_suffix('.SKIPPED.json')
  if history.exists():
   old=json.loads(history.read_text())
   assert old['model']['revision']==pin['revision']
   if old['valid']:
    validate(json.loads(out.read_text()),task)
   print('PRESERVED_RECORDED_SAMPLE',out,'valid=',old['valid'],flush=True)
   continue
  if skipped.exists():
   print('PRESERVED_DEPENDENCY_FAILURE',skipped,flush=True);continue
  if out.exists():
   raise RuntimeError(f'Output without provenance: {out}')
  cmd=[sys.executable,str(ROOT/'scripts/llm_specs.py'),'generate',
       '--model-lock',str(lock),'--task',task,'--family',str(family),
       '--condition',phase,'--out',str(out)]
  if phase=='refined':
   initial=out.parent/'initial.json'
   initial_record=json.loads(initial.with_suffix('.generation.json').read_text())
   if not initial_record['valid']:
    write_json(skipped,dict(reason='initial_failed_bounded_validation',
        dependency_sha256=digest(initial.with_suffix('.generation.json')),
        task=task,family=family,condition=phase,scientific_call_made=False))
    print('SKIPPED_DEPENDENT_REFINEMENT',task,family,flush=True);continue
   validate(json.loads(initial.read_text()),task)
   pilot=next(r for r in rows if r['group']=='llm_pilot' and
              r['task']==task and r['proposal_family']==family)
   feedback=evidence/'pilot'/pilot['id']/'validation/summary.json'
   if not feedback.exists():
    raise FileNotFoundError(f'Finish pilot evaluation before refinement: {feedback}')
   cmd+=['--initial',str(initial),'--feedback',str(feedback)]
  env=os.environ.copy()
  env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
  proc=subprocess.run(cmd,env=env)
  if proc.returncode:
   if history.exists() and not json.loads(history.read_text())['valid']:
    print('RETAINED_GENERATION_FAILURE',out,flush=True);continue
   raise RuntimeError(f'Infrastructure failure; no sample replaced: {out}')
print('PHASE_COMPLETE',phase,flush=True)
