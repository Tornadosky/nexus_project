"""Real checkpoint restore and intervention tests on an allocated accelerator."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
profile = os.environ.get('NEXUS_TEST_PROFILE','viper')
if profile == 'viper' and not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Requires a Slurm compute allocation')
rows = json.loads((ROOT/'plan/matrix.json').read_text())
row = next(r for r in rows if r['id'] == sys.argv[1])
sys.path.insert(0,str(ROOT/'scripts'))
import agent
base = Path(agent.load_profile(profile)['results']).parent
run = Path(agent.load_profile(profile)['results'])/(row['id']+'__smoke')
assert json.loads((run/'COMPLETE.json').read_text())['smoke']
out = base/'release_evaluations'/row['id']
checks = [('native', run/'final.pkl', [])]
snaps = sorted((run/'snapshots').glob('step_*.pkl'))
if snaps:
    checks.append(('initial_snapshot', snaps[0], []))
if row['method'] == 'nesy':
    checks += [(label, run/'final.pkl', args) for label, args in [
        ('force0', ['--force','0']), ('delete0',['--remove','0']),
        ('unmasked',['--selector','unmasked']),
        ('symbolic_selector',['--selector','symbolic'])]]
    if row['task'] == 'go1':
        checks += [('stop',run/'final.pkl',['--command-range','0','0','0']),
                   ('rough',run/'final.pkl',['--env-name','Go1JoystickRoughTerrain'])]
results = []
for label, checkpoint, extras in checks:
    dest = out/label
    cmd = [sys.executable,str(ROOT/'scripts/evaluate.py'),'--repo',str(ROOT/'sources/core'),
           '--checkpoint',str(checkpoint),'--out',str(dest),'--episodes','4',
           '--num-envs','4','--max-steps','8','--seed','41000',*extras]
    out.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with (out/(label+'.log')).open('x') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    metadata = json.loads((dest/'metadata.json').read_text())
    assert metadata['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with np.load(dest/'episodes.npz') as arrays:
        for key in arrays.files:
            value = arrays[key]
            if value.shape != (4,):
                raise ValueError(f'{label}/{key}: expected four episodes, got {value.shape}')
            if not np.isfinite(value).all():
                raise ValueError(f'{label}/{key}: nonfinite values')
    results.append(dict(test=label, pass_restore_and_rollout=True))
    print(row['id'], label, 'PASS', flush=True)
(out/'PASS.json').write_text(json.dumps(dict(run_id=row['id'], tests=results,
    smoke_only=True, learning_quality_evaluated=False), indent=2))
