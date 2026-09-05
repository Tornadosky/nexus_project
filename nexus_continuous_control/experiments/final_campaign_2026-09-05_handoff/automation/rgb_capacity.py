"""Two-update 128-environment memory test, not a scientific training run."""
import fcntl, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from common import add_repo
add_repo(str(ROOT/'sources/rgb'))
lock = open('/tmp/nexus_local_gpu.lock','a')
fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['MUJOCO_GL']='egl'
from nexus_continuous.scripts import rgb_pixel_ablation as rgb
import jax
name = sys.argv[1]
assert name in ('cartpole','walker')
out = Path('/home/smirn/nexus_campaign_verified_outputs/capacity')/name
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
t0 = time.monotonic()
rgb.main(['--config',str(ROOT/f'plan/configs/rgb__{name}__pixels__s0.yaml'),
          '--meta','nesy','--seed','7000','--updates','2','--num-envs','128',
          '--episodes','2','--eval-steps','8','--save-policy',str(out/'probe.pkl'),
          '--out',str(out/'evaluation')])
stats = jax.devices()[0].memory_stats()
(out/'PASS.json').write_text(json.dumps(dict(task=name,updates=2,num_envs=128,
    scientific_sample=False,wall_seconds=time.monotonic()-t0,memory_stats=stats),indent=2))
print('FULL_SIZE_RGB_MEMORY_PASS',name,stats,flush=True)
