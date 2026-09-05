"""Bounded readiness diagnostic; its outputs are never scientific campaign data."""
import faulthandler, runpy, sys
from pathlib import Path
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
root = Path(__file__).resolve().parents[1]
out = sys.argv[1]
sys.path.insert(0, str(root/'scripts'))
sys.argv = ['train_state.py','--repo',str(root/'sources/core'),
    '--config',str(root/'plan/configs/core__hopper__flat__s0.yaml'),
    '--out',out,'--smoke']
runpy.run_path(str(root/'scripts/train_state.py'),run_name='__main__')
