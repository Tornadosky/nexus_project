"""Inventory source/dependencies and user-confirmed trusted checkpoints. No writes to inputs."""
from __future__ import annotations
import argparse,json,subprocess,sys,pickle
from pathlib import Path
from common import digest,write_json

def run(cmd,cwd=None):
    r=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return {'returncode':r.returncode,'output':r.stdout}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--out',required=True)
    p.add_argument('--checkpoint-root'); p.add_argument('--trusted-checkpoints',action='store_true')
    a=p.parse_args(); root=Path(a.repo).resolve()
    paths=sorted(set(root.glob('nexus_continuous/**/*.py'))|set(root.glob('configs/*.yaml'))|set(root.glob('tools/*.py')))
    out={'repo':str(root),'source_sha256':{str(f.relative_to(root)):digest(f) for f in paths},
         'git_head':run(['git','rev-parse','HEAD'],root),'git_status':run(['git','status','--porcelain'],root),
         'submodules':run(['git','submodule','status','--recursive'],root),
         'pip_freeze':run([sys.executable,'-m','pip','freeze']),'checkpoints':[]}
    if a.checkpoint_root:
        if not a.trusted_checkpoints: raise ValueError('Pickle can execute code; explicitly confirm these are your own trusted files')
        for path in sorted(Path(a.checkpoint_root).rglob('*.pkl')):
            row={'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest(path)}
            try:
                ck=pickle.loads(path.read_bytes()); cfg=ck.get('config',{})
                row.update(keys=list(ck),has_weights=any(k in ck for k in ('runner_state','params','actor_params')),
                           has_normalizer=('normalization_stats' in ck or 'params' in ck),config=cfg,
                           actual_steps=ck.get('actual_steps'),train_budget=cfg.get('TOTAL_TIMESTEPS'),
                           seed=cfg.get('SEED',ck.get('seed')))
                del ck
            except Exception as e: row['error']=f'{type(e).__name__}: {e}'
            out['checkpoints'].append(row)
    write_json(Path(a.out),out)
    print('Sources',len(paths),'checkpoints',len(out['checkpoints']))
if __name__=='__main__': main()
