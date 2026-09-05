"""Execute the predeclared checkpoint-only suites; never trains or selects winners."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from common import ROOT,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--matrix',required=True); p.add_argument('--repo',required=True)
    p.add_argument('--results',required=True); p.add_argument('--out',required=True)
    p.add_argument('--suite',choices=('curves','probes','shifts','pilot','llm'),required=True)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0); a=p.parse_args()
    rows=json.loads(Path(a.matrix).read_text()); jobs=[]; results=Path(a.results).resolve()
    def job(row,label,ck,episodes=128,extra=(),seed=30000):
        jobs.append((row,label,ck,episodes,list(extra),seed))
    for r in rows:
        rd=results/r['id']; ck=rd/'final.pkl'
        if a.suite=='curves' and r['group']=='core':
            snapshots=sorted((rd/'snapshots').glob('step_*.pkl'))
            if len(snapshots)!=11: raise RuntimeError(f'{rd}: expected 11 snapshots, found {len(snapshots)}')
            for cp in snapshots:
                steps=int(cp.stem.split('_')[1]); job(r,cp.stem,cp,256 if steps==r['budget'] else 64)
        elif a.suite=='probes' and r['group']=='core' and r['method'] in ('hpqn','neural','symbolic','nesy'):
            job(r,'native',ck)
            for i in range(4): job(r,f'force_{i}',ck,extra=('--force',str(i)))
            if r['method'] in ('neural','nesy'):
                for i in range(4): job(r,f'remove_{i}',ck,extra=('--remove',str(i)))
            if r['method']=='nesy':
                for sel in ('unmasked','symbolic'): job(r,f'selector_{sel}',ck,extra=('--selector',sel))
        elif a.suite=='shifts' and r['group']=='core':
            for n in (0.,.05,.1,.2): job(r,f'noise_{n:g}',ck,extra=('--noise',str(n)))
            if r['task']=='go1':
                conditions={'stop':(0.,0.,0.),'half':(.75,.4,.6),'high':(2.25,1.2,1.8),
                            'forward':(1.5,0.,0.),'yaw':(0.,0.,1.2)}
                for name,vals in conditions.items(): job(r,name,ck,extra=('--command-range',*map(str,vals)))
                job(r,'rough',ck,extra=('--env-name','Go1JoystickRoughTerrain'))
        elif a.suite=='pilot' and r['group']=='llm_pilot': job(r,'validation',ck,64,seed=20000)
        elif a.suite=='llm' and r['group'] in ('llm_final','llm_reference'): job(r,'test',ck,256)
    if not 0<=a.shard_index<a.shards: raise ValueError('Invalid shard')
    jobs=[j for n,j in enumerate(jobs) if n%a.shards==a.shard_index]
    if not jobs: raise ValueError('No jobs')
    for row,label,ck,n,extras,seed in jobs:
        if not ck.exists(): raise FileNotFoundError(f'Missing required weights: {ck}. Do not replace with training metrics.')
        dest=Path(a.out).resolve()/a.suite/row['id']/label
        if dest.exists():
            m=dest/'metadata.json'; data=dest/'episodes.npz'; s=dest/'summary.json'
            if all(f.exists() for f in (m,data,s)) and json.loads(m.read_text())['checkpoint_sha256']==digest(ck):
                print('HAVE VERIFIED EVALUATION',dest); continue
            raise RuntimeError(f'Partial or stale evaluation at {dest}; preserve it and use a separate --out')
        cmd=[sys.executable,str(ROOT/'scripts/evaluate.py'),'--repo',str(Path(a.repo).resolve()),
             '--checkpoint',str(ck),'--out',str(dest),'--episodes',str(n),
             '--num-envs','64','--seed',str(seed),*extras]
        print(__import__('shlex').join(cmd),flush=True)
        if not a.dry_run: subprocess.run(cmd,check=True,cwd=Path(a.repo).resolve())
if __name__=='__main__': main()
