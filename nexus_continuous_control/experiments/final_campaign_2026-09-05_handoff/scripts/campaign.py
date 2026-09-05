"""Non-destructive launcher. Groups and rows are frozen in matrix.json.

Use one process per accelerator. Existing runs are skipped ONLY after a verified
COMPLETE record. A crashed/partial directory is never silently treated as success.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from common import ROOT,digest,write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--matrix',required=True)
    p.add_argument('--core-repo'); p.add_argument('--rgb-repo'); p.add_argument('--results',required=True)
    p.add_argument('--specs'); p.add_argument('--groups',default='core')
    p.add_argument('--row',type=int); p.add_argument('--id'); p.add_argument('--list',action='store_true')
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--shard-index',type=int,default=0); p.add_argument('--shards',type=int,default=1)
    a=p.parse_args(); matrix=Path(a.matrix).resolve(); rows=json.loads(matrix.read_text())
    if not 0<=a.shard_index<a.shards: raise ValueError('Invalid shard')
    indexed=[(i,r) for i,r in enumerate(rows) if r['group'] in a.groups.split(',')]
    if a.row is not None: indexed=[(i,r) for i,r in indexed if i==a.row]
    if a.id is not None: indexed=[(i,r) for i,r in indexed if r['id']==a.id]
    indexed=[v for n,v in enumerate(indexed) if n%a.shards==a.shard_index]
    if not indexed: raise ValueError('No matching jobs')
    for i,row in indexed:
        if a.list: print(i,row['id'],row['budget']); continue
        repo=a.rgb_repo if row['repo']=='rgb' else a.core_repo
        if not repo: raise ValueError(f'Missing --{row["repo"]}-repo')
        folder=Path(a.results).resolve()/row['id']
        if a.smoke: folder=folder.with_name(folder.name+'__smoke')
        complete=folder/'COMPLETE.json'
        if folder.exists():
            if complete.exists():
                c=json.loads(complete.read_text())
                expected=c['actual_steps'] if a.smoke else row['budget']
                if c['actual_steps']!=expected or bool(c.get('smoke'))!=a.smoke or digest(folder/'final.pkl')!=c['final_sha256']:
                    raise RuntimeError(f'Completion verification failed: {folder}')
                print('VERIFIED COMPLETE',row['id']); continue
            raise RuntimeError(f'Partial directory {folder}. Keep it; use a separate results directory for a documented retry.')
        script={'state':'train_state.py','ppo':'train_ppo.py','rgb':'rgb_run.py'}[row['engine']]
        cmd=[sys.executable,str(ROOT/'scripts'/script),'--repo',str(Path(repo).resolve()),
             '--config',str(matrix.parent/row['config']),'--out',str(folder)]
        if row['spec']:
            if not a.specs: raise ValueError('This row requires --specs')
            spec=Path(a.specs)/row['spec']
            if not spec.exists(): raise FileNotFoundError(spec)
            cmd+=['--specs',str(Path(a.specs).resolve())]
        if a.smoke: cmd+=['--smoke']
        print('COMMAND',__import__('shlex').join(cmd),flush=True)
        if not a.dry_run:
            subprocess.run(cmd,check=True,cwd=Path(repo).resolve())
if __name__=='__main__': main()
