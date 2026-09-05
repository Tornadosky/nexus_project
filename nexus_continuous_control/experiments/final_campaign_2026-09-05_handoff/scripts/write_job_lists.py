"""Materialize immutable job lists; submitting them is a separate action."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common import ROOT,atomic,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    rows=json.loads((ROOT/'plan/matrix.json').read_text());out=Path(a.out)
    for group in ('core','llm_reference','llm_pilot','llm_final','rgb'):
        selected=[r for r in rows if r['group']==group]
        atomic(out/(group+'.txt'),(''.join(r['id']+'\n' for r in selected)).encode())
        seen=set();smokes=[]
        for r in selected:
            key=(r['task'],'llm' if r['spec'] else r['method'])
            if key not in seen:seen.add(key);smokes.append(r['id'])
        atomic(out/(group+'_smoke.txt'),(''.join(x+'\n' for x in smokes)).encode())
    print(out)
if __name__=='__main__':main()
