"""Validate a freshly frozen package. No GPU imports, training, or Git writes.

The only state transition is renaming this package's INSTALLING guard after all
code/config checks pass. GPU/render/restore and production-size checks are separate.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
import yaml
from common import ROOT,digest,write_json,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--finish',action='store_true');a=p.parse_args()
    manifest_path=ROOT/'source_manifest.json'
    if not manifest_path.is_file():raise FileNotFoundError('Run freeze_sources.py into this fresh package first')
    manifest=json.loads(manifest_path.read_text())
    for rel,sha in manifest.items():
        path=ROOT/'sources'/rel
        if not path.is_file() or digest(path)!=sha:raise RuntimeError(f'Source integrity failure: {rel}')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    record=ROOT/'installation_checks'/stamp
    record.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu')
    commands=[
        [sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        [sys.executable,str(ROOT/'scripts/make_matrix.py'),'--core-repo',str(ROOT/'sources/core'),
         '--rgb-repo',str(ROOT/'sources/rgb'),'--out',str(record/'regenerated_plan')]]
    for i,cmd in enumerate(commands):
        result=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        atomic(record/f'check_{i}.log',result.stdout.encode())
        print(result.stdout)
        if result.returncode:raise RuntimeError(f'Installation check {i} failed; guard retained')
    expected=json.loads((ROOT/'plan/matrix.json').read_text())
    actual=json.loads((record/'regenerated_plan/matrix.json').read_text())
    if expected!=actual:raise RuntimeError('Regenerated matrix differs. Do not overwrite the frozen plan')
    for row in expected:
        one=yaml.safe_load((ROOT/'plan'/row['config']).read_text())
        two=yaml.safe_load((record/'regenerated_plan'/row['config']).read_text())
        if one!=two:raise RuntimeError(f'Regenerated configuration differs: {row["id"]}')
    for shell in sorted((ROOT/'scripts').glob('*.sh'))+sorted((ROOT/'scripts').glob('*.sbatch')):
        subprocess.run(['bash','-n',str(shell)],check=True)
    report={'source_files':len(manifest),'matrix_rows':len(expected),'source_manifest_sha256':digest(manifest_path),
            'static_checks':'PASS','training_executed':False,'rendering_executed':False,'model_restore_tested':False,
            'runtime_smokes_required':True,'production_authorized_by_this_check':False}
    write_json(record/'result.json',report)
    if a.finish:
        guard=ROOT/'INSTALLING';saved=ROOT/'INSTALLING.closed_after_static_checks'
        if not guard.is_file():raise RuntimeError('Expected own INSTALLING guard; no state changed')
        if saved.exists():raise FileExistsError(saved)
        guard.rename(saved)
        print('Static installation checks passed. Guard archived, not deleted. Runtime smokes remain mandatory.')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
