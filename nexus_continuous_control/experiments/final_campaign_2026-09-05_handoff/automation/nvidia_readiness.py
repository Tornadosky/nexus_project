"""Run all 16 non-vision API smokes and restore checks on the actual NVIDIA profile."""
import fcntl, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import agent
profile=agent.load_profile('wsl_core')
env=agent.base_env(profile);env['NEXUS_TEST_PROFILE']='wsl_core';env['PYTHONUNBUFFERED']='1'
rows={r['id']:r for r in json.loads((ROOT/'plan/matrix.json').read_text())}
ids=[i for i in json.loads((ROOT/'plan/smoke_ids.json').read_text()) if not i.startswith('rgb__')]
for ident in ids:
    if rows[ident]['group'].startswith('llm_'):
        command=[sys.executable,str(ROOT/'automation/llm_smoke.py'),ident]
    else:
        command=[sys.executable,str(ROOT/'scripts/agent.py'),'run','--profile','wsl_core','--id',ident,'--smoke','--execute']
    subprocess.run(command,env=env,check=True)
    with open(profile['gpu_lock'],'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run([sys.executable,str(ROOT/'automation/eval_smoke.py'),ident],env=env,check=True)
    print('READINESS_PASS',ident,flush=True)
print('ALL_16_NVIDIA_METHOD_AND_RESTORE_CHECKS_PASSED',flush=True)
