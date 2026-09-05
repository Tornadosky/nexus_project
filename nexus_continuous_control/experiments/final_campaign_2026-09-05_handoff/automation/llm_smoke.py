"""Allocated synthetic API tests, separate from scientific LLM generations."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import agent
from llm_specs import validate
test_profile = os.environ.get('NEXUS_TEST_PROFILE','viper')
if test_profile == 'viper' and not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Requires a Slurm compute allocation')
row_id = sys.argv[1]
rows = json.loads((ROOT / 'plan/matrix.json').read_text())
row = next(r for r in rows if r['id'] == row_id)
assert row['group'] in ('llm_pilot', 'llm_reference')
fixture_root = ROOT / 'automation/fixtures'
for task, count in (('cheetah', 3), ('walker', 4)):
    spec = {'skills': [{'name': f'api_test_{i}', 'activation_rule': 'True',
        'reward_terms': [{'type': 'positive_velocity', 'weight': 1.0,
                         'lhs': 'forward_velocity'}]} for i in range(count)]}
    validate(spec, task)
    target = fixture_root / task / 'g0/initial.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert json.loads(target.read_text()) == spec
    else:
        target.write_text(json.dumps(spec, indent=2))
original = agent.load_profile
def profile(name):
    result = original(name)
    result['specs'] = str(fixture_root)
    return result
agent.load_profile = profile
sys.argv = ['agent.py', 'run', '--profile', test_profile, '--id', row_id,
            '--smoke', '--execute']
agent.main()
