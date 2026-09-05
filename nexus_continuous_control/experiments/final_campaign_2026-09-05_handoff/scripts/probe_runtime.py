"""Import/API probe. Does not train, reset a simulator, or allocate a Slurm job.

--device is an optional device-initialization test; it is never implied on a
Viper login node. RGB still requires an actual rendering smoke test afterward.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib, importlib.metadata as md
import inspect, json, os, sys
from pathlib import Path
from common import add_repo, write_json

PACKAGES = ('jax','jaxlib','flax','optax','brax','mujoco','mujoco-mjx','playground',
            'numpy','PyYAML','warp-lang','torch','transformers','huggingface-hub','accelerate')

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--repo', required=True)
    p.add_argument('--kind', choices=('state','ppo','rgb'), required=True)
    p.add_argument('--out', required=True); p.add_argument('--device', action='store_true')
    a = p.parse_args(); source = add_repo(a.repo)
    report = {'python': sys.version, 'executable': sys.executable, 'kind': a.kind,
              'source': str(source), 'packages': {}, 'errors': [],
              'training_executed': False, 'rendering_executed': False}
    for name in PACKAGES:
        try: report['packages'][name] = md.version(name)
        except md.PackageNotFoundError: report['packages'][name] = None
    try:
        import jax
        from nexus_continuous.algorithms import hierarchical_ac_pqn_playground as trainer
        report['trainer'] = str(Path(trainer.__file__).resolve())
        if source not in Path(trainer.__file__).resolve().parents:
            raise RuntimeError('Wrong trainer imported: editable install shadowed frozen source')
        if a.kind == 'state':
            text = inspect.getsource(trainer.make_train)
            anchor = '        runner_state, metrics = jax.lax.scan(\n            _update_step, runner_state, None, config["NUM_UPDATES"]\n        )'
            if text.count(anchor) != 1: raise RuntimeError('Snapshot injection anchor changed')
            if 'SHARED_SKILL_REWARD' not in text: raise RuntimeError('Core source lost the HPQN control')
            import robustness_eval as re
            ev = inspect.getsource(re.evaluate)
            for anchor in ('    return summary', 'jnp.zeros((max(num_skills, 1),), jnp.float32)'):
                if ev.count(anchor) != 1: raise RuntimeError('Common evaluator source anchor changed')
        if a.kind == 'ppo':
            from train_ppo_baseline import ppo_config_for, _shim_device_put_replicated
            from brax.training.agents.ppo import train as ppo
            sig = inspect.signature(ppo.train)
            report['ppo_signature'] = str(sig)
            for key in ('policy_params_fn','num_evals','num_resets_per_eval','wrap_env_fn',
                        'num_eval_envs','clipping_epsilon','gae_lambda','normalize_advantage'):
                if key not in sig.parameters: raise RuntimeError(f'PPO API lacks {key}')
            report['ppo_supported_tasks'] = {n: bool(ppo_config_for(n)) for n in ('HopperHop','Go1JoystickFlatTerrain')}
        if a.kind == 'rgb':
            from nexus_continuous.scripts import rgb_pixel_ablation as rgb
            import mujoco.mjx as mjx
            import warp
            report['rgb_harness'] = str(Path(rgb.__file__).resolve())
            report['mjx_render_api'] = {n: hasattr(mjx,n) for n in ('render','create_render_context','refit_bvh')}
            if not all(report['mjx_render_api'].values()):
                raise RuntimeError('Installed MuJoCo-MJX lacks the frozen RGB renderer API; use a separate verified RGB environment')
            report['requires_render_smoke'] = True
        from mujoco_playground import registry
        import mujoco_playground
        report['playground_module'] = str(Path(mujoco_playground.__file__).resolve())
        try: report['playground_direct_url'] = json.loads(md.distribution('playground').read_text('direct_url.json') or 'null')
        except md.PackageNotFoundError: pass
        if a.device:
            report['devices'] = [str(d) for d in jax.devices()]
            report['backend'] = jax.default_backend()
            if report['backend'] != 'gpu' or len(jax.local_devices()) != 1:
                raise RuntimeError('One exposed accelerator required')
    except Exception as e: report['errors'].append(f'{type(e).__name__}: {e}')
    report['import_api_pass'] = not report['errors']
    # No permissive NaN JSON; every probe has a unique destination.
    write_json(Path(a.out), report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['import_api_pass'] else 2)

if __name__ == '__main__': main()
