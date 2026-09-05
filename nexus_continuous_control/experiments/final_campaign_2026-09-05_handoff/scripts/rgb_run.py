"""Run the supplied controlled RGB campaign, plus a constant-input CNN control.

All three arms KEEP the vision environment. The constant arm zeroes the pixel
accessor before either trainer or evaluator is imported, so the same CNN receives
constant input during training AND testing. Report 250-step windows, not complete
episode returns: the supplied RGB harness does not stop at the first done.
"""
from __future__ import annotations
import argparse,json,pickle,time
from pathlib import Path
import yaml
from common import add_repo,write_json,write_pickle,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--config',required=True)
    p.add_argument('--out',required=True); p.add_argument('--load-policy'); p.add_argument('--reuse-proof'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--episodes',type=int,default=64); a=p.parse_args()
    add_repo(a.repo); cfg=yaml.safe_load(Path(a.config).read_text())
    constant=bool(cfg.get('CAMPAIGN_CONSTANT_PIXELS',False)); state=not cfg.get('RGB_ACTOR',True)
    if constant and state: raise ValueError('Constant-pixel control requires the same CNN as the informative-image arm')
    if not cfg.get('USE_RGB'): raise ValueError('All arms must keep the vision environment')
    if cfg.get('RGB_SHARED_ENCODER') or cfg.get('RGB_META_SEES_PIXELS'): raise ValueError('Outside frozen experiment scope')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    import jax,jax.numpy as jnp
    from nexus_continuous.envs import playground_adapter as adapter
    if constant:
        original=adapter.get_actor_pixels
        def zero_pixels(obs):
            value=original(obs)
            return None if value is None else jnp.zeros_like(value)
        adapter.get_actor_pixels=zero_pixels
    from nexus_continuous.scripts import rgb_pixel_ablation as rgb
    write_json(out/'config.json',cfg)
    write_json(out/'source.json',dict(rgb_harness=str(Path(rgb.__file__).resolve()),
        sha256=digest(Path(rgb.__file__)),constant_pixels=constant,
        devices=[str(x) for x in jax.devices()],scope='250-step windows, may include resets'))
    updates=2 if a.smoke else 250; envs=8 if a.smoke else 128
    # Keep the original training minibatch divisibility in a smoke run (8*64 /64=8).
    args=['--config',a.config,'--meta','nesy','--seed',str(cfg['SEED']),
          '--updates',str(updates),'--num-envs',str(envs),
          '--episodes',str(2 if a.smoke else a.episodes),'--eval-steps',str(20 if a.smoke else 250),
          '--out',str(out/'evaluation')]
    if state: args+=['--no-rgb']
    if a.load_policy:
        old=pickle.loads(Path(a.load_policy).read_bytes())
        if bool(old.get('constant_pixels',False))!=constant: raise ValueError('Constant-pixel metadata mismatch')
        proof=old
        if 'actual_steps' not in old or 'config' not in old:
            if not a.reuse_proof:
                raise ValueError('Legacy checkpoint lacks budget/config: pass a reuse proof built from its preserved run records')
            proof=json.loads(Path(a.reuse_proof).read_text())
            if proof.get('checkpoint_sha256')!=digest(Path(a.load_policy)):
                raise ValueError('Reuse proof does not match checkpoint bytes')
            if not proof.get('provenance_description'): raise ValueError('Reuse proof needs provenance_description')
        if int(proof['actual_steps'])!=updates*envs*64:
            raise ValueError('Old checkpoint budget does not match the declared experiment')
        pc=proof['config']
        ignored={'CAMPAIGN_ID','CAMPAIGN_CONSTANT_PIXELS','TASK_POLICY','SAVE_PATH','PRINT_EVERY',
                 'EVAL_NUM_ENVS','EVAL_NUM_EPISODES','EVAL_MAX_STEPS','EVAL_SEED'}
        mismatches={k:(pc.get(k),v) for k,v in cfg.items() if k not in ignored and pc.get(k)!=v}
        if mismatches: raise ValueError(f'Reuse configuration mismatch: {mismatches}')
        write_json(out/'reuse_proof.json',proof)
        args+=['--load-policy',str(Path(a.load_policy).resolve())]
    else: args+=['--save-policy',str(out/'policy.raw.pkl')]
    t0=time.monotonic(); rgb.main(args)
    src=Path(a.load_policy) if a.load_policy else out/'policy.raw.pkl'
    blob=pickle.loads(src.read_bytes()); blob.update(config=cfg,constant_pixels=constant,
        actual_steps=updates*envs*64,checkpoint_kind='rgb_policy',source_policy_sha256=digest(src))
    write_pickle(out/'final.pkl',blob)
    write_json(out/'COMPLETE.json',dict(smoke=a.smoke,actual_steps=updates*envs*64,
        final_sha256=digest(out/'final.pkl'),wall_seconds=time.monotonic()-t0,loaded=a.load_policy))
if __name__=='__main__': main()
