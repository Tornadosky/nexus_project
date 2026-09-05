"""Train LLM-generated skills head-to-head against the hand-written policy.

Loads a generated skill set (docs/reports/llm_generated_skills/<Env>.json),
compiles it with nexus_continuous.llm.interpreter, and trains it with the SAME
trainer / env / config / seed and the SAME (hand-written) task_metrics as the
hand-written policy, so the primary-success numbers are directly comparable.

Usage:
    python tools/llm_skill_compare.py --env CartpoleBalance --meta nesy --updates 180
    python tools/llm_skill_compare.py --env PandaPickCube --mask-mode progressive
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import nexus_continuous.algorithms.hierarchical_ac_pqn_playground as algo  # noqa: E402
from nexus_continuous.llm.interpreter import make_policy_module, load_skillset_json  # noqa: E402
from nexus_continuous.policies.registry import load_policy_module  # noqa: E402
from nexus_continuous.utils import load_config  # noqa: E402

ENV_CFG = {
    "CartpoleBalance": ("cartpole_balance_nesy.yaml", "cartpole_balance",
                        ("cart_position", "pole_angle", "cart_velocity", "pole_angular_velocity")),
    "CheetahRun": ("cheetah_run_nesy.yaml", "cheetah_run",
                   ("forward_velocity", "torso_pitch", "joint_speed")),
    "WalkerWalk": ("walker_walk_nesy.yaml", "walker_walk",
                   ("torso_height", "torso_pitch", "forward_velocity", "joint_speed")),
    "HopperHop": ("hopper_hop_nesy.yaml", "hopper_hop",
                  ("torso_height", "torso_pitch", "forward_velocity", "joint_speed")),
    "PandaPickCube": ("panda_pick_cube_nesy.yaml", "panda_pick_cube",
                      ("cube_height", "dist_tcp_cube", "dist_cube_target", "gripper_open")),
    "Go1JoystickFlatTerrain": ("go1_joystick_nesy.yaml", "go1_joystick",
                               ("base_height", "roll", "pitch", "lin_vel_x", "lin_vel_y",
                                "yaw_rate", "command_x", "command_y", "command_yaw")),
}


def _panda_field_fn():
    from nexus_continuous.policies import panda_pick_cube as p

    def fn(obs, info=None):
        tcp, cube, target, gripper, ch, dtc, dct, grasp = p._features(obs, info)
        return {"cube_height": ch, "dist_tcp_cube": dtc, "dist_cube_target": dct, "gripper_open": gripper}
    return fn


def _train(policy_name, mod, env_name, cfg_file, meta, updates, seed):
    orig = algo.load_policy_module
    algo.load_policy_module = lambda n: mod if n == policy_name else orig(n)
    try:
        nenvs, nsteps = 1024, 64
        cfg = load_config(f"configs/{cfg_file}", [
            f"META_POLICY_TYPE={meta}", f"POLICY={policy_name}", f"TASK_POLICY={policy_name}",
            f"SEED={seed}", f"TOTAL_TIMESTEPS={nenvs * nsteps * updates}", f"NUM_ENVS={nenvs}",
            f"NUM_STEPS={nsteps}", "NUM_MINIBATCHES=32", "NUM_EPOCHS=4",
            "EVAL_AFTER_TRAIN=True", "EVAL_NUM_ENVS=128", "EVAL_NUM_EPISODES=128"])
        t0 = time.time()
        out = algo.run_training(cfg)
        dt = time.time() - t0
    finally:
        algo.load_policy_module = orig
    m = jax.device_get(out.metrics)
    ev = jax.device_get(out.eval_metrics)
    usage = {k.split("/", 1)[1]: round(float(np.asarray(m[k])[-5:].mean()), 3)
             for k in sorted(m) if k.startswith("skill_usage/")}
    succ = float(np.asarray(ev.get("primary_success_rate", 0.0)))
    ret = float(np.asarray(ev.get("episode_return_mean", 0.0)))
    print(f"  [{policy_name} {meta}] eval_return={ret:.2f} primary_success={succ:.3f} "
          f"wall={dt:.0f}s usage={usage}")
    return ret, succ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=list(ENV_CFG), required=True)
    ap.add_argument("--meta", default="nesy", choices=["neural", "symbolic", "nesy"])
    ap.add_argument("--updates", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-mode", default="strict", choices=["strict", "progressive"])
    ap.add_argument("--skills-dir", default="docs/reports/llm_generated_skills")
    ap.add_argument("--suffix", default="", help="skill-set filename suffix, e.g. _refined")
    args = ap.parse_args()

    cfg_file, hand_name, fields = ENV_CFG[args.env]
    hand = load_policy_module(hand_name)
    ss = load_skillset_json(os.path.join(args.skills_dir, f"{args.env}{args.suffix}.json"))
    field_fn = _panda_field_fn() if args.env == "PandaPickCube" else None
    llm_name = f"llm_{hand_name}"
    llm_mod = make_policy_module(ss, fields, task_metrics_fn=hand.task_metrics,
                                 name=llm_name, field_fn=field_fn, mask_mode=args.mask_mode)

    print(f"### {args.env} meta={args.meta} seed={args.seed} updates={args.updates} "
          f"mask_mode={args.mask_mode} backend={jax.default_backend()}")
    print(f"  LLM skills: {llm_mod.SKILL_NAMES}")
    _train(llm_name, llm_mod, args.env, cfg_file, args.meta, args.updates, args.seed)
    _train(hand_name, hand, args.env, cfg_file, args.meta, args.updates, args.seed)


if __name__ == "__main__":
    main()
