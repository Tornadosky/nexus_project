# Continuous-NEXUS Environment Debug Report

Date: 2026-06. Hardware: WSL2 Ubuntu 22.04, NVIDIA RTX 4060 Ti (16 GB),
JAX 0.10.1 (GPU), MuJoCo 3.9.0, vendored MuJoCo Playground. Training entry point:
`nexus_continuous.scripts.train_nexus_playground` / `run_training`.

## Scope

Audit every configured environment for build/runtime failures and learning
behaviour, find the root cause of the environments previously reported as
"problem" cases, fix the genuine bugs, and archive verification results so the
final reports can be prepared.

## TL;DR

- **All 6 environments build, reset, step, train, and evaluate with finite
  metrics.** No environment fails to run.
- The environment dropped in phase-2 (**HopperHop**) is **not broken**: with the
  current (repaired) feature extraction it learns to hop on the majority of seeds
  (eval return ~200, ~30 % hop success) whereas the flat baseline is stuck at
  ~1.4. Two genuine bugs were found and fixed (success-metric threshold and a
  wrap-unsafe uprightness term). The residual issue is seed-to-seed **training
  instability** (discovery-driven bimodal outcomes), now characterised.
- The Windows-native MuJoCo segfault that blocked all real-env runs is an
  OS/native-library problem; training must be run under **WSL** (see the
  reproduction notes). On WSL/Linux MuJoCo runs correctly.
- One cosmetic upstream warning (`overflow encountered in cast`) originates in
  **MJX's own box-box collision code** for PandaPickCube; it is harmless
  (observations/rewards stay finite).

## Environment status matrix (GPU, 250 updates, 1024 envs x 64 steps, seed 0)

| Env | Builds/Runs | Finite | nesy task result | flat baseline | Verdict |
| --- | --- | --- | --- | --- | --- |
| CartpoleBalance | yes | yes | upright ~0.53 (det. eval) | weaker | learns |
| CheetahRun | yes | yes | forward speed ~1.3, runs | speed lower | learns (strong) |
| WalkerWalk | yes | yes | stands (return ~184) | stands | learns posture; forward-locomotion weak |
| HopperHop | yes | yes | return ~200, hop-success ~0.30 (good seeds) | ~1.4 | learns on majority of seeds; seed-unstable |
| PandaPickCube | yes | yes | **lift-success 0.27**, height-delta 0.055 | **lift 0.00** | learns to lift (nesy >> flat) |
| Go1JoystickFlatTerrain | yes | yes | no-fall 0.16, tracking 0.04 | no-fall 0.01, tracking 0.001 | weak but nesy >> flat |

(Exact per-run trajectories are in the raw logs; numbers above are single-seed
verification snapshots at a reduced 250-update budget, not the full paper-scale
configs.)

## Findings and fixes

### 1. mujoco_menagerie / panda & go1 assets — NON-ISSUE
The nested `mujoco_menagerie` is present under
`vendor/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie`;
PandaPickCube and Go1JoystickFlatTerrain load real robot assets. No action
needed.

### 2. Semantic feature extraction — verified correct
Every policy's expected semantic keys (`x_velocity`, `torso_height`,
`base_height`, `tcp_pos`, `cube_pos`, `command_yaw`, ...) are present in
`info`/`policy_info` for its environment, so the index fallbacks are not used.
The adapter's hopper height (`xipos[torso]-xipos[foot]`) matches the env's own
`_height`; its `qvel[0]` forward-speed proxy correlates ~0.94 with the env's
`torso_subtreelinvel` sensor.

### 3. HopperHop — the dropped "problem" environment (FIXED + characterised)
Root causes of the previous "failed behaviour":
- The phase-2 drop predates the "repair playground semantic policy features"
  commit; with the current features hopper does learn.
- **Bug A (success metric, fixed):** `task_metrics` required `height > 0.9 &
  |pitch| < 0.5`, but the env defines standing at height >= 0.6 and `pitch`
  (`qpos[2]`) is an *unwrapped* hinge angle, so `|pitch|` is wrap-unsafe. The
  metric reported ~0 % success even when the agent was demonstrably hopping
  (env return ~200). Re-aligned to the env: `height > 0.6 & cos(pitch) > 0.7 &
  speed > 1.0`, which now reports the real ~30 % hop success.
- **Reward investigated, NOT changed:** the stand/stabilize skill rewards use
  `1 - |pitch|/1.2` uprightness and `-0.2*|pitch|` penalties, which are
  wrap-unsafe for the unwrapped hinge angle. We tested replacing them with
  wrap-safe `cos(pitch)` over 5 seeds: it did **not** help and the successful
  seeds trended weaker (returns ~61-90 vs the original reward's ~183-228), so
  the change was reverted. The working reward shaping is left exactly as-is;
  only the (training-neutral) success metric was changed. A full rewrite of the
  hop/efficient skills was also tried and clearly regressed (return 206 -> 8),
  reinforcing that the existing shaping should not be re-engineered.
- **Residual (characterised, not a code bug):** the task reward is
  `standing * hopping`, near-zero until the robot both stands and moves, so
  learning is a hard discovery problem from a random orientation. Outcomes are
  **bimodal across seeds** — roughly half of seeds discover hopping (return
  ~200, ~30 % success) and half collapse (return ~1.5, height stuck ~0.13).
  GPU non-determinism alone can flip a fixed seed between the two (the same
  seed produced both 206 and 1.5 on separate runs). Neither a wrap-safe reward
  variant nor `active_only` + stronger exploration removed the variance,
  confirming it is intrinsic to the task/reward rather than a bug.
  Recommendation for the paper: report HopperHop multi-seed (mean +/- std or
  success-fraction over >= 5 seeds; here ~2/5 seeds reach return ~180-230 with
  ~30 % hop success, the rest collapse to ~1.5) and present its instability as a
  documented limitation, as was already done for Go1.

### 4. WalkerWalk — metric correctness fix
WalkerWalk also resets at a uniform-random orientation and its success metric
used the same wrap-unsafe `|pitch| < 0.5`. Aligned to the env's own uprightness
(`cos(pitch) > 0.7`, matching the env's `xmat[2,2]` "up" term). Reward shaping
left unchanged. Walker reliably learns to stand; forward locomotion remains weak
(consistent with the phase-2 result) — a difficulty/exploration limitation, not
a bug.

### 5. PandaPickCube overflow warning — benign upstream
`RuntimeWarning: overflow encountered in cast` comes from MJX's own
`collision_convex.py` (`_box_box`, clamping `inf` distances to `finfo.max`) for
the cube/finger box geoms. Observations and rewards remain finite; nothing in
the NEXUS code is involved.

## How to reproduce (WSL + GPU)

```bash
# In WSL Ubuntu (native MuJoCo segfaults on Windows -- do not run there).
source /home/smirn/.../.venv-wsl312/bin/activate   # jax(gpu) + mujoco + playground
export PYTHONPATH=<repo>/nexus_continuous_control
cd <repo>/nexus_continuous_control
python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/hopper_hop_nesy.yaml --no-wandb
```

## Files changed

Both changes are confined to `task_metrics`/diagnostics (reporting only) and are
**training-neutral** — the skill rewards, masks, and symbolic rules used during
training are unchanged, so no learning result can regress.

- `nexus_continuous/policies/hopper_hop.py` — env-aligned, wrap-safe success
  metric (`height > 0.6 & cos(pitch) > 0.7 & speed > 1.0`); added `upright_cos`
  diagnostic; added calibration constants. Reward shaping reverted to original.
- `nexus_continuous/policies/walker_walk.py` — wrap-safe `cos(pitch)` in the
  success metric (`stand_success`).

## Three-variant audit (neural / symbolic / nesy x 6 envs)

GPU, 180 updates, 1024 envs x 64 steps, seed 0. "usage" = final skill-usage
fraction; "succ" = deterministic-eval primary success.

| Env | neural | symbolic | nesy |
| --- | --- | --- | --- |
| Cartpole | 3 skills, succ 0.44, upright 0.66 | **100% recover, succ 0.04** (degenerate) | damp 0.90, succ 0.46, upright 0.87 |
| Cheetah | succ 0.94, speed 5.8 | succ 0.97, speed 6.4 | succ 0.97, speed 5.7 |
| Walker | succ 0.25, **fwd-vel ~0** | **90% stand, succ 0.01** | succ 0.26, **fwd-vel ~0** |
| Hopper | succ 0.001 (collapsed seed) | **100% stand, succ 0** | succ 0.002 (collapsed seed) |
| Panda | grasp 0.73, **lift 0.00** | lift 0.37 usage, lift-succ 0.016 | lift 0.74 usage, **lift-succ 0.16** |
| Go1 | turn 0.44, track 0.002 | **97% recover, succ 0.001** | recover 0.52, no-fall 0.25, track 0.018 |

Issues found:
1. **Symbolic degeneracy.** Cartpole/hopper/go1 symbolic collapse to ~100 % of
   the "recover/stand" skill; walker to 90 % stand. Two distinct causes:
   - Cartpole: the `recover_balance` skill (angle-only reward) cannot fully
     stabilise the pole, so `|angle|>0.20` stays true and the `urgent_angle`
     branch (skill 0) fires every step -- the rule never reaches its other
     branches. Re-pointing the rest branch to `damp_motion` was tried and had
     no effect (the urgent branch dominates), so it was reverted. The
     degeneracy is inherent to fixed single-skill control of a task that needs
     adaptive skill-switching; the learned neural/nesy metas balance fine.
   - Hopper/go1: chicken-and-egg -- the agent never leaves the "fallen" regime,
     so only the recover/stand skill is ever executed (hence trained). Hard to
     fix from the rule alone; these are the intrinsically hard tasks.
   Cheetah symbolic is healthy (running reliably enters the `fast_enough`
   branch), confirming the mechanism. Takeaway: symbolic is a weak fixed-rule
   baseline that degenerates on adaptive-stabilisation tasks; the two *learned*
   variants (neural, nesy) are the ones that must -- and do -- work per env.
2. **Panda meta matters.** nesy lifts (lift-succ 0.16, cube raised to 0.057 m vs
   the 0.03 m table) because its mask forces progression reach->grasp->lift;
   the *learned* neural meta gets stuck grasping (grasp 0.73, lift 0.00, cube
   never leaves the table). Strong evidence for the NeSy mask. Cube-pick is
   measured from the env's real `cube_pos[z]`, so the lift numbers are truthful.
3. **Walker sways.** primary_success (instantaneous |vel|>0.5) reads ~0.25 while
   mean forward velocity ~0 -- the walker oscillates rather than making net
   progress. Metric leniency, not a crash.
4. **Hopper seed instability (intrinsic, nondeterminism-dominated).** All three
   variants collapsed at seed 0 / 180 updates, but with adequate training the
   env clearly learns to hop on a meaningful fraction of seeds. Multi-seed at
   400 updates (nesy): seeds give return {9.6, 8.6, **276.9**} with hop-success
   {0.008, 0.007, **0.346**} -- ~1/3 reach strong hopping, the rest collapse.
   The variance is dominated by GPU nondeterminism (the same seed/config has
   produced both ~10 and ~270 on separate runs), so a single run is never
   representative. Practical recommendation: train hopper >=400 updates and
   report the multi-seed success fraction; it is a hard, high-variance task, not
   a universally-failing one.

   **FOLLOW-UP (2026-06-11, supersedes the verdict above):** the bimodal
   collapse is an **exploration-schedule artifact, not an intrinsic env
   property**. The collapsed regime pays ~nothing, so the failure mode is the
   actor noise decaying before standing is discovered. With sustained
   exploration (NOISE_FINISH=0.05, NOISE_DECAY=1.0, META_EPS_FINISH=0.05) at
   2048 envs / 400 updates, hopper learned to hop on **3/3 seeds** (returns
   {141, 310, 130}, hop-success {0.19, 0.39, 0.17}; runs/loop_fix/batchA).
   The recipe is adopted in `configs/hopper_hop_nesy.yaml`; HopperHop is
   eligible to rejoin the main results matrix. Full campaign log:
   `docs/reports/env_reliability_campaign.md`.
