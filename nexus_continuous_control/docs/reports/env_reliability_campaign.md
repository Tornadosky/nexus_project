# Environment-reliability campaign (post-phase-2)

Date: 2026-06-10/11. Hardware/setup: WSL2 Ubuntu, RTX 4060 Ti 16 GB, JAX GPU,
`.venv-wsl312`. Raw runs: `runs/loop_fix/` (batches A-D + probes); working log:
`runs/loop_fix/LOOP_NOTES.md`.

## Goal

Phase-2 left 5 failed research gates and several "weak env" verdicts. This
campaign attacks each one experimentally — budget, exploration schedule, meta
commitment, symbolic-rule fixes, reward shaping — adopting what is validated
on multiple seeds and documenting honest negative results where no lever
works. A core sub-goal was making the *metrics* truthful (several were
flattering non-behaviors).

## Headline outcomes

| Env | Phase-2 status | Campaign outcome |
| --- | --- | --- |
| HopperHop | dropped from matrix ("intrinsic bimodal instability") | **FIXED: 3/3 seeds hop** — instability was an exploration-schedule artifact; recipe adopted into config |
| PandaPickCube | nesy lift 0.177 (single budget), neural 0.000 | **FIXED for reliability: lift on every nesy seed** with sustained meta-eps (adopted); neural confirmed 0 at every setting — clean mask-dependence evidence |
| CartpoleBalance | nesy succ 0.131, neural 0.083 (30 updates) | **FIXED: undertrained.** 150-update budget adopted: nesy 0.45, neural 0.38 (3 seeds) |
| WalkerWalk | "stands, weak locomotion" | **stands robustly everywhere; net locomotion NOT achieved** — closed as documented limitation after a 5-lever, 11-run dossier (see final-verdict section) |
| Go1Joystick | succ 0.229 (below 0.35 gate) | budget-dead: 2.2x budget raises no-fall 0.91→0.97 but tracking 0.229→0.189; documented stress-test limitation (no live lever found) |
| CheetahRun | passes | untouched (already strong) |

## Metric truthfulness fixes (training-neutral)

1. **Walker net locomotion.** The per-step `walk_success_rate`
   (stand & v>0.5) counts the forward half of every sway cycle: it reads
   0.15-0.36 on policies with exactly zero net displacement. Added
   `walker/net_walk_success_rate` (per-episode mean forward velocity > 0.5 &
   standing) in the deterministic eval and made it walker's
   `primary_success_rate`. The instantaneous metric is retained for
   comparability with phase-2 numbers.
2. **Env return is not a behavior gate.**
   - Walker: the env's `move` tolerance term credits the forward half of a
     sway cycle and gives ~0 (not negative) for the backward half, so fast
     symmetric sway pays ~0.6/step with zero locomotion (observed return 931
     at net velocity +0.001). Only net velocity is honest.
   - Panda: holding the cube on the table pays a steady income that *exceeds*
     the return of actually lifting on these horizons (trapped seed: return
     445-457 at lift 0.00-0.02; lifting seed: return 337-373 at lift
     0.11-0.34). Gate panda on `lift_success_rate` / height delta, never on
     return.
3. **Hopper/walker wrap-safe success metrics** (cos(pitch) instead of
   unwrapped |pitch|; env-aligned 0.6 stand height) — carried over from the
   environment-debug round, now exercised at scale.

## What was adopted (configs)

- `configs/hopper_hop_nesy.yaml`: 400 updates @ 2048 envs, NOISE_FINISH=0.05,
  NOISE_DECAY=1.0, META_EPS_FINISH=0.05. Validation: 3/3 seeds hop (returns
  {141, 310, 130}, hop-success {0.19, 0.39, 0.17}); stock schedule 1/3
  ({9.6, 8.6, 277}).
- `configs/panda_pick_cube_nesy.yaml`: 400 updates, META_EPS_FINISH=0.05,
  META_EPS_DECAY=1.0. Validation: lift {0.109, 0.172} incl. the
  previously-trapped seed 0; stock 400up bimodal {0.023, 0.344}.
- `configs/cartpole_balance_{nesy,neural,symbolic}.yaml`: TOTAL_TIMESTEPS
  2M → 9.83M (150 updates). Validation (3 seeds each): nesy success
  0.131→0.45, neural 0.083→0.38, symbolic 0.052→0.107.

## What was kept (policy code)

- Cartpole `recover_balance` damping term (−0.1|ang_vel|) and symbolic
  rest-branch → `damp_motion`: symbolic return 142→~230, success
  0.052→~0.107 (3 seeds).
- Hopper + walker wrap-safe, env-aligned symbolic rules: hopper symbolic
  hop-success 0.000→0.028 (small but real), walker symbolic equivalent
  (stand-heavy either way). Kept on correctness grounds.
- Walker walk-skill posture shaping (see walker section).

## What was tried and rejected (A/B-tested, reverted or closed)

- **Cartpole symbolic threshold retune** (urgent band 0.20→0.35 rad): mean
  success 0.070 vs 0.107 dampfix-only — reverted. Notable asymmetry: the
  learned nesy meta genuinely prefers damping in "urgent" states, but the
  fixed rule needs the early recover handoff.
- **Walker budget scaling**: 4/4 runs (nesy/neural × 2 seeds, 10x budget) →
  high return, zero net locomotion.
- **Walker META_DECISION_INTERVAL=10** (skill commitment): nesy return halved,
  neural unchanged; locomotion zero on both.
- **Walker sustained exploration alone**: net-walk 0 (one clean seed; later
  "explore" runs carry the shaped reward due to a mid-batch code change and
  are labeled accordingly in LOOP_NOTES).
- **Panda neural anti-trap** (K10 + META_EPS 0.15 no-decay): lift 0.000 with
  the highest table-farming return observed (457). Without the mask, lift is
  never a distinct creditable choice — exploration cannot find it.
- **Go1 budget** (2.2x): stability up, tracking down. No adopted change;
  documented limitation.
- (Earlier rounds, environment-debug report: hopper wrap-safe *reward*
  variant and hop-skill rewrite — both regressed and were reverted; only the
  metric fix was kept.)

## The symbolic-baseline story (consistent across envs)

Fixed symbolic rules degenerate to the recover/stand skill on
adaptive-stabilization tasks (cartpole ~93-95% recover even after the damping
fix; hopper/walker stand-heavy with near-zero task success). Rule fixes move
success off exact zero but cannot bootstrap discovery (chicken-and-egg). The
claim rests on the *learned* variants; symbolic is a weak fixed-rule baseline,
strong only where the task regime is reliably entered (cheetah).

## Walker final verdict: net locomotion NOT achieved (documented limitation)

Walker stands and stabilizes robustly under every configuration tested
(stand-success 0.69-0.97), but **no lever produced net locomotion**. Final
dossier — five levers, eleven 400-update GPU runs, every one with
|net velocity| < 0.015 m/s:

| Lever | Runs | Net-walk |
| --- | --- | --- |
| 10x budget (stock) | nesy s0/s1, neural s0/s1 | 0, 0, 0, 0 |
| META_DECISION_INTERVAL=10 | nesy s0, neural s0 | 0, 0 (nesy return halved) |
| Sustained exploration alone | nesy s0 | 0 |
| Posture-shaped walk reward (+explore) | neural s0 x2, nesy s0/s1 x2 (incl. replicas) | 0 on all 5 |

Mechanics (established, not speculated): the env's `move` tolerance term
credits the forward half of each sway cycle and charges ~nothing for the
backward half, so fast symmetric sway pays ~0.6/step (observed up to return
931 at net velocity +0.001) and is discoverable in the first updates, while a
true gait pays ~1.0/step but requires a coordinated multi-joint cycle that
none of the exploration/commitment levers reached. The walk-skill posture
shaping is **kept** (it is more correct, never regressed return or stability,
and its code comment records the negative A/B); the sway equilibrium itself
is the documented limitation at this architecture/skill-decomposition scale.

For the paper: report walker as stand/stabilize success (NEXUS variants beat
flat on return and stand-rate) with net locomotion as a limitation, using
`net_walk_success_rate` — the instantaneous `walk_success_rate` (0.15-0.37 on
swaying policies) must not be quoted alone.

## Reproduction

```bash
# WSL Ubuntu only (native Windows MuJoCo segfaults).
source .venv-wsl312/bin/activate
export PYTHONPATH=$PWD
bash tools/run_loop_fix_batchA.sh   # budget scaling (stock code)
bash tools/run_loop_fix_batchB.sh   # K10 / symbolic rule-fix / retune / antitrap
bash tools/run_loop_fix_batchC.sh   # walker explore recipe; panda sustained-eps
bash tools/run_loop_fix_batchD.sh   # walker shaped reward
```

Each script is idempotent (skips runs whose `.pkl` exists) and self-gates on
the previous batch's driver, so they can be launched together.
