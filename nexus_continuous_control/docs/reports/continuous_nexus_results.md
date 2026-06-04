# Continuous-Control NEXUS Results

Commit: `4fb0f94032c1157103efab24376ca58fc7c7691a`

## Environment Info

- python/platform: Windows-10-10.0.26200-SP0
- jax: 0.4.35
- mujoco: 3.6.0
- mujoco_playground: unknown

## Method Summary

Continuous-control NEXUS keeps the NEXUS meta-policy over interpretable skills, but replaces discrete option values with deterministic skill actors and skill critics. Skill critics train from shared rollouts on skill-specific rewards, while learned meta policies train on environment reward with masked max-Q bootstraps for NeSy.

## Exact Configs

- `configs/cartpole_balance_nesy.yaml`
- `configs/cartpole_balance_neural.yaml`
- `configs/cartpole_balance_symbolic.yaml`
- `configs/cheetah_run_nesy.yaml`
- `configs/cheetah_run_neural.yaml`
- `configs/flat_cartpole_balance.yaml`
- `configs/flat_cheetah_run.yaml`
- `configs/flat_go1_joystick.yaml`
- `configs/flat_hopper_hop.yaml`
- `configs/flat_panda_pick_cube.yaml`
- `configs/flat_walker_walk.yaml`
- `configs/go1_joystick_nesy.yaml`
- `configs/go1_joystick_neural.yaml`
- `configs/hopper_hop_nesy.yaml`
- `configs/hopper_hop_neural.yaml`
- `configs/panda_pick_cube_nesy.yaml`
- `configs/panda_pick_cube_neural.yaml`
- `configs/panda_pick_cube_symbolic.yaml`
- `configs/walker_walk_nesy.yaml`
- `configs/walker_walk_neural.yaml`

## Environment and Skill Table

| Environment | Skills | Rule summary |
| --- | --- | --- |
| CartpoleBalance | recover_balance, center_cart, damp_motion | angle/velocity recovery, cart centering, velocity damping |
| CheetahRun | accelerate_forward, stabilize_posture, energy_efficient_run | speed, posture, and control-cost rewards |
| WalkerWalk | stand_recover, walk_forward, stabilize_gait, energy_efficient | height/uprightness, target speed, gait stability, torque efficiency |
| HopperHop | stand_recover, hop_forward, stabilize_landing, energy_efficient | survival/uprightness plus env-reward tracking skills |
| PandaPickCube | reach_cube, grasp_cube, lift_cube, place_or_stabilize | distance-to-cube, grasp, height, target placement phases |
| Go1JoystickFlatTerrain | stand, track_velocity, turn, recover | stance, command tracking, yaw tracking, fall recovery |

## Main Performance

_No rows._

## Baseline Comparison

_No rows._

## Learning Trends

_No rows._

## Skill and Mask Diagnostics

_No rows._

_No rows._

## Raw Feature Diagnostics

_No rows._

## Plots

- `plots/paper/main_return_curves.png`
- `plots/paper/final_performance_vs_flat.png`
- `plots/paper/skill_reward_curves_by_env.png`
- `plots/paper/skill_usage_by_env_variant.png`
- `plots/paper/mask_availability_vs_selection.png`
- `plots/paper/panda_phase_diagnostics.png`
- `plots/paper/loss_and_td_diagnostics.png`
- `plots/paper/raw_feature_diagnostics.png`

## Limitations and Failure Cases

# NEXUS result diagnostics

Generated: 2026-06-04T09:59:37
Runs loaded: 0
Pickle load failures: 0
Metric extraction errors: 0

## Finite-value checks

- FATAL: no metrics were extracted.

## Skill usage checks


## Learning-signal checks


## Required run coverage checklist

- [ ] cartpole_balance_nesy: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] cartpole_balance_neural: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] cartpole_balance_symbolic: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] cheetah_run_nesy: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] cheetah_run_neural: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] flat_cartpole_balance: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] flat_cheetah_run: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] flat_hopper_hop: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] flat_panda_pick_cube: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] flat_walker_walk: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] hopper_hop_nesy: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] hopper_hop_neural: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] panda_pick_cube_nesy: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] panda_pick_cube_neural: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] panda_pick_cube_symbolic: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] walker_walk_nesy: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] walker_walk_neural: 0 seed rows (MISSING_OR_INCOMPLETE)
- [ ] go1_joystick_nesy extension: 0 seed rows

## Checklist failure flags


