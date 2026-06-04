# Continuous-Control NEXUS Phase-2 Results

Final commit hash: recorded in `runs/phase2_review/final_commit_hash.txt` and the final handoff summary.
Branch: `codex/phase2-eval-finalize`

## Claim Boundary

This phase-2 run is mechanically complete but not fully paper-ready under the strict research gates. HopperHop is intentionally absent from the main matrix. The final matrix contains 51 runs: 17 required configurations times 3 seeds.

The implementation now supports deterministic evaluation, aligned task metrics, task-specific diagnostics, NeSy mask-violation logging, and paper-style collection/plotting. Results are mixed: neural passes the return-ratio gate on 5/5 environments; NeSy passes on 3/5, below the 4/5 research gate.

## Environment Info

```text
python 3.12.13 (main, May 10 2026, 19:30:01) [Clang 22.1.3 ]
platform Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
jax 0.10.1
backend gpu
devices [CudaDevice(id=0)]
mujoco_playground unknown
mujoco_playground_file /mnt/c/Users/smirn/VSCodeProjects/nexus_project/nexus_continuous_control/.venv-wsl312/lib/python3.12/site-packages/mujoco_playground/__init__.py
mujoco_playground_inspect_file /mnt/c/Users/smirn/VSCodeProjects/nexus_project/nexus_continuous_control/.venv-wsl312/lib/python3.12/site-packages/mujoco_playground/__init__.py
mujoco_playground_git_root /mnt/c/Users/smirn/VSCodeProjects/nexus_project
mujoco_playground_git_commit d4ee96a5de534d85cfc66e94c40a25e18cd31bd4
```

## Main Environment Set

1. CartpoleBalance
2. CheetahRun
3. WalkerWalk
4. PandaPickCube
5. Go1JoystickFlatTerrain

HopperHop was dropped from phase-2 success claims after Phase 1 showed failed behavior.

## Deterministic Return and Success Table

| Environment | Variant | Seeds | Deterministic return | Ratio to flat | Primary success |
| --- | --- | --- | --- | --- | --- |
| CartpoleBalance | flat | 3.000 | 196.703 | 1.000 | 0.031 |
| CartpoleBalance | neural | 3.000 | 271.831 | 1.382 | 0.083 |
| CartpoleBalance | nesy | 3.000 | 336.778 | 1.712 | 0.131 |
| CartpoleBalance | symbolic | 3.000 | 142.458 | 0.724 | 0.052 |
| CheetahRun | flat | 3.000 | 161.104 | 1.000 | 0.432 |
| CheetahRun | neural | 3.000 | 210.855 | 1.309 | 0.549 |
| CheetahRun | nesy | 3.000 | 218.768 | 1.358 | 0.673 |
| WalkerWalk | flat | 3.000 | 340.699 | 1.000 | 0.063 |
| WalkerWalk | neural | 3.000 | 569.226 | 1.671 | 0.118 |
| WalkerWalk | nesy | 3.000 | 198.456 | 0.582 | 0.028 |
| PandaPickCube | flat | 3.000 | 576.189 | 1.000 | 0.000 |
| PandaPickCube | neural | 3.000 | 467.902 | 0.812 | 0.000 |
| PandaPickCube | nesy | 3.000 | 332.731 | 0.577 | 0.177 |
| PandaPickCube | symbolic | 3.000 | 423.291 | 0.735 | 0.026 |
| Go1JoystickFlatTerrain | flat | 3.000 | 8.079 | 1.000 | 0.027 |
| Go1JoystickFlatTerrain | neural | 3.000 | 6.845 | 0.847 | 0.185 |
| Go1JoystickFlatTerrain | nesy | 3.000 | 6.502 | 0.805 | 0.229 |

## Aligned Task Metrics

| Environment | Variant | Primary success | Primary goal metric | Diagnostic |
| --- | --- | --- | --- | --- |
| CartpoleBalance | flat | 0.031 | -0.167 | upright=0.064 |
| CartpoleBalance | neural | 0.083 | 0.188 | upright=0.129 |
| CartpoleBalance | nesy | 0.131 | 0.187 | upright=0.165 |
| CartpoleBalance | symbolic | 0.052 | -0.387 | upright=0.080 |
| CheetahRun | flat | 0.432 | 1.615 | speed=1.615 |
| CheetahRun | neural | 0.549 | 2.069 | speed=2.069 |
| CheetahRun | nesy | 0.673 | 2.187 | speed=2.187 |
| WalkerWalk | flat | 0.063 | -0.027 | stand=0.233 |
| WalkerWalk | neural | 0.118 | -0.013 | stand=0.553 |
| WalkerWalk | nesy | 0.028 | -0.027 | stand=0.128 |
| PandaPickCube | flat | 0.000 | 0.001 | reach=1.000, lift=0.000 |
| PandaPickCube | neural | 0.000 | 0.001 | reach=1.000, lift=0.000 |
| PandaPickCube | nesy | 0.177 | 0.033 | reach=1.000, lift=0.177 |
| PandaPickCube | symbolic | 0.026 | 0.011 | reach=1.000, lift=0.026 |
| Go1JoystickFlatTerrain | flat | 0.027 | -1.051 | no_fall=0.071, vel_err=0.819 |
| Go1JoystickFlatTerrain | neural | 0.185 | -1.125 | no_fall=0.752, vel_err=0.854 |
| Go1JoystickFlatTerrain | nesy | 0.229 | -1.067 | no_fall=0.908, vel_err=0.844 |

## Skill Usage and Disentanglement

| Environment | Variant | Active skills | Usage entropy | Skill reward std |
| --- | --- | --- | --- | --- |
| CartpoleBalance | flat | 1.000 | 0.000 | 0.000 |
| CartpoleBalance | neural | 3.000 | 0.984 | 26.647 |
| CartpoleBalance | nesy | 3.000 | 0.992 | 10.840 |
| CartpoleBalance | symbolic | 1.000 | 0.000 | 19.428 |
| CheetahRun | flat | 1.000 | 0.000 | 0.000 |
| CheetahRun | neural | 3.000 | 0.982 | 0.807 |
| CheetahRun | nesy | 3.000 | 0.869 | 0.625 |
| WalkerWalk | flat | 1.000 | 0.000 | 0.000 |
| WalkerWalk | neural | 4.000 | 1.237 | 0.478 |
| WalkerWalk | nesy | 4.000 | 1.318 | 0.269 |
| PandaPickCube | flat | 1.000 | 0.000 | 0.000 |
| PandaPickCube | neural | 4.000 | 0.772 | 0.401 |
| PandaPickCube | nesy | 3.333 | 0.846 | 0.212 |
| PandaPickCube | symbolic | 3.000 | 0.930 | 0.284 |
| Go1JoystickFlatTerrain | flat | 1.000 | 0.000 | 0.000 |
| Go1JoystickFlatTerrain | neural | 4.000 | 1.167 | 0.956 |
| Go1JoystickFlatTerrain | nesy | 4.000 | 1.211 | 0.987 |

## Mask Violation Diagnostics

NeSy mask violation should be zero up to floating point tolerance. The final review includes explicit `mask/violation_rate` and per-skill `mask_violation/*` metrics.

| Environment | Variant | Mask violation rate |
| --- | --- | --- |
| CartpoleBalance | flat | 0.000 |
| CartpoleBalance | neural | 0.000 |
| CartpoleBalance | nesy | 0.000 |
| CartpoleBalance | symbolic | 0.000 |
| CheetahRun | flat | 0.000 |
| CheetahRun | neural | 0.000 |
| CheetahRun | nesy | 0.000 |
| WalkerWalk | flat | 0.000 |
| WalkerWalk | neural | 0.000 |
| WalkerWalk | nesy | 0.000 |
| PandaPickCube | flat | 0.000 |
| PandaPickCube | neural | 0.000 |
| PandaPickCube | nesy | 0.000 |
| PandaPickCube | symbolic | 0.000 |
| Go1JoystickFlatTerrain | flat | 0.000 |
| Go1JoystickFlatTerrain | neural | 0.000 |
| Go1JoystickFlatTerrain | nesy | 0.000 |

## Panda Phase Diagnostics

Panda success is measured by cube height, not by the grasp proxy. `grasp_proxy` remains only a rule feature. Lift success is based on max cube height relative to initial/table height.

| Variant | Reach | Closed near cube | Lift | Place | Max height delta |
| --- | --- | --- | --- | --- | --- |
| flat | 1.000 | 0.005 | 0.000 | 0.000 | 0.001 |
| neural | 1.000 | 1.000 | 0.000 | 0.000 | 0.001 |
| nesy | 1.000 | 1.000 | 0.177 | 0.000 | 0.033 |
| symbolic | 1.000 | 0.995 | 0.026 | 0.000 | 0.011 |

## Go1 Tuning Decision and Limitation

Go1 tuning selected the active-only actor update config in `configs/go1_joystick_nesy_phase2.yaml`, as documented in `docs/reports/go1_phase2_tuning_decision.md`. Go1 still misses the 0.35 primary-success research gate and should be presented as a weak robotics stress-test limitation.

| Variant | Primary success | No-fall rate | Velocity error | Yaw error | Return |
| --- | --- | --- | --- | --- | --- |
| flat | 0.027 | 0.071 | 0.819 | 0.464 | 8.079 |
| neural | 0.185 | 0.752 | 0.854 | 0.542 | 6.845 |
| nesy | 0.229 | 0.908 | 0.844 | 0.445 | 6.502 |

## Figures

Generated figures are under `runs/phase2_review/plots/phase2_paper/` in the handoff bundle.

## Strict Validation Outcome

```text
# Phase-2 NEXUS Validation

Review directory: `runs/phase2_review`


## Files
- **PASS** found `metrics_long.csv`
- **PASS** found `metrics_wide.csv`
- **PASS** found `final_summary.csv`
- **PASS** found `baseline_comparison.csv`
- **PASS** found `learning_trends.csv`
- **PASS** found `skill_disentanglement.csv`
- **PASS** found `mask_diagnostics.csv`
- **PASS** found `raw_feature_diagnostics.csv`
- **PASS** found `det_eval_summary.csv`

## Matrix
- **PASS** HopperHop absent from final main summary
- **PASS** CartpoleBalance/neural: 3 seeds
- **PASS** CartpoleBalance/nesy: 3 seeds
- **PASS** CartpoleBalance/symbolic: 3 seeds
- **PASS** CartpoleBalance/flat: 3 seeds
- **PASS** CheetahRun/neural: 3 seeds
- **PASS** CheetahRun/nesy: 3 seeds
- **PASS** CheetahRun/flat: 3 seeds
- **PASS** WalkerWalk/neural: 3 seeds
- **PASS** WalkerWalk/nesy: 3 seeds
- **PASS** WalkerWalk/flat: 3 seeds
- **PASS** PandaPickCube/neural: 3 seeds
- **PASS** PandaPickCube/nesy: 3 seeds
- **PASS** PandaPickCube/symbolic: 3 seeds
- **PASS** PandaPickCube/flat: 3 seeds
- **PASS** Go1JoystickFlatTerrain/neural: 3 seeds
- **PASS** Go1JoystickFlatTerrain/nesy: 3 seeds
- **PASS** Go1JoystickFlatTerrain/flat: 3 seeds

## Finite metrics
- **PASS** metrics_long.csv has no non-finite numeric values

## Baseline ratios
- **PASS** neural return ratio gate: 5/5 envs >= 0.80; required 4/5
- **INFO** neural ratio CartpoleBalance: 1.382
- **INFO** neural ratio CheetahRun: 1.309
- **INFO** neural ratio Go1JoystickFlatTerrain: 0.847
- **INFO** neural ratio PandaPickCube: 0.812
- **INFO** neural ratio WalkerWalk: 1.671
- **FAIL** nesy return ratio gate: 3/5 envs >= 0.70; required 4/5
- **INFO** nesy ratio CartpoleBalance: 1.712
- **INFO** nesy ratio CheetahRun: 1.358
- **INFO** nesy ratio Go1JoystickFlatTerrain: 0.805
- **INFO** nesy ratio PandaPickCube: 0.577
- **INFO** nesy ratio WalkerWalk: 0.582

## Deterministic eval
- **FAIL** CartpoleBalance: best NEXUS primary success 0.131; threshold 0.60
- **PASS** CheetahRun: best NEXUS primary success 0.673; threshold 0.40
- **FAIL** WalkerWalk: best NEXUS primary success 0.118; threshold 0.35
- **FAIL** PandaPickCube: best NEXUS primary success 0.177; threshold 0.20
- **FAIL** Go1JoystickFlatTerrain: best NEXUS primary success 0.229; threshold 0.35
- **PASS** Panda panda/reach_success_rate: best 1.000; threshold 0.50
- **PASS** Panda panda/lift_success_rate: best 0.258; threshold 0.20

## Masks and skills
- **PASS** checked skill-usage sums where available
- **PASS** mask violation columns present: ['mask_violation_value']

## Summary
- errors: 5
- warnings: 0
- strict: True

```

## Reproducibility

The handoff bundle includes `final_commit_hash.txt`, `git_status.txt`, `final_diff_stat.txt`, `source_snapshot_<commit>.zip`, `pip_freeze.txt`, `env_info_start.txt`, and final matrix `.out`/`.err` logs.

## Limitations

- The strict research validation fails 5 gates.
- Cartpole and Walker deterministic primary-success thresholds are not met.
- Panda lift success passes the best-NEXUS lift threshold but remains near the boundary; place success is low.
- Go1 remains weak and should be described as a stress-test limitation.
- The policy modules use privileged semantic state features rather than learned RGB features.
