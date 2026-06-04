# Continuous-Control NEXUS Phase-2 Results

Commit: `<fill commit hash>`  
Branch: `<fill branch>`  
Hardware: `<fill GPU/CPU>`  
MuJoCo Playground version/commit: `<must be exact, not unknown>`

## Claim boundary

This report must not claim Hopper success. HopperHop was intentionally dropped from the main matrix. The main environment set is:

1. CartpoleBalance
2. CheetahRun
3. WalkerWalk
4. PandaPickCube
5. Go1JoystickFlatTerrain

The correct paper-style claim is: continuous-control NEXUS trains stably on these environments, uses interpretable skill-level control, exposes skill/mask/task diagnostics, and is competitive with a flat AC-PQN baseline on the accepted gates. Any weak Go1 performance must be explicitly reported.

## Implementation changes in phase 2

- Deterministic evaluation rollouts with noise off and meta-epsilon off.
- Aligned task-success metrics for every environment.
- Hopper removed from the main experiment matrix.
- Panda skill/task metrics distinguish grasp proxy from real cube-height lift evidence.
- NeSy mask violation rate logged.
- Source snapshot included in final handoff.

## Main result table

Paste or generate from `det_eval_summary.csv` and `baseline_comparison.csv`.

| Environment | Variant | Deterministic return | Ratio to flat | Primary success | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CartpoleBalance | flat | | 1.00 | | |
| CartpoleBalance | neural | | | | |
| CartpoleBalance | nesy | | | | |
| CartpoleBalance | symbolic | | | | |
| CheetahRun | flat | | 1.00 | | |
| CheetahRun | neural | | | | |
| CheetahRun | nesy | | | | |
| WalkerWalk | flat | | 1.00 | | |
| WalkerWalk | neural | | | | |
| WalkerWalk | nesy | | | | |
| PandaPickCube | flat | | 1.00 | | |
| PandaPickCube | neural | | | | |
| PandaPickCube | nesy | | | | |
| PandaPickCube | symbolic | | | | |
| Go1JoystickFlatTerrain | flat | | 1.00 | | |
| Go1JoystickFlatTerrain | neural | | | | |
| Go1JoystickFlatTerrain | nesy | | | | |

## Aligned task metrics

For each environment, report the primary metric and at least two raw/semantic diagnostics.

### CartpoleBalance

- Primary: upright-and-centered success rate.
- Diagnostics: absolute wrapped pole angle, centered fraction, episode length.

### CheetahRun

- Primary: speed success rate or forward velocity.
- Diagnostics: forward velocity, posture stable rate, action norm.

### WalkerWalk

- Primary: stable walking success rate.
- Diagnostics: torso height, pitch, forward velocity.

### PandaPickCube

- Primary: lift success rate measured by cube height, not the grasp proxy.
- Diagnostics: reach success, closed-near-cube rate, max cube height, max cube height delta, place/target rate if applicable.

### Go1JoystickFlatTerrain

- Primary: no-fall plus tracking success rate.
- Diagnostics: base height, roll/pitch, velocity tracking error, yaw tracking error.

## Interpretability and skill disentanglement

Summarize:

- skill usage entropy and number of active skills;
- per-skill reward learning curves;
- NeSy mask availability vs selected skill;
- mask violation rate;
- Panda phase progression.

## Figures

The final handoff must include at least:

- `plots/phase2_paper/phase2_return_vs_flat.png`
- `plots/phase2_paper/phase2_deterministic_task_success.png`
- `plots/phase2_paper/phase2_training_return_curves_by_env.png`
- `plots/phase2_paper/phase2_skill_usage_heatmap.png`
- `plots/phase2_paper/phase2_panda_eval_phases.png`
- `plots/phase2_paper/phase2_nesy_mask_diagnostics.png`
- `plots/phase2_paper/phase2_td_stability.png`

## Limitations

- HopperHop failed and was dropped.
- Go1 may remain a stress test if it underperforms flat; report this honestly.
- Panda success must be scoped to pick/lift unless place success is non-trivial.
- Current object/state features are privileged semantic features, not learned from RGB.

## Reproducibility

Include:

- exact git commit hash;
- `git diff --stat HEAD~1..HEAD`;
- `pip freeze`;
- JAX backend and device list;
- MuJoCo Playground package version or git commit;
- exact scripts used to run the matrix;
- source snapshot zip.
