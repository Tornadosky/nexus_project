# Candid implementation review — final verification pass

The earlier portable handoff was not production-ready. Static tests and source-anchor checks did not establish GPU execution, renderer compatibility, save/restore correctness, or production-batch capacity. This pass exercised those paths and preserved the failed attempts rather than relabeling them as successful runs.

## Corrections

| Before | Correction and location |
|---|---|
| Blindly updating the dirty main branch could discard local HPQN/core fixes. | New `campaign/final-matrix-2026-09-05` worktree starts at current GitHub main; `sources/core` freezes the actual local core and `sources/rgb` freezes the latest-main RGB implementation. Original checkout untouched. |
| Snapshot callback received a complete training state and assumed `state.meta` always existed. | `scripts/train_state.py` sends only actor/meta parameters, update counter and normalization statistics to the host, and handles missing meta state for flat/symbolic agents. Learning equations unchanged. |
| Stop-condition guard used a legacy norm containing a positive numerical epsilon. | `scripts/evaluate.py` separately records the raw commanded planar norm and checks it plus absolute yaw command. Tracking rewards/thresholds unchanged. |
| Import-only RGB checks passed despite incompatible Warp 1.13 rendering. | Separate `/home/smirn/nexus_campaign_rgb_venv` pins Warp 1.12.0 and inherits the existing dependencies. Original RL environment not upgraded or downgraded. |
| Runtime identity omitted the Warp version. | `scripts/agent.py` binds executable, NumPy/Warp and other ML versions, and XLA flags to smoke/run identity. |
| LLM generator environment/model were missing. | Separate CPU-only Torch/Transformers environment, pinned downloaded Qwen revision, and an actual model inference test. No primary scientific proposal was generated during readiness. |
| Viper was assumed production-capable from small tests. | Full-batch tests failed with ROCm illegal-address/segmentation faults, including the original trainer without snapshot callbacks. Viper is excluded from primary training; its old files and failed diagnostics remain intact. |
| Core results could overfill C: or mix different simulator versions. | Core/LLM results and common evaluations use D: and one NVIDIA/MuJoCo 3.9 cohort; Viper is an independent backup destination. |
| Potential RGB checkpoint reuse was counted without actual weights. | No primary reuse is approved. The locked maximum remains 142 rows. |

## What is deliberately not claimed

Readiness tests do not establish learned skill quality, superiority over PPO, or a successful LLM refinement result. The primary matrix still has to run.
Short production-shape checks establish executable tensor shapes and indicative throughput, not a guaranteed completion date or a full-budget learning result.
The RTX 5080 has not been inspected. It is not required by the active launch sequence.
A generic all-in-one controller file could not be written through the connector; the existing phase launcher is the maintained entry point. The full sequential command block is in `RUNBOOK.md`.
