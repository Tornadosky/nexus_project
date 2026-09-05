# Final execution release: review and scope

The machine-readable release decision is `READY.json`. Installation and static tests alone are not a production approval. The release is sealed only after the device, checkpoint-restoration, capacity, and media checks referenced there have passed.

## What changed, and why

| Earlier state | Released behavior |
|---|---|
| Old dirty `main`, 58 commits behind GitHub | A separate `campaign/final-matrix-2026-09-05` worktree based on current `origin/main`; the original working tree is preserved. |
| Core fixes existed only in uncommitted working files | A hash-verified frozen core snapshot retains those fixes; the RGB snapshot comes from pinned GitHub main. |
| The callback assumed every method had a learned meta-network | The callback handles absent meta parameters and saves only the actor/meta parameters needed by evaluation snapshots. |
| The stop-command assertion looked for a metric that was not emitted | Evaluation explicitly logs the raw commanded-velocity norm and tests that recorded value. |
| RGB imports passed, but rendering crashed | A separate RGB environment pins Warp 1.12.0, which works with the installed MuJoCo 3.9 rendering path. The original environment was not downgraded. |
| Viper was the planned primary trainer | Its current production-size Go1 execution was not reliable; primary training and evaluation use the verified NVIDIA environment. Viper primary submission remains disabled. |
| Setup stopped at generic command templates | Exact machine paths, source/runtime gates, locked outputs, a sequential controller, checkpoint-only evaluations, media, and backups are wired. |

These are execution and validation changes. The research matrix, task definitions, training seeds, transition budgets, and algorithm conditions remain unchanged.

## Candid assessment

The earlier import/static-only verification was insufficient. It missed both the Warp rendering incompatibility and the production-size ROCm failure. Those errors were found by actual training/rendering/capacity tests, not by inspecting package names.

The MI300A training problem has **not** been declared fixed. Some APU tests, including a PPO capacity test, worked; the native continuous-NEXUS Go1 path still failed at the required production shape. Moving the primary cohort to a tested NVIDIA stack is an explicit hardware decision, not a hidden reduction in the experiment or a false green check on Viper.

The release does not depend on the RTX 5080, which was not available for verification through this connection. A successful short capacity test is not a guarantee against a later interruption, disk failure, or a learning failure.

## Evidence and data separation

`verification/wsl_core.json` records the 16 core/LLM task-and-method smoke families, saved-checkpoint evaluation tests, and four production-shape core capacity tests. `verification/wsl_rgb.json` records six vision smoke families, two full-shape image capacity tests, and the pinned-model inference check.

`verification/rgb_saved_restore.json` separately checks reloading all six saved vision policies. The media smoke receipts confirm real simulator video/frame creation and logged rule eligibility/meta-Q values. No simulated reward from a smoke or capacity run is a paper result.

The primary allocation is still 60 core runs, 10 hand-written LLM references, 6 LLM feedback pilots, 36 final LLM runs, and 30 RGB runs: **142 maximum**. An invalid generated proposal is recorded as a generation failure; it is not replaced by a silently hand-written or repeatedly resampled skillset. Dependent invalid cells are reported rather than treated as completed training.

The evaluation matrix includes learning-curve snapshots, actor/objective probes, actor removal, selector interventions, actuation noise, changed Go1 commands, rough-terrain transfer, LLM validation/final evaluation, and image corruptions. Rendering uses preselected seed 41000 and training seed 0, not a search for a visually successful run.

## Exact location and execution

Open this worktree in WSL Ubuntu:

```text
/home/smirn/nexus_project_final
```

The package is under:

```text
nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff
```

To print the plan, then execute the complete sequence:

```bash
set -euo pipefail
cd /home/smirn/nexus_project_final/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff
bash RUN_ALL.sh
bash RUN_ALL.sh --execute
```

`docs/RUNBOOK.md` also gives every individual phase, output locations, and preservation/retry rules. The controller does not run during installation or readiness verification. It has a controller lock in addition to the existing per-GPU lock.

## Preservation and remaining operational risks

No original checkpoint was deleted, deduplicated, or rewritten. The six audited critical files in the original working core still match their earlier hashes. The old incomplete installation is not the release entry point.

Do **not** delete the original project directory or upgrade its environment while the campaign is running: the verified core interpreter still lives in its `.venv-wsl312`, and the isolated RGB overlay uses those installed base dependencies. The training workers import the frozen release source, not the mutable original source files.

Core results and evaluation evidence are assigned to `/mnt/d/nexus_final_campaign_2026-09-05`. RGB outputs use their separately recorded WSL profile. Space guards remain enabled. Backups copy to the existing Viper connection without `--delete`; a backup connection failure is reported and does not erase local work.

The full-budget matrix was deliberately not started during preparation. Production-shape tests establish that the selected tasks and array sizes execute; they do not measure the complete campaign's wall time or prove statistical significance. This release closes setup and wiring for the inspected machine, not the possibility of an infrastructure interruption. Failed runs must be preserved and any identical retry recorded; do not retune the matrix in response to disappointing returns.

The report's analysis, figure composition, and the interpretation of positive or negative results follow data collection. The package does not claim that named skills, LLM refinement, or added images will necessarily improve performance.
