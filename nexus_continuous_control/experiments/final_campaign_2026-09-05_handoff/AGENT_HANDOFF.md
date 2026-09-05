# Coding-agent handoff

Open the WSL Ubuntu worktree `/home/smirn/nexus_project_final` on `campaign/final-matrix-2026-09-05`.
The active package is `nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff`.
Do not continue from the original Windows checkout or the old guarded partial installation.

## Assignment

Execute the complete sequence in `docs/RUNBOOK.md`, after checking `READY.json` and `SHA256SUMS.txt`. All actual training/evaluation parameters are already in `plan/` and the worker scripts. Do not design more experiments or retune an underperforming method.
Use `bash RUN_ALL.sh --execute` for the complete fixed sequence, or `RUN_MATRIX.sh <phase> --execute` for an individual phase. Without the explicit argument, either entry point only prints the plan. The controller is a thin sequential wrapper over the same tested phase commands.
The maximum is 142 production training rows. Readiness/smoke/capacity runs are not part of those rows or the paper.
All core/LLM primary training and their common evaluations use the verified NVIDIA/MuJoCo 3.9 environment. RGB uses its isolated Warp 1.12 environment. Viper is backup-only after reproducible full-batch ROCm failures. The 5080 is not required.

## Preserve evidence

Never delete a partial run, replace an unsuccessful seed/specification, edit the source manifest to bypass an error, or mix old endpoint runs into the new curve cohort.
Generation may produce a bounded validation failure. `select_executable.py` records that outcome and excludes only its dependent training cells. It must not trigger another unplanned proposal.
Use the pilot validation summaries for refinement. Do not feed final-test results back into the LLM.
Check actual snapshot budgets and completion hashes. Keep the explicit per-episode tables, raw LLM proposals/repair histories, and intact/corrupted image evaluations.
Run the backup phase after major stages as well as at the end. It does not delete either source or destination data.

## Completion

Deliver the trained matrix and declared checkpoint evaluations, a table of any recorded invalid-generation cells, actual compute times, and the fixed figure ingredients. Research conclusions follow the observations, not readiness test scores.
Final figure composition, student-authored analysis, contribution/LLM-use statements, and paper editing remain report work, not additional training.
The full executable source/configuration text is in `ALL_CODE.md`; the final archive and Git branch preserve the exact release.
