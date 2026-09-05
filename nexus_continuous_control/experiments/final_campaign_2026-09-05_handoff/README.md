# NEXUS final experiment campaign

Active branch: `campaign/final-matrix-2026-09-05`, based on GitHub main `7557d5d9b9c75fbe93091ead6ae525a1c377cdf6`.
Worktree: `/home/smirn/nexus_project_final` (WSL Ubuntu).
The original dirty Windows checkout and all historical checkpoints are preserved.

## Execution decision

The primary campaign uses the inspected RTX 4060 Ti, not the failing MI300A runtime.
All 112 non-image training rows and their evaluations use the original NVIDIA JAX/MuJoCo 3.9 environment.
The 30 image rows use a separate tested environment with Warp 1.12.0; the three input arms share that same environment.
Viper is an independent backup destination. Its full-batch ROCm training failed even without our snapshot instrumentation. Do not combine Viper/MuJoCo 3.10 results with this primary cohort.
The 5080 machine is not a prerequisite and has not been certified.

## Start

Read `docs/RUNBOOK.md` for the complete phase sequence. No dependency installation, source merging, or manual checkpoint discovery is part of that sequence.
`bash RUN_MATRIX.sh status` is read-only. Every production phase requires the explicit `--execute` argument and a passing `READY.json` plus code checksums.
The fixed matrix is `plan/matrix.csv`: maximum 142 runs (60 core, 52 LLM-related, 30 image). Invalid bounded LLM generations are recorded, not replaced until successful.
Core results and common evaluations go to `D:\nexus_final_campaign_2026-09-05`; image results use the documented WSL directory. No old outputs are deleted.

## Evidence and source

`verification/` contains machine-specific test receipts. `docs/REVIEW_REMOTE.md` records the corrections and limitations. `ALL_CODE.md` contains the complete source/configuration text.
`sources/core` preserves the audited working core; `sources/rgb` is the latest-main Git export. Their hashes are verified before launching.
Intermediate snapshots are for evaluation, not mid-run training resumption. A failed partial run is preserved and requires a documented identical retry in a fresh output directory.
Historical instructions under `docs/history/` are not active launch instructions.
