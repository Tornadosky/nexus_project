# Complete execution sequence — verified NVIDIA campaign

Run in **WSL Ubuntu**, not the default Ubuntu-20.04 distribution. From PowerShell: `wsl.exe -d Ubuntu`.
Use the new worktree. Do not pull, reset, switch, or clean the original dirty checkout.
No production experiment was started while preparing this release. Readiness/capacity outputs are excluded from the paper cohort.

## 1. Verify the installed release

```bash
set -euo pipefail
export REPO=/home/smirn/nexus_project_final
export KIT="$REPO/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff"
cd "$KIT"
git -C "$REPO" status --short --branch
sha256sum --quiet -c SHA256SUMS.txt
python3 -c 'import json; r=json.load(open("READY.json")); assert r["production_ready"]; print(json.dumps(r,indent=2))'
bash RUN_MATRIX.sh status
```

The release is pinned to its recorded source and runtimes. A checksum or identity failure means stop and inspect; never remove a guard or update expected hashes to suppress it.
Missing scientific LLM proposals before the generation phases are expected; they are experiment outputs, not missing installation files.
Keep the WSL session alive while running. The phase commands are blocking, log to the terminal, and return a nonzero status on infrastructure failures.
Every GPU training/evaluation path uses the same local GPU lock. Do not run older project GPU queues concurrently.

## 2. Execute the complete locked matrix

The same complete sequence is wired into one foreground controller, with a controller lock, a log, between-phase backups, and a final backup. No jobs start without `--execute`.

```bash
set -euo pipefail
cd /home/smirn/nexus_project_final/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff
bash RUN_ALL.sh
bash RUN_ALL.sh --execute
```

The first command prints the plan; the second executes it. Keep the WSL session alive. The individual equivalent phase commands are listed below.

The following block is the full sequence, including validation-driven refinement and common checkpoint evaluations. It does not change the scientific budgets or select a better-performing seed. It can be rerun: completed identical rows are verified and skipped; partial rows stop rather than being overwritten.

```bash
set -euo pipefail
cd /home/smirn/nexus_project_final/nexus_continuous_control/experiments/final_campaign_2026-09-05_handoff
mkdir -p /mnt/d/nexus_final_campaign_2026-09-05/logs
LOG=/mnt/d/nexus_final_campaign_2026-09-05/logs/matrix_$(date -u +%Y-%m-%d_%H-%M-%S).log
exec > >(tee -a "$LOG") 2>&1
bash RUN_MATRIX.sh core --execute
bash RUN_MATRIX.sh initial --execute
bash RUN_MATRIX.sh resample --execute
bash RUN_MATRIX.sh llm_reference --execute
bash RUN_MATRIX.sh llm_pilot --execute
bash RUN_MATRIX.sh pilot_eval --execute
bash RUN_MATRIX.sh refine --execute
bash RUN_MATRIX.sh llm_final --execute
bash RUN_MATRIX.sh rgb --execute
bash RUN_MATRIX.sh curves --execute
bash RUN_MATRIX.sh probes --execute
bash RUN_MATRIX.sh shifts --execute
bash RUN_MATRIX.sh llm_eval --execute
bash RUN_MATRIX.sh media --execute
bash RUN_MATRIX.sh backup --execute
```

Initial/resample generation uses the pinned local Qwen model, CPU-only Torch, three declared families per task, and at most two schema/type repairs. A failed bounded generation remains a reported failure. The selection files record which training rows were not executable and why. Refinement reads only the pilot validation summaries (seed 20000), never final-test metrics (seed 30000).
All common curve, actor-probe, rule-intervention, command-shift, noise, and terrain-transfer evaluations run on the same NVIDIA/MuJoCo 3.9 stack as the corresponding primary training. RGB corruptions and intact evaluation are part of each RGB row.
Backup uses rsync without deletion into a new Viper directory. A backup connection failure does not invalidate completed training; rerun the backup phase after reconnecting.

## 3. Output map and completion criteria

Core/LLM checkpoints: `/mnt/d/nexus_final_campaign_2026-09-05/results/<matrix-id>/`.
Common evaluations: `/mnt/d/nexus_final_campaign_2026-09-05/evidence/{curves,probes,shifts,pilot,llm}/`.
RGB checkpoints, learning histories and corruption evaluations: `/home/smirn/nexus_campaign_verified_v2/rgb_release/<matrix-id>/`.
LLM specifications and raw generation/repair provenance: `/home/smirn/nexus_campaign_specs_2026-09-05/`.
Fixed-seed real videos, PNG frames, rule eligibility, and meta-Q traces: `/mnt/d/nexus_final_campaign_2026-09-05/media/`.
Backup: `/ptmp/akalenik/nexus/nvidia_final_campaign_2026-09-05_backup/`.

Exclude every `__smoke`, `capacity`, `api_test`, and readiness output from scientific analyses. Only the 142 declared matrix IDs, minus explicitly recorded invalid-generation dependencies, define the cohort.
Each trained state/PPO run must have `COMPLETE.json`, a matching final checkpoint hash, and 11 actual-step snapshots. RGB rows instead have a final policy plus the original RGB learning histories and intact/corruption evaluations.
The full valid-generation evaluation schedule contains 660 curve checkpoints, 300 probe conditions, 420 shifts, 6 pilot validations, and 46 final LLM/reference evaluations. Invalid generated policies reduce only their documented dependent cells; they never justify replacing a failed proposal or seed.

## 4. Infrastructure interruption

No command clears partial outputs. Intermediate snapshots are evaluation-only, not optimizer-resumable checkpoints. Preserve the partial directory and log, diagnose the interruption, and rerun the identical row in a fresh `--results` directory. Do not change rewards, hyperparameters, seeds, or the LLM proposal to rescue performance. Record the retry and select its verified completed artifact by explicit provenance, not by its score.
No full-size run-duration guarantee is possible from short readiness probes. The probes establish executable production shapes and indicative throughput; each full run records its actual steps and elapsed time. Keep the GPU available and review the first completed full-budget runs before relying on a calendar estimate.

## 5. What not to do

Do not submit the old Viper training scripts for the primary matrix. Do not upgrade JAX/MuJoCo/Warp midway through the cohort. Do not count historical endpoint checkpoints as missing trajectory snapshots. Do not run another algorithm/environment sweep after this matrix.
The original working tree was deliberately not rebased because it contains uncommitted fixes. Open `/home/smirn/nexus_project_final` on the dedicated campaign branch; that is the active implementation.
Figure composition, interpretation, student contribution attribution, and report writing follow the completed data. No claim of successful learning is made by the readiness tests.
