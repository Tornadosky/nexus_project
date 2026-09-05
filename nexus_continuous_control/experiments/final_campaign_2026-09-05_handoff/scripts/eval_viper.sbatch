#!/bin/bash
#SBATCH --job-name=nexus_final_eval
#SBATCH --partition=apu
#SBATCH --account=mage_apu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=108000
#SBATCH --time=24:00:00
#SBATCH --output=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/logs/%x_%A_%a.out
#SBATCH --error=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/logs/%x_%A_%a.err
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${EVAL_SUITE:?curves,probes,shifts,pilot,llm}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disable
export PYTHONPATH=/ptmp/akalenik/nexus/site
ROOT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified
srun /ptmp/akalenik/jaxrocm_venv/bin/python "$CAMPAIGN_ROOT/scripts/evaluate_campaign.py" \
 --matrix "${MATRIX_JSON:-$CAMPAIGN_ROOT/plan/matrix.json}" --repo "$CAMPAIGN_ROOT/sources/core" \
 --results "$ROOT/results_release" --out "$ROOT/evidence" --suite "$EVAL_SUITE" \
 --shards "${EVAL_SHARDS:-8}" --shard-index "${SLURM_ARRAY_TASK_ID:?}"
