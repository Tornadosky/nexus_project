#!/bin/bash
#SBATCH --job-name=nexus_final
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
: "${CAMPAIGN_ROOT:?Set installed package path}" "${RUN_LIST:?One matrix ID per line}"
index="${SLURM_ARRAY_TASK_ID:?Array index is required}"
RUN_ID="$(sed -n "$((index + 1))p" "$RUN_LIST")"
[ -n "$RUN_ID" ] || { echo "No run at array index $index"; exit 2; }
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disable
export PYTHONPATH=/ptmp/akalenik/nexus/site
args=("$CAMPAIGN_ROOT/scripts/agent.py" run --profile viper --id "$RUN_ID" --execute)
if [[ "${SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
# This uses exactly one allocated APU and the already installed ROCm interpreter.
# No retry loops: a failed/partial run remains visible and cannot be overwritten.
srun /ptmp/akalenik/jaxrocm_venv/bin/python "${args[@]}"
