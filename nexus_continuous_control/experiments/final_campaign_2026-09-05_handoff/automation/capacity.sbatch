#!/bin/bash
#SBATCH --job-name=nexus_capacity
#SBATCH --account=mage_apu
#SBATCH --partition=apu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=108000
#SBATCH --time=02:00:00
#SBATCH --output=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/logs/%x_%A_%a.out
#SBATCH --error=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/logs/%x_%A_%a.err
set -euo pipefail
export XLA_FLAGS="${CAPACITY_XLA:---xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disable
export PYTHONPATH=/ptmp/akalenik/nexus/site
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=24
KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
methods=(nesy ppo)
srun --cpu-bind=cores --mem-bind=local /ptmp/akalenik/jaxrocm_venv/bin/python "$KIT/automation/core_capacity.py" "${methods[$SLURM_ARRAY_TASK_ID]}"
