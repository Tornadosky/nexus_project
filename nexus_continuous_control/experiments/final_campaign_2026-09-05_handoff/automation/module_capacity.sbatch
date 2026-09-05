#!/bin/bash
#SBATCH --job-name=nexus_module_capacity
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
module load python-waterboa/2025.06 rocm/7.2 jax/0.8.2
export PYTHONPATH="$PYTHONPATH:/ptmp/akalenik/nexus/site"
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disable PYTHONUNBUFFERED=1 OMP_NUM_THREADS=24
export CAPACITY_LABEL='module082ready_{method}'
unset CAPACITY_CHUNK_ENV
KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
methods=(nesy ppo)
srun --cpu-bind=cores --mem-bind=local /ptmp/akalenik/nexus_campaign_module_venv/bin/python "$KIT/automation/core_capacity.py" "${methods[$SLURM_ARRAY_TASK_ID]}"
