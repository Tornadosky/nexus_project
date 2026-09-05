#!/bin/bash
#SBATCH --job-name=nexus_native_capacity
#SBATCH --account=mage_apu
#SBATCH --partition=apu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=108000
#SBATCH --time=01:00:00
#SBATCH --output=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/logs/%x_%j.out
#SBATCH --error=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified/logs/%x_%j.err
set -euo pipefail
KIT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
export PYTHONPATH=/ptmp/akalenik/nexus/site
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=disable PYTHONUNBUFFERED=1 OMP_NUM_THREADS=24
srun /ptmp/akalenik/jaxrocm_venv/bin/python "$KIT/automation/native_capacity.py"
