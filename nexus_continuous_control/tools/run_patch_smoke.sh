#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
mkdir -p runs/patch_smoke

"${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_symbolic.yaml \
  --override SEED=0 \
  --override TOTAL_TIMESTEPS=65536 \
  --override SAVE_PATH=runs/patch_smoke/cartpole_symbolic_seed0.pkl

"${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/panda_pick_cube_symbolic.yaml \
  --override SEED=0 \
  --override TOTAL_TIMESTEPS=65536 \
  --override SAVE_PATH=runs/patch_smoke/panda_symbolic_seed0.pkl

"${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/panda_pick_cube_nesy.yaml \
  --override SEED=0 \
  --override TOTAL_TIMESTEPS=65536 \
  --override SAVE_PATH=runs/patch_smoke/panda_nesy_seed0.pkl

"${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/hopper_hop_nesy.yaml \
  --override SEED=0 \
  --override TOTAL_TIMESTEPS=65536 \
  --override SAVE_PATH=runs/patch_smoke/hopper_nesy_seed0.pkl

"${PYTHON_BIN}" tools/collect_nexus_results.py \
  --runs runs/patch_smoke \
  --out runs/patch_smoke_review \
  --zip runs/nexus_patch_smoke_for_chatgpt.zip
