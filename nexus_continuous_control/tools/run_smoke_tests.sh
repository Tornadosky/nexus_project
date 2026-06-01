#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${NEXUS_RUN_ROOT:-runs/verification}"
mkdir -p "${RUN_ROOT}/smoke" "${RUN_ROOT}/logs"

python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml \
  --override TOTAL_TIMESTEPS=131072 \
  --override NUM_ENVS=64 \
  --override NUM_STEPS=32 \
  --override NUM_EPOCHS=1 \
  --override NUM_MINIBATCHES=4 \
  --override SEED=0 \
  --override PRINT_EVERY=32768 \
  --override SAVE_PATH="${RUN_ROOT}/smoke/cartpole_balance_nesy_smoke_seed0.pkl" \
  2>&1 | tee "${RUN_ROOT}/logs/cartpole_balance_nesy_smoke_seed0.log"

python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/panda_pick_cube_nesy.yaml \
  --override TOTAL_TIMESTEPS=131072 \
  --override NUM_ENVS=64 \
  --override NUM_STEPS=32 \
  --override NUM_EPOCHS=1 \
  --override NUM_MINIBATCHES=4 \
  --override SEED=0 \
  --override PRINT_EVERY=32768 \
  --override SAVE_PATH="${RUN_ROOT}/smoke/panda_pick_cube_nesy_smoke_seed0.pkl" \
  2>&1 | tee "${RUN_ROOT}/logs/panda_pick_cube_nesy_smoke_seed0.log"
