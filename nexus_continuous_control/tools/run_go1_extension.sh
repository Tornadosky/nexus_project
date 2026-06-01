#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${NEXUS_RUN_ROOT:-runs/verification}"
SEEDS="${NEXUS_SEEDS:-0 1 2}"
mkdir -p "${RUN_ROOT}/extension" "${RUN_ROOT}/logs"

for seed in ${SEEDS}; do
  echo "=== Running go1_joystick_nesy seed${seed} ==="
  python -m nexus_continuous.scripts.train_nexus_playground \
    --config configs/go1_joystick_nesy.yaml \
    --override SEED="${seed}" \
    --override SAVE_PATH="${RUN_ROOT}/extension/go1_joystick_nesy_seed${seed}.pkl" \
    2>&1 | tee "${RUN_ROOT}/logs/go1_joystick_nesy_seed${seed}.log"
done
