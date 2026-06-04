#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
mkdir -p runs/final_go1_matrix

CONFIGS=(
  configs/flat_go1_joystick.yaml
  configs/go1_joystick_neural.yaml
  configs/go1_joystick_nesy.yaml
)
SEEDS=(0 1 2)

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  for seed in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
      --config "${cfg}" \
      --override SEED="${seed}" \
      --override SAVE_PATH="runs/final_go1_matrix/${name}_seed${seed}.pkl"
  done
done
