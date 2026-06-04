#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

RUN_DIR="${RUN_DIR:-runs/finalization_one_seed}"
REVIEW_DIR="${REVIEW_DIR:-runs/finalization_one_seed_review}"
ZIP_PATH="${ZIP_PATH:-runs/nexus_one_seed_finalization_for_chatgpt.zip}"
mkdir -p "${RUN_DIR}"

CONFIGS=(
  configs/flat_go1_joystick.yaml
  configs/go1_joystick_neural.yaml
  configs/go1_joystick_nesy.yaml
)

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  "${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
    --config "${cfg}" \
    --override SEED=0 \
    --override SAVE_PATH="${RUN_DIR}/${name}_seed0.pkl"
done

"${PYTHON_BIN}" tools/collect_nexus_results.py \
  --runs "${RUN_DIR}" \
  --out "${REVIEW_DIR}" \
  --zip "${ZIP_PATH}"

"${PYTHON_BIN}" tools/plot_nexus_paper_figures.py \
  --review "${REVIEW_DIR}" \
  --out "${REVIEW_DIR}/plots/paper"
