#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
mkdir -p runs/final_research_matrix

CONFIGS=(
  configs/flat_cartpole_balance.yaml
  configs/cartpole_balance_neural.yaml
  configs/cartpole_balance_symbolic.yaml
  configs/cartpole_balance_nesy.yaml
  configs/flat_cheetah_run.yaml
  configs/cheetah_run_neural.yaml
  configs/cheetah_run_nesy.yaml
  configs/flat_walker_walk.yaml
  configs/walker_walk_neural.yaml
  configs/walker_walk_nesy.yaml
  configs/flat_hopper_hop.yaml
  configs/hopper_hop_neural.yaml
  configs/hopper_hop_nesy.yaml
  configs/flat_panda_pick_cube.yaml
  configs/panda_pick_cube_neural.yaml
  configs/panda_pick_cube_symbolic.yaml
  configs/panda_pick_cube_nesy.yaml
  configs/flat_go1_joystick.yaml
  configs/go1_joystick_neural.yaml
  configs/go1_joystick_nesy.yaml
)

if [[ -n "${SEED_LIST:-}" ]]; then
  read -r -a SEEDS <<< "${SEED_LIST}"
else
  SEEDS=(0 1 2)
fi

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  for seed in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
      --config "${cfg}" \
      --override SEED="${seed}" \
      --override SAVE_PATH="runs/final_research_matrix/${name}_seed${seed}.pkl"
  done
done

"${PYTHON_BIN}" tools/collect_nexus_results.py \
  --runs runs/final_research_matrix \
  --out runs/final_research_review \
  --zip runs/nexus_final_research_results_for_chatgpt.zip

"${PYTHON_BIN}" tools/plot_nexus_paper_figures.py \
  --review runs/final_research_review \
  --out runs/final_research_review/plots/paper

"${PYTHON_BIN}" tools/write_final_nexus_report.py \
  --review runs/final_research_review \
  --out docs/reports/continuous_nexus_results.md
