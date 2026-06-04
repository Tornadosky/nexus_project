#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
mkdir -p runs/finalization_one_seed

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
)

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg}" .yaml)"
  "${PYTHON_BIN}" -m nexus_continuous.scripts.train_nexus_playground \
    --config "${cfg}" \
    --override SEED=0 \
    --override SAVE_PATH="runs/finalization_one_seed/${name}_seed0.pkl"
done

"${PYTHON_BIN}" tools/collect_nexus_results.py \
  --runs runs/finalization_one_seed \
  --out runs/finalization_one_seed_review \
  --zip runs/nexus_one_seed_finalization_for_chatgpt.zip

"${PYTHON_BIN}" tools/plot_nexus_paper_figures.py \
  --review runs/finalization_one_seed_review \
  --out runs/finalization_one_seed_review/plots/paper
