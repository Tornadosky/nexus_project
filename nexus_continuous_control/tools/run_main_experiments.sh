#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${NEXUS_RUN_ROOT:-runs/verification}"
SEEDS="${NEXUS_SEEDS:-0 1 2}"
mkdir -p "${RUN_ROOT}/main" "${RUN_ROOT}/logs"

run_job() {
  local run_id="$1"
  local config="$2"
  local seed="$3"
  shift 3
  echo "=== Running ${run_id} seed${seed} ==="
  python -m nexus_continuous.scripts.train_nexus_playground \
    --config "${config}" \
    --override SEED="${seed}" \
    --override SAVE_PATH="${RUN_ROOT}/main/${run_id}_seed${seed}.pkl" \
    "$@" \
    2>&1 | tee "${RUN_ROOT}/logs/${run_id}_seed${seed}.log"
}

for seed in ${SEEDS}; do
  run_job cartpole_balance_nesy configs/cartpole_balance_nesy.yaml "${seed}"
  run_job cartpole_balance_symbolic configs/cartpole_balance_symbolic.yaml "${seed}"
  run_job cartpole_balance_neural configs/cartpole_balance_nesy.yaml "${seed}" --override META_POLICY_TYPE=neural
  run_job cheetah_run_nesy configs/cheetah_run_nesy.yaml "${seed}"
  run_job walker_walk_nesy configs/walker_walk_nesy.yaml "${seed}"
  run_job hopper_hop_nesy configs/hopper_hop_nesy.yaml "${seed}"
  run_job panda_pick_cube_nesy configs/panda_pick_cube_nesy.yaml "${seed}"
  run_job panda_pick_cube_symbolic configs/panda_pick_cube_symbolic.yaml "${seed}"
  run_job panda_pick_cube_neural configs/panda_pick_cube_nesy.yaml "${seed}" --override META_POLICY_TYPE=neural
done
