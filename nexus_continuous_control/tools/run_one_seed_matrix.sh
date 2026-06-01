#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${NEXUS_RUN_ROOT:-runs/verification}"
mkdir -p "${RUN_ROOT}/one_seed" "${RUN_ROOT}/logs"

run_job() {
  local run_id="$1"
  local config="$2"
  shift 2
  echo "=== Running ${run_id} seed0 ==="
  python -m nexus_continuous.scripts.train_nexus_playground \
    --config "${config}" \
    --override SEED=0 \
    --override SAVE_PATH="${RUN_ROOT}/one_seed/${run_id}_seed0.pkl" \
    "$@" \
    2>&1 | tee "${RUN_ROOT}/logs/${run_id}_seed0.log"
}

run_job cartpole_balance_nesy configs/cartpole_balance_nesy.yaml
run_job cartpole_balance_symbolic configs/cartpole_balance_symbolic.yaml
run_job cartpole_balance_neural configs/cartpole_balance_nesy.yaml --override META_POLICY_TYPE=neural
run_job cheetah_run_nesy configs/cheetah_run_nesy.yaml
run_job walker_walk_nesy configs/walker_walk_nesy.yaml
run_job hopper_hop_nesy configs/hopper_hop_nesy.yaml
run_job panda_pick_cube_nesy configs/panda_pick_cube_nesy.yaml
run_job panda_pick_cube_symbolic configs/panda_pick_cube_symbolic.yaml
run_job panda_pick_cube_neural configs/panda_pick_cube_nesy.yaml --override META_POLICY_TYPE=neural
