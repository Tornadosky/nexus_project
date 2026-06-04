#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-runs/phase2_final_matrix}"
mkdir -p "$ROOT"

CONFIGS=(
  configs/flat_cartpole_balance.yaml
  configs/cartpole_balance_neural.yaml
  configs/cartpole_balance_nesy.yaml
  configs/cartpole_balance_symbolic.yaml
  configs/flat_cheetah_run.yaml
  configs/cheetah_run_neural.yaml
  configs/cheetah_run_nesy.yaml
  configs/flat_walker_walk.yaml
  configs/walker_walk_neural.yaml
  configs/walker_walk_nesy.yaml
  configs/flat_panda_pick_cube.yaml
  configs/panda_pick_cube_neural.yaml
  configs/panda_pick_cube_nesy.yaml
  configs/panda_pick_cube_symbolic.yaml
  configs/flat_go1_joystick.yaml
  configs/go1_joystick_neural.yaml
  configs/go1_joystick_nesy_phase2.yaml
)

read -r -a SEEDS <<< "${PHASE2_SEEDS:-0 1 2}"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  for seed in "${SEEDS[@]}"; do
    out="$ROOT/${name}_seed${seed}.pkl"
    echo "=== $cfg seed=$seed -> $out ==="
    python -m nexus_continuous.scripts.train_nexus_playground \
      --config "$cfg" \
      --override SEED="$seed" \
      --override NUM_SEEDS=1 \
      --override EVAL_AFTER_TRAIN=true \
      --override EVAL_NUM_ENVS=64 \
      --override EVAL_NUM_EPISODES=128 \
      --override EVAL_SEED=$((10000 + seed)) \
      --override SAVE_EVAL_ROLLOUTS=true \
      --save "$out" \
      > "$ROOT/${name}_seed${seed}.out" \
      2> "$ROOT/${name}_seed${seed}.err"
  done
done
