#!/usr/bin/env bash
# Batch A of the env-reliability loop: pure budget/exploration scaling, no code
# changes. Question per env: does the phase-2 weakness persist at ~10x the
# final-matrix training budget?
set -uo pipefail

ROOT="${ROOT:-runs/loop_fix/batchA}"
mkdir -p "$ROOT"

run() {
  local name="$1"; shift
  if [[ -f "$ROOT/${name}.pkl" ]]; then
    echo "=== $name already done, skipping ==="
    return 0
  fi
  echo "=== $name start $(date -Is) ==="
  python -m nexus_continuous.scripts.train_nexus_playground "$@" \
    --no-wandb \
    --save "$ROOT/${name}.pkl" \
    > "$ROOT/${name}.out" 2> "$ROOT/${name}.err"
  echo "=== $name exit=$? $(date -Is) ==="
}

common=(--override NUM_SEEDS=1 --override EVAL_AFTER_TRAIN=true
        --override EVAL_NUM_ENVS=64 --override EVAL_NUM_EPISODES=128)

# 400 updates at 2048 envs x 64 steps.
WALKER_TS=52428800
# 400 updates at 2048 envs (hopper config default is 1024; we double it for
# more parallel discovery of the hop).
HOPPER_TS=52428800
# 400 updates at 1024 envs.
PANDA_TS=26214400
# 500 updates at 2048 envs.
GO1_TS=65536000

walker() {
  local variant="$1" seed="$2"
  run "walker_${variant}_400up_seed${seed}" \
    --config "configs/walker_walk_${variant}.yaml" \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=$WALKER_TS \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

hopper() {
  local seed="$1"
  run "hopper_nesy_400up_explore_seed${seed}" \
    --config configs/hopper_hop_nesy.yaml \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=$HOPPER_TS \
    --override NUM_ENVS=2048 \
    --override NOISE_FINISH=0.05 --override NOISE_DECAY=1.0 \
    --override META_EPS_FINISH=0.05 \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

panda() {
  local seed="$1"
  run "panda_nesy_400up_seed${seed}" \
    --config configs/panda_pick_cube_nesy.yaml \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=$PANDA_TS \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

go1() {
  local seed="$1"
  run "go1_nesy_500up_seed${seed}" \
    --config configs/go1_joystick_nesy_phase2.yaml \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=$GO1_TS \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

# Ordered so each env yields one data point early.
walker nesy 0
hopper 0
walker neural 0
panda 0
hopper 1
walker nesy 1
walker neural 1
hopper 2
panda 1
go1 0

echo "=== batch A complete $(date -Is) ==="
