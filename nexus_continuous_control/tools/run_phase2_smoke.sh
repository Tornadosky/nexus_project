#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-runs/phase2_smoke}"
mkdir -p "$ROOT"

python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/cartpole_balance_nesy.yaml \
  --override TOTAL_TIMESTEPS=262144 \
  --override NUM_ENVS=128 \
  --override NUM_STEPS=32 \
  --override NUM_MINIBATCHES=8 \
  --override EVAL_AFTER_TRAIN=true \
  --override EVAL_NUM_ENVS=16 \
  --override EVAL_NUM_EPISODES=16 \
  --save "$ROOT/cartpole_nesy_smoke_seed0.pkl"

python -m nexus_continuous.scripts.train_nexus_playground \
  --config configs/panda_pick_cube_nesy.yaml \
  --override TOTAL_TIMESTEPS=524288 \
  --override NUM_ENVS=128 \
  --override NUM_STEPS=32 \
  --override NUM_MINIBATCHES=8 \
  --override EVAL_AFTER_TRAIN=true \
  --override EVAL_NUM_ENVS=16 \
  --override EVAL_NUM_EPISODES=16 \
  --save "$ROOT/panda_nesy_smoke_seed0.pkl"
