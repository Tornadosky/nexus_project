#!/usr/bin/env bash
# Batch C of the env-reliability loop. Waits for batch B's driver to exit.
# Rationale (see runs/loop_fix/LOOP_NOTES.md):
#   - Walker: budget (batch A, 4/4) and K10 commitment (batch B, 2/2) both
#     failed to produce net locomotion; the untried lever is the hopper
#     recipe — sustained exploration so the gait can be discovered before
#     noise collapses into the sway optimum.
#   - Panda: nesy 400up is seed-bimodal {0.02, 0.34}; sustained meta-eps is
#     the de-bimodalization candidate (same logic that fixed hopper).
set -uo pipefail

ROOT="${ROOT:-runs/loop_fix/batchC}"
mkdir -p "$ROOT"

while pgrep -f run_loop_fix_batchB >/dev/null; do sleep 120; done

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

# Walker, hopper-style sustained exploration, 400 updates.
walker_explore() {
  local variant="$1" seed="$2"
  run "walker_${variant}_explore_400up_seed${seed}" \
    --config "configs/walker_walk_${variant}.yaml" \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=52428800 \
    --override NOISE_FINISH=0.05 --override NOISE_DECAY=1.0 \
    --override META_EPS_FINISH=0.05 \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

# Panda nesy, sustained meta-exploration (skill-level de-bimodalization).
panda_eps() {
  local seed="$1"
  run "panda_nesy_susteps_400up_seed${seed}" \
    --config configs/panda_pick_cube_nesy.yaml \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=26214400 \
    --override META_EPS_FINISH=0.05 --override META_EPS_DECAY=1.0 \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

# Ordered so each hypothesis yields one data point early.
walker_explore nesy 0
panda_eps 0
walker_explore neural 0
panda_eps 2
walker_explore nesy 1

echo "=== batch C complete $(date -Is) ==="
