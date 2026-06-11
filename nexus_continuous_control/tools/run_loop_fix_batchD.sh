#!/usr/bin/env bash
# Batch D of the env-reliability loop. Waits for batch C's driver to exit.
# Single hypothesis: walker walk-skill posture shaping (walk reward now
# carries 0.5*(height+upright)) unlocks net locomotion. Batches A-C showed
# budget, K10 commitment, and sustained exploration all fail without it.
# Shaped runs keep the sustained-exploration overrides (discovery still
# needed; explore alone was necessary-not-sufficient on hopper's evidence).
# Pre-registered gate: net_walk_success_rate > 0.3 on >=1 seed = hypothesis
# confirmed -> ablate explore + add seeds; all seeds ~0 = policy-shaping
# branch also dead -> document walker as limitation.
set -uo pipefail

ROOT="${ROOT:-runs/loop_fix/batchD}"
mkdir -p "$ROOT"

while pgrep -f run_loop_fix_batchC >/dev/null; do sleep 120; done

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

walker_shaped() {
  local variant="$1" seed="$2"
  run "walker_${variant}_shaped_400up_seed${seed}" \
    --config "configs/walker_walk_${variant}.yaml" \
    "${common[@]}" \
    --override TOTAL_TIMESTEPS=52428800 \
    --override NOISE_FINISH=0.05 --override NOISE_DECAY=1.0 \
    --override META_EPS_FINISH=0.05 \
    --override SEED="$seed" --override EVAL_SEED=$((10000 + seed))
}

walker_shaped nesy 0
walker_shaped neural 0
walker_shaped nesy 1

echo "=== batch D complete $(date -Is) ==="
