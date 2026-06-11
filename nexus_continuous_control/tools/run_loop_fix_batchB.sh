#!/usr/bin/env bash
# Batch B of the env-reliability loop. Waits for batch A's driver to exit,
# then runs sequentially at full GPU:
#   1-2. walker nesy/neural 400up with META_DECISION_INTERVAL=10 (gait-dithering
#        hypothesis; budget alone left fwd-vel at ~0).
#   3.   cartpole symbolic retune 150up x3 seeds (urgent band 0.35/2.5).
#   4-5. hopper + walker symbolic 400up with env-aligned wrap-safe rules.
#   6.   panda neural anti-grasp-trap probe (skill commit + sustained meta-eps).
set -uo pipefail

ROOT="${ROOT:-runs/loop_fix/batchB}"
mkdir -p "$ROOT"

while pgrep -f run_loop_fix_batchA >/dev/null; do sleep 120; done

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

run walker_nesy_400up_K10_seed0 \
  --config configs/walker_walk_nesy.yaml "${common[@]}" \
  --override TOTAL_TIMESTEPS=52428800 --override META_DECISION_INTERVAL=10 \
  --override SEED=0 --override EVAL_SEED=10000

run walker_neural_400up_K10_seed0 \
  --config configs/walker_walk_neural.yaml "${common[@]}" \
  --override TOTAL_TIMESTEPS=52428800 --override META_DECISION_INTERVAL=10 \
  --override SEED=0 --override EVAL_SEED=10000

for s in 0 1 2; do
  run "cartpole_symbolic_retune_150up_seed${s}" \
    --config configs/cartpole_balance_symbolic.yaml "${common[@]}" \
    --override TOTAL_TIMESTEPS=9830400 \
    --override SEED="$s" --override EVAL_SEED=$((10000 + s))
done

run hopper_symbolic_rulefix_400up_seed0 \
  --config configs/hopper_hop_nesy.yaml "${common[@]}" \
  --override META_POLICY_TYPE=symbolic \
  --override TOTAL_TIMESTEPS=52428800 --override NUM_ENVS=2048 \
  --override NOISE_FINISH=0.05 --override NOISE_DECAY=1.0 \
  --override SEED=0 --override EVAL_SEED=10000

run walker_symbolic_rulefix_400up_seed0 \
  --config configs/walker_walk_nesy.yaml "${common[@]}" \
  --override META_POLICY_TYPE=symbolic \
  --override TOTAL_TIMESTEPS=52428800 \
  --override SEED=0 --override EVAL_SEED=10000

run panda_neural_antitrap_400up_seed0 \
  --config configs/panda_pick_cube_neural.yaml "${common[@]}" \
  --override TOTAL_TIMESTEPS=26214400 \
  --override META_DECISION_INTERVAL=10 \
  --override META_EPS_FINISH=0.15 --override META_EPS_DECAY=1.0 \
  --override SEED=0 --override EVAL_SEED=10000

echo "=== batch B complete $(date -Is) ==="
