#!/usr/bin/env bash
# Local CPU pipeline check: train every environment SHORT, then plot each.
# Purpose is to verify the whole pipeline (env -> train -> eval -> plot) runs end
# to end and produces plot files. It does NOT train good policies (budgets are
# tiny for CPU) -- do the real training on Colab GPU with the configs' defaults.
#
# Usage (from nexus_continuous_control/, venv active):
#   bash tools/run_all_local.sh
# Override budget/envs via env vars:
#   STEPS=200000 ENVS="cartpole_balance_nesy cheetah_run_nesy" bash tools/run_all_local.sh
#
# Resumable: skips any checkpoint that already exists.

set -u
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"   # force CPU unless caller overrides

OUT="${OUT:-runs/local_all}"
STEPS="${STEPS:-150000}"          # tiny CPU budget (pipeline check, not a good policy)
NUM_ENVS="${NUM_ENVS:-32}"
CRITIC_AGG="${CRITIC_AGG:-mean}"
SEED="${SEED:-0}"
ENVS="${ENVS:-cartpole_balance_nesy cheetah_run_nesy walker_walk_nesy hopper_hop_nesy panda_pick_cube_nesy go1_joystick_nesy}"

mkdir -p "$OUT"
echo "backend=$JAX_PLATFORMS  steps=$STEPS  num_envs=$NUM_ENVS  critic_agg=$CRITIC_AGG"
echo "envs: $ENVS"

for env in $ENVS; do
  pkl="$OUT/${env}_seed${SEED}.pkl"
  if [ -f "$pkl" ]; then
    echo "== skip (exists): $env"
    continue
  fi
  echo ""
  echo "========== TRAIN $env =========="
  python -m nexus_continuous.scripts.train_nexus_playground \
    --config "configs/${env}.yaml" \
    --override SEED="${SEED}" \
    --override TOTAL_TIMESTEPS="${STEPS}" \
    --override NUM_ENVS="${NUM_ENVS}" \
    --override CRITIC_AGG="${CRITIC_AGG}" \
    --override EVAL_AFTER_TRAIN=true \
    --override SAVE_PATH="$pkl" \
    2>&1 | tee "$OUT/${env}_seed${SEED}.log" | tail -8
done

echo ""
echo "========== PLOTS =========="
for env in $ENVS; do
  pkl="$OUT/${env}_seed${SEED}.pkl"
  [ -f "$pkl" ] && python tools/plot_run.py "$pkl"
done

echo ""
echo "Done. Plots (open from Windows Explorer / VS Code):"
ls -1 "$OUT"/*.curves.png 2>/dev/null || echo "  (no plots produced -- check the logs in $OUT)"
