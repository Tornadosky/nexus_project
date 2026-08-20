#!/usr/bin/env bash
# Higher-power RE-EVALUATION of the state-vs-state+RGB campaign.
#
# WHY: the campaign headline was quoted from `results.intact`, a 5-episode
# deterministic eval. Five episodes at these effect sizes has ~0.16 achieved
# power, which is why the eval metric and the training-return metric disagreed
# on walker. NOTHING is retrained here: the 18 policies pickled in ~/runs_spr
# are re-loaded with --load-policy and re-scored with EPISODES=30.
#
# The original 5-episode JSONs are NOT touched. Output goes to a sibling tree,
# results/rgb/state_plus_rgb_eval30/, so both are on disk with provenance.
cd ~/nexus_project/nexus_continuous_control
source .venv/bin/activate
source ~/mesa_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false

LOG=~/state_plus_rgb_eval30.log
OUTROOT=results/rgb/state_plus_rgb_eval30
POLROOT=~/runs_spr
EPISODES=30
NEED_FREE=3000

log() { echo "$*" | tee -a "$LOG"; }

wait_for_gpu() {
  local waited=0
  while true; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$((total - used))
    if [ "$free" -ge "$NEED_FREE" ]; then
      [ "$waited" -gt 0 ] && log "    GPU free again after ${waited}s (${free} MiB)"
      return 0
    fi
    [ $((waited % 300)) -eq 0 ] && log "    waiting for GPU: ${free} MiB free, need ${NEED_FREE} (${waited}s)"
    sleep 30
    waited=$((waited + 30))
  done
}

for spec in "$@"; do
  IFS=: read -r ENV ARM SEED <<< "$spec"
  case "$ENV" in
    cartpole) CFGSTEM=cartpole_balance_nesy ;;
    walker)   CFGSTEM=walker_walk_nesy ;;
    cheetah)  CFGSTEM=cheetah_run_nesy ;;
    *) log "UNKNOWN ENV $ENV"; continue ;;
  esac
  CFG="configs/${CFGSTEM}_${ARM}.yaml"
  TAG="${ENV}/${ARM}_seed${SEED}"
  OUT="${OUTROOT}/${TAG}"
  POL="${POLROOT}/${ENV}_${ARM}_s${SEED}.pkl"
  EXTRA=""
  case "$ARM" in state_matched*) EXTRA="--no-rgb" ;; esac

  [ -f "$CFG" ] || { log "[$(date -Is)] MISSING CONFIG $CFG -- skip $TAG"; continue; }
  [ -f "$POL" ] || { log "[$(date -Is)] MISSING POLICY $POL -- skip $TAG"; continue; }
  if [ -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] SKIP ${TAG} (already done)"; continue
  fi

  mkdir -p "$OUT"
  wait_for_gpu
  log "[$(date -Is)] START ${TAG} (episodes=${EPISODES}, no retraining)"
  T0=$(date +%s)
  python -m nexus_continuous.scripts.rgb_pixel_ablation $EXTRA \
      --config "$CFG" --meta nesy --seed "$SEED" \
      --episodes "$EPISODES" --load-policy "$POL" \
      --out "$OUT" > "${OUT}/run.log" 2>&1
  RC=$?
  T1=$(date +%s); WALL=$((T1 - T0))
  if [ $RC -ne 0 ] || [ ! -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] FAILED ${TAG} rc=${RC} after ${WALL}s"
    grep -v "^Module " "${OUT}/run.log" | tail -20 | sed "s/^/      /" | tee -a "$LOG"
    continue
  fi
  log "[$(date -Is)] DONE ${TAG} in ${WALL}s"
  sleep 5
done
log "[$(date -Is)] EVAL30 BATCH COMPLETE: $*"
