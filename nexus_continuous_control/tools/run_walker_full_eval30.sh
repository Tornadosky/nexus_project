#!/usr/bin/env bash
# 30-EPISODE re-scoring of the WalkerWalk full-budget arms. NOTHING IS RETRAINED.
#
# WHY 30 AND NOT 5: the campaign's 5-episode deterministic eval has ~0.16
# achieved power at these effect sizes. It is what made the eval metric and the
# training metric disagree on walker at the matched budget, and it misled this
# project once already. The policies pickled by the training pass in
# ~/runs_spr_full are re-loaded with --load-policy and re-scored at 30 episodes,
# so this costs minutes rather than the ~36 h the training did.
#
# The 5-episode JSONs written during training are NOT touched: output goes to a
# sibling tree, results/rgb/state_plus_rgb_full_eval30/, so both are on disk with
# provenance. Reset keys are PRNGKey(9000 + 97*episode + seed) -- a function of
# (episode, seed) and NOT of the arm -- so both arms see identical initial
# states and the seeds are genuinely paired.
#
# Usage: run_walker_full_eval30.sh ARM:SEED [ARM:SEED ...]
# NB: no `set -u` -- ~/mesa_env.sh appends to an unset LD_LIBRARY_PATH.
cd ~/nexus_project/nexus_continuous_control
source .venv/bin/activate
source ~/mesa_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false

LOG=~/walker_full_campaign.log
OUTROOT=results/rgb/state_plus_rgb_full_eval30
POLROOT=~/runs_spr_full
ENV=walker
CFGSTEM=walker_walk_nesy
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
      [ "$waited" -gt 0 ] && log "[$(date -Is)]     GPU free again after ${waited}s (${free} MiB)"
      return 0
    fi
    [ $((waited % 300)) -eq 0 ] && log "[$(date -Is)]     WAITING for shared GPU: ${free} MiB free, need ${NEED_FREE} (${waited}s)"
    sleep 30
    waited=$((waited + 30))
  done
}

for spec in "$@"; do
  IFS=: read -r ARM SEED <<< "$spec"
  CFG="configs/${CFGSTEM}_${ARM}.yaml"
  TAG="${ENV}/${ARM}_seed${SEED}"
  OUT="${OUTROOT}/${TAG}"
  POL="${POLROOT}/${ENV}_${ARM}_s${SEED}.pkl"
  EXTRA=""
  case "$ARM" in state_matched*) EXTRA="--no-rgb" ;; esac

  [ -f "$CFG" ] || { log "[$(date -Is)] EVAL30 MISSING CONFIG $CFG -- skip $TAG"; continue; }
  [ -f "$POL" ] || { log "[$(date -Is)] EVAL30 MISSING POLICY $POL -- skip $TAG"; continue; }
  if [ -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] EVAL30 SKIP ${TAG} (already done)"; continue
  fi

  mkdir -p "$OUT"
  wait_for_gpu
  log "[$(date -Is)] EVAL30 START ${TAG} (episodes=${EPISODES}, no retraining)"
  T0=$(date +%s)
  python -u -m nexus_continuous.scripts.rgb_pixel_ablation $EXTRA \
      --config "$CFG" --meta nesy --seed "$SEED" \
      --episodes "$EPISODES" --load-policy "$POL" \
      --out "$OUT" > "${OUT}/run.log" 2>&1
  RC=$?
  T1=$(date +%s); WALL=$((T1 - T0))
  if [ $RC -ne 0 ] || [ ! -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] EVAL30 FAILED ${TAG} rc=${RC} after ${WALL}s"
    grep -v "^Module " "${OUT}/run.log" | tail -20 | sed "s/^/      /" | tee -a "$LOG"
    continue
  fi
  log "[$(date -Is)] EVAL30 DONE ${TAG} in ${WALL}s"
  python - "$OUT/pixel_ablation.json" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1])); i = d["results"]["intact"]
print(f"    intact reward_per_step = {i['reward_per_step_mean']:.4f} "
      f"+/- {i['reward_per_step_std']:.4f}  (n={len(i.get('per_episode', []))})")
if not d.get("state_only"):
    print("    camera conditions: " + ", ".join(
        f"{c}={100*v:+.1f}%" for c, v in d["performance_drop_fraction"].items()))
PY
  sleep 5
done
log "[$(date -Is)] ===== EVAL30 BATCH COMPLETE: $* ====="
