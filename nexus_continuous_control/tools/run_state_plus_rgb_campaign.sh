#!/usr/bin/env bash
# state-only vs state+RGB at matched budget. Runs arms SERIALLY, yielding the
# shared GPU between arms. Usage: run_state_plus_rgb_campaign.sh ARM [ARM ...]
# where ARM is "<env>:<arm>:<seed>", e.g. "cartpole:state_matched:0".
# NB: no `set -u` -- ~/mesa_env.sh appends to an unset LD_LIBRARY_PATH and would
# abort the whole campaign before the first arm starts.
cd ~/nexus_project/nexus_continuous_control
source .venv/bin/activate
source ~/mesa_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false

LOG=~/state_plus_rgb_campaign.log
OUTROOT=results/rgb/state_plus_rgb
POLROOT=~/runs_spr
mkdir -p "$POLROOT"

MATCHED_UPDATES=250
FULL_UPDATES=6400      # 128 x 64 x 6400 = 52,428,800 env steps
NUM_ENVS=128           # 256 OOMs the in-loop renderer on the shared 2080 Ti
EPISODES=5
NEED_FREE=6000         # MiB

log() { echo "$*" | tee -a "$LOG"; }

wait_for_gpu() {
  # Yield to other users of the shared 2080 Ti.
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
    if [ $((waited % 300)) -eq 0 ]; then
      log "    waiting for GPU: only ${free} MiB free, need ${NEED_FREE} (waited ${waited}s)"
    fi
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
  RUNLOG="${OUT}/run.log"
  # `--no-rgb` keeps the vision ENVIRONMENT and removes only the actor's camera.
  EXTRA=""
  case "$ARM" in state_matched*) EXTRA="--no-rgb" ;; esac
  case "$ARM" in *_full) UPDATES=$FULL_UPDATES ;; *) UPDATES=$MATCHED_UPDATES ;; esac

  if [ ! -f "$CFG" ]; then log "[$(date -Is)] MISSING CONFIG $CFG -- skipping $TAG"; continue; fi

  if [ -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] SKIP ${TAG} (pixel_ablation.json already exists)"
    continue
  fi

  mkdir -p "$OUT"
  wait_for_gpu
  log "[$(date -Is)] START ${TAG}"
  log "    config ${CFG} ${EXTRA} | ${UPDATES} upd x ${NUM_ENVS} envs x 64 steps = $((UPDATES*NUM_ENVS*64)) steps | episodes ${EPISODES} | seed ${SEED}"
  T0=$(date +%s)
  python -m nexus_continuous.scripts.rgb_pixel_ablation $EXTRA \
      --config "$CFG" --meta nesy --seed "$SEED" \
      --updates "$UPDATES" --num-envs "$NUM_ENVS" --episodes "$EPISODES" \
      --save-policy "${POLROOT}/${ENV}_${ARM}_s${SEED}.pkl" \
      --out "$OUT" > "$RUNLOG" 2>&1
  RC=$?
  T1=$(date +%s)
  WALL=$((T1 - T0))

  if [ $RC -ne 0 ] || [ ! -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] FAILED ${TAG} rc=${RC} after ${WALL}s -- tail of ${RUNLOG}:"
    grep -v '^Module ' "$RUNLOG" | tail -25 | sed 's/^/      /' | tee -a "$LOG"
    continue
  fi

  log "[$(date -Is)] DONE ${TAG} in ${WALL}s ($((WALL/60))m)"
  python - "$OUT/pixel_ablation.json" "$WALL" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
k = d.get("metric_key") or ("upright_fraction_mean"
    if "upright_fraction_mean" in d["results"]["intact"] else "reward_per_step_mean")
i = d["results"]["intact"]
print(f"    intact {k} = {i[k]:.4f} +/- {i.get(k.replace('_mean','_std'), 0.0):.4f}")
print(f"    intact reward_per_step = {i['reward_per_step_mean']:.4f} "
      f"+/- {i['reward_per_step_std']:.4f}   (episode return over 250 steps "
      f"= {i['reward_per_step_mean']*250:.2f})")
print(f"    final_train_return = {d.get('final_train_return')}")
print(f"    actor_input = {d.get('actor_input')}  state_only = {d.get('state_only')}")
if not d.get("state_only"):
    print(f"    pixel drops: " + ", ".join(
        f"{c}={100*v:.1f}%" for c, v in d["performance_drop_fraction"].items()))
    print(f"    pixel_drop_median = {100*d['pixel_drop_median']:.1f}%  "
          f"-> actor_uses_pixels = {d['actor_uses_pixels']}")
PY
  # Let the GPU settle / give other users a window between arms.
  sleep 20
done

log "[$(date -Is)] CAMPAIGN BATCH COMPLETE: $*"
