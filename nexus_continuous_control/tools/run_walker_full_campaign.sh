#!/usr/bin/env bash
# WalkerWalk at the paper's FULL budget: 6400 updates x 128 envs x 64 steps
# = 52,428,800 environment steps, 25.6x the matched-budget Campaign 3.
#
# WHY WALKER, WHY NOW
# -------------------
# At 2.05M steps walker was the ONE env where state+RGB beat state-only:
# training return 189.51 +/- 1.80 -> 203.20 +/- 6.25 (+7.2%, 3/3 seeds,
# non-overlapping, Welch p = 0.054), 30-episode eval 0.713 -> 0.816 (+14.4%),
# and the only env where the actor demonstrably used its camera (frozen_first
# costs 62-72%). A +7% edge at 4% of the paper's budget can be an optimisation
# artefact -- the camera acting as a regulariser or an extra gradient path that
# a longer run makes redundant. This campaign settles that.
#
# ISOLATION FROM THE COMMITTED CAMPAIGN
# -------------------------------------
# Outputs go to results/rgb/state_plus_rgb_full/ and policies to ~/runs_spr_full/.
# The committed 2.05M tree (results/rgb/state_plus_rgb/, ~/runs_spr/) is never
# written to, so both budgets stay on disk side by side and comparable.
#
# NUM_ENVS IS FIXED AT 128 AND IS NOT A TUNABLE: 256 OOMs the in-loop renderer
# on the shared 11 GB 2080 Ti. Both arms use it, so it cannot confound.
#
# Usage: run_walker_full_campaign.sh ARM [ARM ...]   with ARM = "<arm>:<seed>",
#   e.g.  run_walker_full_campaign.sh state_matched_full:0 state_plus_rgb_full:0
# Arms run SERIALLY and the GPU is yielded between them.
# NB: no `set -u` -- ~/mesa_env.sh appends to an unset LD_LIBRARY_PATH.
cd ~/nexus_project/nexus_continuous_control
source .venv/bin/activate
source ~/mesa_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false

LOG=~/walker_full_campaign.log
OUTROOT=results/rgb/state_plus_rgb_full
POLROOT=~/runs_spr_full
mkdir -p "$POLROOT"

ENV=walker
CFGSTEM=walker_walk_nesy
UPDATES=6400           # 128 x 64 x 6400 = 52,428,800 env steps
NUM_ENVS=128           # 256 OOMs the in-loop renderer on the shared 2080 Ti
EPISODES=5             # the high-power 30-episode re-score is a separate pass
NEED_FREE=6000         # MiB
HEARTBEAT=600          # s between PROGRESS lines while an arm trains

log() { echo "$*" | tee -a "$LOG"; }

wait_for_gpu() {
  # The 2080 Ti is shared with other students. Yield rather than compete.
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
    if [ $((waited % 300)) -eq 0 ]; then
      log "[$(date -Is)]     WAITING for shared GPU: ${free} MiB free, need ${NEED_FREE} (waited ${waited}s)"
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

# A ~10h arm must be pollable from outside. The trainer prints
# "step=<env_step> env_r=... noise=..." every PRINT_EVERY=409600 env steps
# (= every 50 updates) into the run log; this lifts the newest one into the
# campaign log with an ETA so progress is visible without tailing run.log.
heartbeat() {
  local runlog="$1" tag="$2" t0="$3" total_steps="$4"
  while true; do
    sleep "$HEARTBEAT"
    [ -f "$runlog" ] || continue
    local line now elapsed step pct eta
    line=$(grep '^step=' "$runlog" | tail -1)
    now=$(date +%s); elapsed=$((now - t0))
    if [ -n "$line" ]; then
      step=$(echo "$line" | sed 's/^step=\([0-9]*\).*/\1/')
      if [ "$step" -gt 0 ] 2>/dev/null; then
        pct=$((100 * step / total_steps))
        eta=$(( elapsed * (total_steps - step) / step / 60 ))
        log "[$(date -Is)]     PROGRESS ${tag}: ${line} | ${pct}% of ${total_steps} | elapsed $((elapsed/60))m | ETA ~${eta}m"
        continue
      fi
    fi
    log "[$(date -Is)]     PROGRESS ${tag}: no step line yet (compiling?) | elapsed $((elapsed/60))m"
  done
}

TOTAL_STEPS=$((UPDATES * NUM_ENVS * 64))

for spec in "$@"; do
  IFS=: read -r ARM SEED <<< "$spec"
  CFG="configs/${CFGSTEM}_${ARM}.yaml"
  TAG="${ENV}/${ARM}_seed${SEED}"
  OUT="${OUTROOT}/${TAG}"
  RUNLOG="${OUT}/run.log"
  POL="${POLROOT}/${ENV}_${ARM}_s${SEED}.pkl"
  # `--no-rgb` is redundant with RGB_ACTOR: false in the config (the script ORs
  # them) but is passed for exact parity with the 2.05M campaign invocation.
  EXTRA=""
  case "$ARM" in state_matched*) EXTRA="--no-rgb" ;; esac

  if [ ! -f "$CFG" ]; then log "[$(date -Is)] MISSING CONFIG $CFG -- skipping $TAG"; continue; fi
  if [ -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] SKIP ${TAG} (pixel_ablation.json already exists)"
    continue
  fi

  mkdir -p "$OUT"
  wait_for_gpu
  log "[$(date -Is)] START ${TAG}"
  log "[$(date -Is)]     config ${CFG} ${EXTRA} | ${UPDATES} upd x ${NUM_ENVS} envs x 64 steps = ${TOTAL_STEPS} steps | episodes ${EPISODES} | seed ${SEED}"
  log "[$(date -Is)]     policy -> ${POL}   (a crash or eviction must never cost a retrain)"
  T0=$(date +%s)

  heartbeat "$RUNLOG" "$TAG" "$T0" "$TOTAL_STEPS" &
  HB_PID=$!

  # -u: stdout is a file here, so without it the jax.debug.callback progress
  # lines sit in a 4 KiB buffer for hours and the heartbeat has nothing to read.
  python -u -m nexus_continuous.scripts.rgb_pixel_ablation $EXTRA \
      --config "$CFG" --meta nesy --seed "$SEED" \
      --updates "$UPDATES" --num-envs "$NUM_ENVS" --episodes "$EPISODES" \
      --save-policy "$POL" \
      --out "$OUT" > "$RUNLOG" 2>&1
  RC=$?
  kill "$HB_PID" 2>/dev/null; wait "$HB_PID" 2>/dev/null
  T1=$(date +%s)
  WALL=$((T1 - T0))

  if [ $RC -ne 0 ] || [ ! -f "${OUT}/pixel_ablation.json" ]; then
    log "[$(date -Is)] FAILED ${TAG} rc=${RC} after ${WALL}s ($((WALL/60))m) -- tail of ${RUNLOG}:"
    grep -v '^Module ' "$RUNLOG" | tail -25 | sed 's/^/      /' | tee -a "$LOG"
    continue
  fi

  log "[$(date -Is)] DONE ${TAG} in ${WALL}s ($((WALL/60))m = $(awk "BEGIN{printf \"%.2f\", $WALL/3600}")h)"
  python - "$OUT/pixel_ablation.json" "$OUT/training_curves.json" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
i = d["results"]["intact"]
k = d.get("metric_key", "reward_per_step_mean")
print(f"    intact {k} = {i[k]:.4f} +/- {i.get(k.replace('_mean','_std'), 0.0):.4f}"
      f"  (n={len(i.get('per_episode', []))} episodes)")
print(f"    final_train_return = {d.get('final_train_return')}")
# The PRIMARY metric: last-20-update mean of the 128-env training curve. The
# 5-episode eval above has ~0.16 power and misled this project once already.
try:
    c = json.load(open(sys.argv[2]))
    curve = (c.get("curves") or c).get("episode_return")
    if curve:
        last20 = curve[-20:]
        print(f"    PRIMARY train_return last-20-update mean = "
              f"{sum(last20)/len(last20):.3f}   (curve len {len(curve)})")
except Exception as e:
    print(f"    (training_curves.json unreadable: {e})")
print(f"    actor_input = {d.get('actor_input')}  state_only = {d.get('state_only')}")
if not d.get("state_only"):
    print("    camera conditions (per-condition drop, NOT the binarised median): "
          + ", ".join(f"{c}={100*v:.1f}%" for c, v in d["performance_drop_fraction"].items()))
PY
  sleep 20   # let the GPU settle and give other users a window between arms
done

log "[$(date -Is)] ===== BATCH COMPLETE: $* ====="
