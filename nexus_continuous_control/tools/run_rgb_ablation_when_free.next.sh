#!/usr/bin/env bash
# Wait for the shared pool GPU to free up, then run the pixel-dependence ablation.
#
# The student-pool GPUs are first-come-first-served and usually busy. This waits
# for a stable block of free memory (two consecutive checks, so we do not race a
# job that is merely restarting), then runs the ablation for cartpole and cheetah
# in sequence. Safe to kill at any time: `pkill -f run_rgb_ablation_when_free`.
#
#   nohup bash tools/run_rgb_ablation_when_free.sh > ~/abl_watcher.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

NEED_MB=${NEED_MB:-6000}      # in-loop RGB at 128 envs; 256 envs OOMs an 11 GB card
# Poll fast. The gaps between other users' runs are SHORT -- a 120 s poll with a
# 2-check stability rule missed a real 10.8 GB window on 2026-08-17 because the
# next job started within two minutes. 15 s keeps the anti-race check while
# making the total confirmation delay 15 s instead of 2 min.
POLL_S=${POLL_S:-15}
UPDATES=${UPDATES:-250}
NUM_ENVS=${NUM_ENVS:-128}
EPISODES=${EPISODES:-5}

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_gpu() {
  local stable=0 need=${NEED_MB}
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$need" ]; then
      stable=$((stable + 1))
      log "free=${free}MB >= ${need}MB (stable ${stable}/2)"
      [ "$stable" -ge 2 ] && return 0
    else
      [ "$stable" -gt 0 ] && log "free=${free}MB dropped back below threshold"
      stable=0
    fi
    sleep "$POLL_S"
  done
}

run_one() {
  local tag=$1 config=$2 meta=$3
  # Idempotent: a finished stage is never redone, so the watcher can be
  # restarted (retuned, script edited, machine rebooted) without losing or
  # overwriting a result that already cost a 45-minute GPU window.
  if [ -f "results/rgb/ablation/${tag}/pixel_ablation.json" ]; then
    log "=== $tag: already done (results/rgb/ablation/${tag}/pixel_ablation.json) -- skipping ==="
    return 0
  fi
  # Retry: a GPU window is expensive to wait for, so a crashed attempt must not
  # silently skip the stage until someone restarts the watcher. Success is
  # defined by the RESULT FILE existing, not by the exit code (the script can
  # exit 0 after a pipeline swallows an error).
  local attempt need=$NEED_MB
  for attempt in 1 2 3; do
    log "=== $tag: attempt ${attempt}/3, waiting for >= ${need}MB free ==="
    NEED_MB=$need wait_for_gpu
    log "=== $tag: launching ($NUM_ENVS envs, $UPDATES updates) ==="
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export XLA_PYTHON_CLIENT_PREALLOCATE=false   # never grab the whole shared card
    # Unbuffered: with the default buffering nothing reaches the log until the
    # run ends, which defeats the point of the train/rgb/pixel_sensitivity
    # monitor -- a blind encoder should be visible (and killable) within minutes
    # instead of after a 45-minute job.
    export PYTHONUNBUFFERED=1
    python -m nexus_continuous.scripts.rgb_pixel_ablation \
      --config "$config" --meta "$meta" --seed 0 \
      --updates "$UPDATES" --num-envs "$NUM_ENVS" --episodes "$EPISODES" \
      --save-policy "runs/abl_${tag}.pkl" \
      --out "results/rgb/ablation/${tag}" > "runs/abl_${tag}.log" 2>&1
    local rc=$?
    if [ -f "results/rgb/ablation/${tag}/pixel_ablation.json" ]; then
      log "=== $tag: SUCCESS (exit ${rc}, log: runs/abl_${tag}.log) ==="
      return 0
    fi
    if grep -qi 'out of memory\|CUDA_ERROR_OUT_OF_MEMORY\|RESOURCE_EXHAUSTED' \
        "runs/abl_${tag}.log" 2>/dev/null; then
      need=$((need + 1500))   # it OOMed at this threshold -- demand more headroom
      log "=== $tag: attempt ${attempt} OOMed; raising requirement to ${need}MB ==="
    else
      log "=== $tag: attempt ${attempt} failed (exit ${rc}); retrying ==="
    fi
  done
  log "=== $tag: GIVING UP after 3 attempts (see runs/abl_${tag}.log) ==="
  return 1
}

run_one cartpole     configs/cartpole_balance_nesy_rgb.yaml     neural
run_one cheetah      configs/cheetah_run_neural.yaml            neural
# Does the fix work? Same ablation, but on a policy trained with the auxiliary
# pixel->state loss + a committed meta. Success = large drops under corruption.
run_one cartpole_aux configs/cartpole_balance_nesy_rgb_aux.yaml neural
log "watcher done"
