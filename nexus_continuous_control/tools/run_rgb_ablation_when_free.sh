#!/usr/bin/env bash
# Full RGB ablation campaign on a shared pool GPU: for each configuration, train
# in-loop from pixels, ablate the camera, probe encoder responsiveness, and render
# the qualitative artifacts -- so every run leaves a COMPLETE result set:
#
#   pixel_ablation.json/.png    six-condition camera ablation + verdict
#   training_curves.json/.png   episode return, pixel sensitivity, aux loss
#   pixel_sensitivity.json      open-loop responsiveness + render health check
#   viz/                        rollout video, 64x64 filmstrip, skill timeline
#
# The pool GPUs are first-come-first-served, so this waits for a stable block of
# free memory before each stage. Kill with: pkill -f run_rgb_ablation_when_free
#
#   FORCE=1 nohup bash tools/run_rgb_ablation_when_free.sh > ~/abl_watcher.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1

NEED_MB=${NEED_MB:-6000}
POLL_S=${POLL_S:-15}          # short: gaps between other users' jobs are brief
UPDATES=${UPDATES:-250}
NUM_ENVS=${NUM_ENVS:-128}
EPISODES=${EPISODES:-5}
FORCE=${FORCE:-0}             # 1 = retrain even if a result already exists

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
  local dir="results/rgb/ablation/${tag}"
  if [ "$FORCE" != "1" ] && [ -f "${dir}/pixel_ablation.json" ]; then
    log "=== $tag: already done -- skipping (FORCE=1 to redo) ==="
    return 0
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  export XLA_PYTHON_CLIENT_PREALLOCATE=false   # never grab the whole shared card
  export PYTHONUNBUFFERED=1                    # stream progress into the log

  local attempt need=$NEED_MB
  for attempt in 1 2 3; do
    log "=== $tag: attempt ${attempt}/3, waiting for >= ${need}MB free ==="
    NEED_MB=$need wait_for_gpu

    log "=== $tag [1/3]: train + camera ablation ($NUM_ENVS envs, $UPDATES updates) ==="
    python -m nexus_continuous.scripts.rgb_pixel_ablation \
      --config "$config" --meta "$meta" --seed 0 \
      --updates "$UPDATES" --num-envs "$NUM_ENVS" --episodes "$EPISODES" \
      --save-policy "runs/abl_${tag}.pkl" --out "$dir" > "runs/abl_${tag}.log" 2>&1
    local rc=$?

    if [ -f "${dir}/pixel_ablation.json" ]; then
      log "=== $tag [2/3]: encoder responsiveness probe ==="
      python -m nexus_continuous.scripts.rgb_pixel_sensitivity \
        --config "$config" --meta "$meta" \
        --load-policy "runs/abl_${tag}.pkl" --out "$dir" \
        >> "runs/abl_${tag}.log" 2>&1
      log "=== $tag [3/3]: rollout video + filmstrip + skill timeline ==="
      python -m nexus_continuous.scripts.rgb_inloop_visualize \
        --config "$config" --meta "$meta" --seed 0 \
        --load-policy "runs/abl_${tag}.pkl" --out "${dir}/viz" \
        >> "runs/abl_${tag}.log" 2>&1
      log "=== $tag: SUCCESS (exit ${rc}, log: runs/abl_${tag}.log) ==="
      return 0
    fi

    if grep -qi 'out of memory\|CUDA_ERROR_OUT_OF_MEMORY\|RESOURCE_EXHAUSTED' \
        "runs/abl_${tag}.log" 2>/dev/null; then
      need=$((need + 1500))
      log "=== $tag: attempt ${attempt} OOMed; raising requirement to ${need}MB ==="
    else
      log "=== $tag: attempt ${attempt} failed (exit ${rc}); retrying ==="
    fi
  done
  log "=== $tag: GIVING UP after 3 attempts (see runs/abl_${tag}.log) ==="
  return 1
}

run_one cheetah      configs/cheetah_run_neural.yaml            neural
run_one cartpole     configs/cartpole_balance_nesy_rgb.yaml     neural
run_one cartpole_aux configs/cartpole_balance_nesy_rgb_aux.yaml neural
log "campaign done"
