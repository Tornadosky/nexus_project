#!/usr/bin/env bash
# SHARED-ENCODER / PIXEL-AWARE-META CAMPAIGN (2026-08-19). NOT COMMITTED.
#
# Generated from tools/run_rgb_ablation_when_free.sh -- the wait_for_gpu and
# run_one machinery below is that file's, unchanged. Only the arm list differs.
#
#   bash tools/run_shared_encoder_campaign.sh <tier>     tier = 0 | 1 | 2
#
# Arms are SERIAL (each needs ~6 GB on a shared 11 GiB RTX 2080 Ti; 256 envs
# OOMs this card, so NUM_ENVS stays at 128 and is never raised).
#
# Per-arm progress is appended to ~/shared_encoder_campaign.log so the campaign
# can be polled from outside.
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
  # tag: flat identifier used for the pkl/log filenames (must stay slash-free).
  # outdir: where results land, results/rgb/ablation/<env>/<meta>_<status>[_seedN].
  local tag=$1 outdir=$2 config=$3 meta=$4 seed=${5:-0}
  local dir="results/rgb/ablation/${outdir}"
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
      --config "$config" --meta "$meta" --seed "$seed" \
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
        --config "$config" --meta "$meta" --seed "$seed" \
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

# THE PROJECT'S FLAGSHIP META. NEXUS's selling point is the neuro-symbolic meta
# (learned meta-Q masked by hand-written skill preconditions), and every RGB
# config already declares META_POLICY_TYPE: nesy. The first campaign overrode
# that with --meta neural, so the in-loop arm never actually demonstrated the
# extension on the project's own configuration. These runs fix that; the earlier
# neural results are kept as a meta-variant comparison.

# ---------------------------------------------------------------------------
# Campaign bookkeeping: append a one-line result per arm to the poll log.
# ---------------------------------------------------------------------------
CAMPAIGN_LOG="$HOME/shared_encoder_campaign.log"

arm() {
  # arm <tag> <outdir> <config> <meta> <seed>
  local tag=$1 outdir=$2 config=$3 meta=$4 seed=${5:-0}
  local dir="results/rgb/ablation/${outdir}"
  if [ "${FORCE:-0}" != "1" ] && [ -f "${dir}/pixel_ablation.json" ]; then
    echo "  [skip] ${outdir} already has pixel_ablation.json" >> "$CAMPAIGN_LOG"
    return 0
  fi
  local t0 t1 mins
  t0=$(date +%s)
  echo "  [start $(date '+%F %T')] ${outdir}  (config=${config})" >> "$CAMPAIGN_LOG"
  if run_one "$tag" "$outdir" "$config" "$meta" "$seed"; then
    t1=$(date +%s); mins=$(( (t1 - t0) / 60 ))
    echo "  [ OK   $(date '+%F %T')] ${outdir}  wall=${mins}min" >> "$CAMPAIGN_LOG"
  else
    t1=$(date +%s); mins=$(( (t1 - t0) / 60 ))
    # Skip and continue: a single bad arm must not abort the campaign.
    echo "  [FAIL  $(date '+%F %T')] ${outdir}  wall=${mins}min -- see runs/abl_${tag}.log" >> "$CAMPAIGN_LOG"
    echo "        tail: $(tail -n 3 runs/abl_${tag}.log 2>/dev/null | tr '\n' ' ')" >> "$CAMPAIGN_LOG"
  fi
}

mkdir -p runs
TIER=${1:?usage: run_shared_encoder_campaign.sh <tier 0|1|2>}
echo "" >> "$CAMPAIGN_LOG"
echo "[TIER $TIER] launched $(date '+%F %T')" >> "$CAMPAIGN_LOG"

case "$TIER" in
0)
  # THE MISSING CONTROL: aux OFF at META_DECISION_INTERVAL=4. If these SEE, the
  # aux loss was never load-bearing and MDI alone fixed the blindness.
  arm walker_noaux_mdi4_s0   walker/nesy_noaux_mdi4_seed0 \
      configs/walker_walk_nesy_rgb_noaux.yaml        nesy 0
  arm cartpole_noaux_mdi4_s0 cartpole/nesy_noaux_mdi4_seed0 \
      configs/cartpole_balance_nesy_rgb_noaux.yaml   nesy 0
  ;;
1)
  # LOAD-BEARING: shared encoder vs the tier-0 control, one flag apart.
  arm walker_shared_noaux_mdi4_s0   walker/nesy_shared_noaux_mdi4_seed0 \
      configs/walker_walk_nesy_rgb_shared_noaux.yaml      nesy 0
  arm cartpole_shared_noaux_mdi4_s0 cartpole/nesy_shared_noaux_mdi4_seed0 \
      configs/cartpole_balance_nesy_rgb_shared_noaux.yaml nesy 0
  # "Sharing alone, shortcut intact": MDI stays 1, so a blind outcome here is
  # NOT evidence against sharing.
  arm walker_shared_s0   walker/nesy_shared_seed0 \
      configs/walker_walk_nesy_shared_only.yaml           nesy 0
  arm cartpole_shared_s0 cartpole/nesy_shared_seed0 \
      configs/cartpole_balance_nesy_rgb_shared_only.yaml  nesy 0
  ;;
2)
  # Meta sees pixels (requires shared).
  arm walker_shared_metaz_noaux_s0   walker/nesy_shared_metaz_noaux_seed0 \
      configs/walker_walk_nesy_rgb_shared_metaz_noaux.yaml      nesy 0
  arm cartpole_shared_metaz_noaux_s0 cartpole/nesy_shared_metaz_noaux_seed0 \
      configs/cartpole_balance_nesy_rgb_shared_metaz_noaux.yaml nesy 0
  ;;
3)
  # CONFIRMATION SEEDS for the headline arm (tier-2 cartpole was n=1).
  arm cartpole_shared_metaz_noaux_s1 cartpole/nesy_shared_metaz_noaux_seed1 \
      configs/cartpole_balance_nesy_rgb_shared_metaz_noaux.yaml nesy 1
  arm cartpole_shared_metaz_noaux_s2 cartpole/nesy_shared_metaz_noaux_seed2 \
      configs/cartpole_balance_nesy_rgb_shared_metaz_noaux.yaml nesy 2
  ;;
4)
  # CONFIRMATION SEEDS for the tier-0 control the headline is measured against.
  arm cartpole_noaux_mdi4_s1 cartpole/nesy_noaux_mdi4_seed1 \
      configs/cartpole_balance_nesy_rgb_noaux.yaml nesy 1
  arm cartpole_noaux_mdi4_s2 cartpole/nesy_noaux_mdi4_seed2 \
      configs/cartpole_balance_nesy_rgb_noaux.yaml nesy 2
  ;;
*)
  echo "unknown tier: $TIER" ; exit 2 ;;
esac

echo "[TIER $TIER] done $(date '+%F %T')" >> "$CAMPAIGN_LOG"
log "campaign tier $TIER done"
