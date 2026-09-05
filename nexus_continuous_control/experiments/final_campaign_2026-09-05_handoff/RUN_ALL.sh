#!/usr/bin/env bash
# One explicit entry point for the fixed campaign; never invoked by installation.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
phases=(core initial resample llm_reference llm_pilot pilot_eval refine llm_final rgb curves probes shifts llm_eval media)
if [[ "${1:-}" != --execute ]]; then
  [[ $# == 0 ]] || { echo 'Usage: bash RUN_ALL.sh [--execute]'; exit 2; }
  printf 'Planned phase: %s\n' "${phases[@]}"
  echo 'No jobs started. Execute with: bash RUN_ALL.sh --execute'
  exit
fi
[[ $# == 1 ]] || { echo 'Unexpected arguments'; exit 2; }
[[ -f "$KIT/READY.json" ]] || { echo 'Missing verified release; nothing started'; exit 2; }
exec 9>/tmp/nexus_final_campaign_controller.lock
flock --nonblock 9 || { echo 'Another campaign controller is running'; exit 2; }
OUT=/mnt/d/nexus_final_campaign_2026-09-05
mkdir -p "$OUT/logs"
LOG="$OUT/logs/matrix_$(date -u +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; echo "STOP rc=$rc phase=${phase:-startup} UTC=$(date -u +%FT%TZ); completed/partial data were not deleted"; exit "$rc"' ERR
for phase in "${phases[@]}"; do
  echo "BEGIN phase=$phase UTC=$(date -u +%FT%TZ)"
  bash "$KIT/RUN_MATRIX.sh" "$phase" --execute
  echo "END phase=$phase UTC=$(date -u +%FT%TZ)"
  case "$phase" in
    core|llm_reference|llm_pilot|llm_final|rgb)
      if ! bash "$KIT/RUN_MATRIX.sh" backup --execute; then
        echo "WARNING: remote backup failed after $phase; local checkpoints are intact. Re-run the backup phase after reconnecting."
      fi
      ;;
  esac
done
phase=backup
bash "$KIT/RUN_MATRIX.sh" backup --execute
echo "ALL_DECLARED_PHASES_COMPLETED UTC=$(date -u +%FT%TZ)"
echo 'This indicates data collection completion, not successful learning or statistical significance.'
