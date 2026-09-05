#!/usr/bin/env bash
# Explicit phase launcher. Installation never executes production training.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"; export KIT
PY=/mnt/c/Users/smirn/VSCodeProjects/nexus_project/nexus_continuous_control/.venv-wsl312/bin/python
RGB_PY=/home/smirn/nexus_campaign_rgb_venv/bin/python
LLM_PY=/home/smirn/nexus_campaign_llm_venv/bin/python
OUT=/mnt/d/nexus_final_campaign_2026-09-05
SPECS=/home/smirn/nexus_campaign_specs_2026-09-05
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=0
[[ -x "$PY" ]] || { echo "Run this release inside WSL Ubuntu on Tornadosky, not Viper"; exit 2; }
phase="${1:-status}"
if [[ "$phase" == status ]]; then
 "$PY" "$KIT/scripts/agent.py" status --profile wsl_core --group core
 "$RGB_PY" "$KIT/scripts/agent.py" status --profile wsl_rgb --group rgb
 exit
fi
[[ "${2:-}" == --execute ]] || { echo "Preview only. Execute with: bash $0 $phase --execute"; exit; }
[[ -f "$KIT/READY.json" ]] || { echo "Readiness verification is incomplete; nothing launched"; exit 2; }
(cd "$KIT" && sha256sum --quiet -c SHA256SUMS.txt)
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["production_ready"]' "$KIT/READY.json"
mkdir -p "$OUT/lists" "$OUT/logs" "$OUT/evidence"
case "$phase" in
 rgb)
  export PY="$RGB_PY"; unset XLA_FLAGS
  bash "$KIT/scripts/run_rgb_serial.sh"
  ;;
 core|llm_reference|llm_pilot|llm_final)
  if [[ "$phase" == llm_pilot || "$phase" == llm_final ]]; then
   "$PY" "$KIT/automation/select_executable.py" --group "$phase" --specs "$SPECS" --out "$OUT/lists"
   mapfile -t ids < "$OUT/lists/$phase.txt"
  else
   mapfile -t ids < <("$PY" -c 'import json,sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1])) if r["group"]==sys.argv[2]))' "$KIT/plan/matrix.json" "$phase")
  fi
  for id in "${ids[@]}"; do
   [[ -n "$id" ]] || continue
   "$PY" "$KIT/scripts/agent.py" run --profile wsl_core --id "$id" --execute
  done
  ;;
 initial|resample)
  "$LLM_PY" "$KIT/automation/generate_phase.py" "$phase"
  ;;
 refine)
  "$LLM_PY" "$KIT/automation/generate_phase.py" refined
  ;;
 curves|probes|shifts|pilot_eval|llm_eval)
  suite="$phase"; matrix="$KIT/plan/matrix.json"
  if [[ "$phase" == pilot_eval ]]; then
   suite=pilot; matrix="$OUT/lists/llm_pilot_evaluation.json"
  elif [[ "$phase" == llm_eval ]]; then
   suite=llm; matrix="$OUT/lists/llm_final_evaluation.json"
  fi
  if [[ "$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$matrix")" == 0 ]]; then
   echo "No executable policies in this recorded invalid-generation cohort; nothing substituted."
   exit
  fi
  export PYTHONPATH='' MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_FLAGS=--xla_gpu_enable_command_buffer=
  export JAX_COMPILATION_CACHE_DIR="$OUT/jax_cache"
  flock --nonblock /tmp/nexus_local_gpu.lock "$PY" "$KIT/scripts/evaluate_campaign.py" \
   --matrix "$matrix" --repo "$KIT/sources/core" --results "$OUT/results" \
   --out "$OUT/evidence" --suite "$suite"
  ;;
 media)
  export MUJOCO_GL=osmesa
  for task in hopper go1; do
   "$CORE" "$KIT/automation/render_media.py" --task "$task"
  done
  ;;
 backup)
  # Independent filesystem copy; no deletion and no original Viper data modified.
  DEST=/ptmp/akalenik/nexus/nvidia_final_campaign_2026-09-05_backup
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes viper11 "mkdir -p '$DEST/core' '$DEST/rgb' '$DEST/specs' '$DEST/package'"
  rsync -a --partial-dir=.rsync-partial "$OUT/" "viper11:$DEST/core/"
  rsync -a --partial-dir=.rsync-partial /home/smirn/nexus_campaign_verified_v2/rgb_release/ "viper11:$DEST/rgb/"
  rsync -a --partial-dir=.rsync-partial "$SPECS/" "viper11:$DEST/specs/"
  rsync -a --exclude=__pycache__/ "$KIT/" "viper11:$DEST/package/"
  ;;
 *)
  echo "Unknown phase: $phase"; exit 2
  ;;
esac
