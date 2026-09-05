#!/usr/bin/env bash
# Explicit production entry point. Installation never invokes this file.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
export KIT
V=/ptmp/akalenik/nexus/final_campaign_2026-09-05_handoff
O=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified
export PY=/home/smirn/nexus_campaign_rgb_venv/bin/python
export LLM_PY=/home/smirn/nexus_campaign_llm_venv/bin/python
export SPECS=/home/smirn/nexus_campaign_specs_2026-09-05
export EVIDENCE=/home/smirn/nexus_campaign_evidence_2026-09-05
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
phase="${1:-status}"
if [[ "$phase" == status ]]; then
 "$PY" "$KIT/scripts/agent.py" status --profile wsl_rgb --group rgb
 ssh -o BatchMode=yes -o ConnectTimeout=10 viper11 "squeue --me; find '$O/results_release' -name COMPLETE.json | wc -l"
 exit
fi
[[ "${2:-}" == --execute ]] || { echo "Dry run only. To execute: bash $0 $phase --execute"; exit; }
[[ -f "$KIT/READY.json" ]] || { echo "Release verification has not finished; no production launched"; exit 2; }
(cd "$KIT" && sha256sum --quiet -c SHA256SUMS.txt)
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["production_ready"]' "$KIT/READY.json"
case "$phase" in
 rgb)
  unset XLA_FLAGS
  bash "$KIT/scripts/run_rgb_serial.sh"
  ;;
 core|llm_reference|llm_pilot|llm_final)
  if [[ "$phase" == llm_pilot || "$phase" == llm_final ]]; then
   rsync -a "$SPECS/" "viper11:$O/specs/"
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=10 viper11 "export KIT='$V'; /ptmp/akalenik/jaxrocm_venv/bin/python '$V/automation/ensure_lists.py' --out '$O/lists'; bash '$V/scripts/submit_group.sh' '$phase' --submit"
  ;;
 initial|resample)
  "$LLM_PY" "$KIT/automation/generate_phase.py" "$phase"
  ;;
 refine)
  mkdir -p "$EVIDENCE/pilot"
  rsync -a "viper11:$O/evidence/pilot/" "$EVIDENCE/pilot/"
  "$LLM_PY" "$KIT/automation/generate_phase.py" refined
  ;;
 curves|probes|shifts|pilot_eval|llm_eval)
  suite="$phase"
  [[ "$phase" != pilot_eval ]] || suite=pilot
  [[ "$phase" != llm_eval ]] || suite=llm
  ssh -o BatchMode=yes -o ConnectTimeout=10 viper11 "bash '$V/automation/submit_evaluation.sh' '$suite'"
  ;;
 *)
  echo "Unknown phase: $phase"; exit 2
  ;;
esac
