#!/bin/bash
# Generates specifications only. Never submits RL training; no implicit environment installation.
set -euo pipefail
: "${KIT:?Installed package}" "${LLM_PY:?Separate verified Torch/Transformers interpreter}"
: "${SPECS:?New specification directory}" "${EVIDENCE:?Directory containing pilot validation evidence}"
phase="${1:?Choose lock, initial, resample, or refine}"
# CPU-only generation has its own mutex and does not block the RGB GPU.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
exec 9>"${TMPDIR:-/tmp}/nexus_llm_generation.lock"
flock -n 9 || { echo "LLM generation lock held; no generation started"; exit 2; }
mkdir -p "$SPECS"
lock="$SPECS/model_lock.json"
if [[ "$phase" == lock ]]; then
  "$LLM_PY" "$KIT/scripts/llm_specs.py" lock-model --out "$lock"
  exit
fi
[[ "$phase" == initial || "$phase" == resample || "$phase" == refine ]] || exit 2
condition="$phase"
[[ "$condition" != refine ]] || condition=refined
for task in cheetah walker; do
  for family in 0 1 2; do
    out="$SPECS/${task}/g${family}/${condition}.json"
    args=("$KIT/scripts/llm_specs.py" generate --model-lock "$lock" --task "$task" --family "$family" --condition "$condition" --out "$out")
    if [[ "$condition" == refined ]]; then
      # Look up the exact pilot ID from the locked matrix, not a guessed naming convention.
      pilot="$("$LLM_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(next(x["id"] for x in r if x["group"]=="llm_pilot" and x["task"]==sys.argv[2] and x["proposal_family"]==int(sys.argv[3])))' "$KIT/plan/matrix.json" "$task" "$family")"
      args+=(--initial "$SPECS/${task}/g${family}/initial.json" --feedback "$EVIDENCE/pilot/$pilot/validation/summary.json")
    fi
    "$LLM_PY" "${args[@]}"
  done
done
