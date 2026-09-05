#!/usr/bin/env bash
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"
O=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified
PY=/ptmp/akalenik/jaxrocm_venv/bin/python
suite="${1:?curves, probes, shifts, pilot, llm}"
case "$suite" in
 curves|probes|shifts) matrix="$KIT/plan/matrix.json"; limit=8;;
 pilot) matrix="$O/selected/llm_pilot_evaluation.json"; limit=6;;
 llm) matrix="$O/selected/llm_final_evaluation.json"; limit=8;;
 *) echo 'Unknown evaluation suite'; exit 2;;
esac
n="$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$matrix")"
mkdir -p "$O/evidence/$suite" "$O/logs"
if (( n == 0 )); then
 echo 'No executable proposals; recorded generation failures are the result.'
 exit 0
fi
(( n < limit )) || n="$limit"
export CAMPAIGN_ROOT="$KIT" EVAL_SUITE="$suite" EVAL_SHARDS="$n" MATRIX_JSON="$matrix"
sbatch --array="0-$((n-1))%8" --export=ALL "$KIT/scripts/eval_viper.sbatch"
