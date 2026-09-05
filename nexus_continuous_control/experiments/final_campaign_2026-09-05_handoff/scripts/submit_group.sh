#!/bin/bash
# Explicit, bounded Slurm submission. Defaults to printing, not submission.
set -euo pipefail
: "${KIT:?Set installed Viper package path}"
group="${1:?core, llm_reference, llm_pilot, or llm_final}"
shift
case "$group" in core|llm_reference|llm_pilot|llm_final) ;; *) echo "Unknown group"; exit 2;; esac
smoke=0; submit=0
for arg in "$@"; do
 case "$arg" in --smoke) smoke=1;; --submit) submit=1;; *) echo "Unknown argument: $arg"; exit 2;; esac
done
OUT=/ptmp/akalenik/nexus/final_campaign_2026-09-05_verified
suffix=""; [[ "$smoke" == 0 ]] || suffix=_smoke
RUN_LIST="$OUT/lists/${group}${suffix}.txt"
if [[ "$smoke" == 0 && ( "$group" == llm_pilot || "$group" == llm_final ) ]]; then
 /ptmp/akalenik/jaxrocm_venv/bin/python "$KIT/automation/select_executable.py" --group "$group" --specs "$OUT/specs" --out "$OUT/selected"
 RUN_LIST="$OUT/selected/$group.txt"
fi
[[ -f "$RUN_LIST" ]] || { echo "Missing job list: $RUN_LIST"; exit 2; }
n="$(wc -l < "$RUN_LIST")"
(( n > 0 )) || { echo "No executable proposals; recorded failures are retained"; exit 0; }
mkdir -p "$OUT/logs"
export CAMPAIGN_ROOT="$KIT" RUN_LIST SMOKE="$smoke"
limit=24:00:00; [[ "$smoke" == 0 ]] || limit=01:00:00
cmd=(sbatch --time="$limit" --array="0-$((n-1))%8" --export=ALL "$KIT/scripts/viper.sbatch")
printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$submit" == 1 ]]; then "${cmd[@]}"; fi
