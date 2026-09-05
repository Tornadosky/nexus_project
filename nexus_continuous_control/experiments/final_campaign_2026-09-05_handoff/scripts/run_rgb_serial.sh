#!/bin/bash
# Explicit execution, not called by installation. One local process per GPU.
set -euo pipefail
: "${KIT:?Set installed package path}" "${PY:?Set verified local interpreter}"
mapfile -t ids < <("$PY" -c 'import json,sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1])) if r["group"]=="rgb"))' "$KIT/plan/matrix.json")
for id in "${ids[@]}"; do
  "$PY" "$KIT/scripts/agent.py" run --profile wsl_rgb --id "$id" --execute
done
