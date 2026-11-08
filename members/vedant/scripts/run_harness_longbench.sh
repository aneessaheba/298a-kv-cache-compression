#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
DATA=${TEAM6_DATA:-data/processed_longbench/processed_longbench_dev.jsonl}
test -s "$DATA" || { echo "Missing team dataset: $DATA" >&2; exit 1; }
# Smoke outputs stay separate; a changed subset must not contaminate full runs.
python harness/run_longbench.py --data "$DATA" --press turboquant_2.5bit --per-task 2 \
  --output-dir results/turboquant_smoke
# A successful exit can still include OOM rows. Fail before the full matrix if so.
python - <<'PY'
from pathlib import Path
import json
p=Path('results/turboquant_smoke/Llama-3.1-8B-Instruct_turboquant_2.5bit_0.0.jsonl')
rows=[json.loads(line) for line in p.open()]
assert len(rows)==32 and not any(row.get('oom') for row in rows), 'Smoke test incomplete or contains OOM records.'
PY
# Main mixed rates first, then uniform and QJL-off ablations.
for press in turboquant_2.5bit turboquant_3.5bit turboquant_mse_2.5bit turboquant_mse_3.5bit \
             turboquant_3bit turboquant_4bit turboquant_mse_3bit turboquant_mse_4bit turboquant_2bit; do
  python harness/run_longbench.py --data "$DATA" --press "$press"
done
