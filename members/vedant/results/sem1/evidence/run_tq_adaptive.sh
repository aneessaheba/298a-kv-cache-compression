#!/usr/bin/env bash
set -euo pipefail
cd /app/298a-kv-cache-compression
source members/vedant/.venv/bin/activate
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

exec 9>/tmp/vedant_tq_adaptive.lock
flock -n 9 || { echo "Adaptive queue already running."; exit 1; }
trap 'echo "STOPPED: inspect the error above."' ERR

out=results/turboquant_adaptive_sem1
mkdir -p "$out"

python - <<'PY'
import shutil, torch
from pathlib import Path
from turboquant.protocol import validate_dataset
assert torch.cuda.is_available(), "CUDA unavailable"
assert len(validate_dataset(Path("data/processed_longbench/processed_longbench_dev.jsonl"))) == 800
assert shutil.disk_usage("/app").free > 8 * 1024**3, "Need 8 GiB free"
import matplotlib, safetensors, kvpress
print("Preflight passed", flush=True)
PY

python -m turboquant.splits \
  --input data/processed_longbench/processed_longbench_dev.jsonl \
  --output-dir "$out/splits"

python - "$out" <<'PY'
import json, sys
from pathlib import Path
import turboquant.protocol as protocol
p = Path(sys.argv[1]) / "splits/split_manifest.json"
frozen = Path(protocol.__file__).parent / "data/adaptive_excluded_ids.json"
assert set(json.loads(p.read_text())["record_ids"]) == set(json.loads(frozen.read_text())["record_ids"])
PY

reports=()
for sample in $(seq 0 15); do
  folder="$out/calibration/sample_$sample"
  tensor="$folder/calibration_kv.safetensors"
  mkdir -p "$folder"
  echo "CALIBRATION $((sample + 1))/16 $(date -Is)"

  python -m turboquant.extract_kv \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt-jsonl "$out/splits/adaptive_calibration.jsonl" \
    --record-index "$((sample * 2))" \
    --max-input-tokens 4096 \
    --shared-prompt --save-queries \
    --dtype bfloat16 --device cuda \
    --output "$tensor"

  python -m turboquant.real_kv "$tensor" \
    --all-layers --bits 2 3 4 --output-dir "$folder"

  for layer in $(seq 0 31); do
    test -s "$folder/layer_$layer/real_kv_report.json"
    reports+=(--report "$layer=$folder/layer_$layer/real_kv_report.json")
  done

  if (( sample > 0 )); then rm -- "$tensor"; fi
done

python -m turboquant.adaptive "${reports[@]}" \
  --reference-bits 3 --reference-qjl on --qjl-policy on \
  --output-dir "$out/allocation"

python -u harness/run_longbench.py \
  --press turboquant_adaptive \
  --turboquant-layer-config "$out/allocation/adaptive_allocation.json" \
  --output-dir "$out/longbench"

python - "$out" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
from turboquant.protocol import validate_dataset

p = Path(sys.argv[1])
data = validate_dataset(Path("data/processed_longbench/processed_longbench_dev.jsonl"))
f = p / "longbench/Llama-3.1-8B-Instruct_turboquant_adaptive_0.0.jsonl"
rows = [json.loads(line) for line in f.open()]
assert len(rows) == len({r["record_id"] for r in rows}) == 800
assert {r["record_id"]:r["task"] for r in rows} == {r["record_id"]:r["task"] for r in data}
assert not any(r.get("oom") or not r["prediction"].strip() for r in rows)

excluded = set(json.loads((p/"splits/split_manifest.json").read_text())["record_ids"])
heldout = [r for r in rows if r["record_id"] not in excluded]
assert len(heldout) == 768
assert set(Counter(r["task"] for r in heldout).values()) == {48}
(p/"heldout_768.jsonl").write_text("".join(json.dumps(r)+"\n" for r in heldout))
print("PASS: 800 predictions; 768 held-out; no OOM or blank predictions.", flush=True)
PY

python -m pip freeze > "$out/python_packages.txt"
nvidia-smi > "$out/nvidia_smi.txt"
cp SOURCE_COMMIT.txt "$out/SOURCE_COMMIT.txt"
tar -czf /app/turboquant_adaptive_results.tgz "$out"
echo "ADAPTIVE QUEUE FINISHED: /app/turboquant_adaptive_results.tgz"
