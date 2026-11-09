#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 ADAPTIVE_CALIBRATION_JSONL MODEL_ID OUTPUT_DIR [CALIBRATION_SAMPLES]" >&2
  exit 2
fi

calibration_path=$1
model_id=$2
output_dir=$3
sample_count=${4:-16}
record_count=$(wc -l < "$calibration_path")

if (( sample_count < 1 || sample_count > record_count )); then
  echo "CALIBRATION_SAMPLES must be between 1 and $record_count" >&2
  exit 2
fi

mkdir -p "$output_dir/layers"

report_args=()
for sample in $(seq 0 $((sample_count - 1))); do
  record_index=$((sample * record_count / sample_count))
  sample_dir="$output_dir/sample_${sample}"
  tensor_path="$sample_dir/calibration_kv.safetensors"
  mkdir -p "$sample_dir"
  turboquant-extract-kv \
    --model "$model_id" \
    --prompt-jsonl "$calibration_path" \
    --record-index "$record_index" \
    --max-input-tokens 4096 \
    --shared-prompt --save-queries \
    --output "$tensor_path" \
    --dtype bfloat16 \
    --device cuda
  turboquant-real-kv "$tensor_path" --all-layers --bits 2 3 4 --output-dir "$sample_dir"
  for report_path in "$sample_dir"/layer_*/real_kv_report.json; do
    layer_name=$(basename "$(dirname "$report_path")")
    layer=${layer_name#layer_}
    report_args+=(--report "$layer=$report_path")
  done
done

turboquant-adaptive \
  "${report_args[@]}" \
  --reference-bits 3 \
  --reference-qjl on \
  --qjl-policy on \
  --output-dir "$output_dir/allocation"
