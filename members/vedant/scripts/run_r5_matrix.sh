#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 PROCESSED_LONGBENCH_DEV_JSONL MODEL_ID OUTPUT_DIR [ADAPTIVE_CONFIG]" >&2
  exit 2
fi

data_path=$1
model_id=$2
output_dir=$3
adaptive_config=${4:-}

mkdir -p "$output_dir"

common_args=(
  --input "$data_path"
  --model "$model_id"
  --device cuda
  --dtype float16
  --seed 42
  --max-input-tokens 32000
  --max-new-tokens-config configs/longbench_v1_max_new_tokens.json
)

turboquant-longbench \
  "${common_args[@]}" \
  --method full \
  --output "$output_dir/full.jsonl"

for bit_width in 2 3 4; do
  turboquant-longbench \
    "${common_args[@]}" \
    --method turboquant \
    --bits "$bit_width" \
    --output "$output_dir/turboquant_${bit_width}bit.jsonl"
done

if [[ -n "$adaptive_config" ]]; then
  turboquant-longbench \
    "${common_args[@]}" \
    --method adaptive \
    --adaptive-config "$adaptive_config" \
    --output "$output_dir/adaptive_turboquant.jsonl"
fi

for bit_width in 3 4; do
  turboquant-longbench \
    "${common_args[@]}" \
    --method polarquant \
    --bits "$bit_width" \
    --output "$output_dir/qjl_off_${bit_width}bit.jsonl"
done
