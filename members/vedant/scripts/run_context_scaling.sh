#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 CONTEXT_SUBSET_JSONL MODEL_ID OUTPUT_DIR [ADAPTIVE_CONFIG]" >&2
  exit 2
fi

data_path=$1
model_id=$2
output_dir=$3
adaptive_config=${4:-}

mkdir -p "$output_dir"

for context_length in 4096 8192 16384 32000; do
  common_args=(
    --input "$data_path"
    --dataset-scope subset
    --model "$model_id"
    --device cuda
    --dtype float16
    --seed 42
    --max-input-tokens "$context_length"
    --max-new-tokens-config configs/longbench_v1_max_new_tokens.json
  )
  turboquant-longbench \
    "${common_args[@]}" \
    --method full \
    --output "$output_dir/full_${context_length}.jsonl"
  turboquant-longbench \
    "${common_args[@]}" \
    --method turboquant \
    --bits 3 \
    --output "$output_dir/turboquant_3bit_${context_length}.jsonl"
  if [[ -n "$adaptive_config" ]]; then
    turboquant-longbench \
      "${common_args[@]}" \
      --method adaptive \
      --adaptive-config "$adaptive_config" \
      --output "$output_dir/adaptive_${context_length}.jsonl"
  fi
done
