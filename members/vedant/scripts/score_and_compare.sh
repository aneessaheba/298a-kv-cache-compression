#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 TEAM_DEV_JSONL RAW_RUN_DIR REPORT_DIR" >&2
  exit 2
fi
data_path=$1
raw_dir=$2
report_dir=$3
mkdir -p "$report_dir/scored"
team_args=()
adaptive_args=()
for name in full turboquant_2bit turboquant_3bit turboquant_4bit qjl_off_3bit qjl_off_4bit adaptive_turboquant; do
  if [[ ! -s "$raw_dir/$name.jsonl" ]]; then
    if [[ "$name" == adaptive_turboquant ]]; then continue; fi
    echo "Missing required prediction file: $raw_dir/$name.jsonl" >&2
    exit 1
  fi
  scored="$report_dir/scored/$name.jsonl"
  # Scoring refuses overwriting prior files, so a partial/stale report cannot pass silently.
  turboquant-score --input "$raw_dir/$name.jsonl" --output "$scored"
  adaptive_args+=(--run "$name=$scored")
  if [[ "$name" != adaptive_turboquant ]]; then
    team_args+=(--run "$name=$scored")
  fi
done
turboquant-compare "${team_args[@]}" --evaluation-set team --dataset "$data_path" --output-dir "$report_dir/team_800"
if [[ -s "$raw_dir/adaptive_turboquant.jsonl" ]]; then
  turboquant-compare "${adaptive_args[@]}" --evaluation-set adaptive --dataset "$data_path" --output-dir "$report_dir/adaptive_768"
fi
