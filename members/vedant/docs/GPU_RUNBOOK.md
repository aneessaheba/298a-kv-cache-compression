# RTX 5090 runbook

Run from the repository root. This package is locally verified; the lab GPU
installation and Llama 3.1 8B experiment results remain pending.

## 1 Install and check the environment

```bash
bash scripts/lab_setup.sh
source .venv/bin/activate
hf auth login
hf auth whoami
```

No Conda or sudo is required. If CUDA preflight fails, use the lab-approved
PyTorch build supporting the RTX 5090 and its driver, then rerun preflight.
The environment record is in `results/environment/`.

## 2 Check data and model access

```bash
DATA=/path/to/data/processed_longbench/processed_longbench_dev.jsonl
MODEL=meta-llama/Llama-3.1-8B-Instruct
turboquant-longbench --input "$DATA" --model "$MODEL" --method full \
  --device cuda --max-samples 1 --output results/smoke/full.jsonl
turboquant-score --input results/smoke/full.jsonl --output results/smoke/full_scored.jsonl
```

This smoke run verifies execution, not answer quality or the reported 47.72
baseline. Align model revision, dtype, prompts/token IDs, generation settings
and scoring policy with the team before treating scores as comparable.
The full matrix uses a fresh `results/longbench/` directory.

## 3 Numerical and real-KV validation

```bash
turboquant-g3 --output-dir results/g3
python -m pytest -q > results/pytest.txt
turboquant-extract-kv --model "$MODEL" --prompt "Summarize the evidence." \
  --output results/llama31_8b_kv.safetensors --dtype float16 --device cuda
for layer in 0 15 31; do
  turboquant-real-kv results/llama31_8b_kv.safetensors \
    --key-name "layer_${layer}.keys" --value-name "layer_${layer}.values" \
    --bits 2 3 4 --output-dir "results/real_kv/layer_${layer}"
done
```

Any failed numerical check blocks experimental claims.

## 4 Standard team comparison on all 800 records

```bash
bash scripts/run_r5_matrix.sh "$DATA" "$MODEL" results/longbench
bash scripts/score_and_compare.sh "$DATA" results/longbench results/report_v1
```

This includes full cache, TQ 2/3/4 bits and QJL-off 3/4 bits. Reports are in
`results/report_v1/team_800/`. A report must contain exactly the expected 800
IDs and tasks, use the same scorer, and have no duplicate or missing records.
Scoring refuses to overwrite files; use a new report directory for a rerun.
Generation resumes only with the original `*.run.json` manifest and settings.

## 5 Adaptive study on held-out 768 records

```bash
turboquant-split --input "$DATA" --output-dir configs/splits
bash scripts/run_adaptive_calibration.sh configs/splits/adaptive_calibration.jsonl \
  "$MODEL" results/adaptive_calibration 16
bash scripts/run_r5_matrix.sh "$DATA" "$MODEL" results/adaptive_matrix \
  results/adaptive_calibration/allocation/adaptive_allocation.json
bash scripts/score_and_compare.sh "$DATA" results/adaptive_matrix results/adaptive_report_v1
```

The standard 800 report excludes adaptive as an evaluated method. Adaptive is
reported only in `adaptive_768/`, excluding the frozen calibration IDs from
full cache and every method. The allocator targets uniform 3-bit QJL payload
storage. Inspect measured bytes before claiming equal memory. The total 800
records are 32 calibration plus 768 evaluation; the context subset is inside
those 768, not an additional disjoint split.

## 6 Context study

```bash
bash scripts/run_context_scaling.sh configs/splits/context_scaling.jsonl "$MODEL" \
  results/context_scaling results/adaptive_calibration/allocation/adaptive_allocation.json
```

For each output, run `turboquant-score --input RAW.jsonl --output SCORED.jsonl`.
Then supply all scored files to `turboquant-context-report`, for example
`--run full,4096=... --run turboquant_3bit,4096=... --run adaptive,4096=...`,
repeating for 8192, 16384 and 32000, followed by `--output-dir results/context_report`.

## 7 Other methods and memory budgets

Other method owners should preserve `record_id`, `task`, `prediction`, `answers`,
`all_classes`, model/revision, seed, dtype, prompt/token-ID hashes and their
storage definition. Use this package's scorer for all methods. Do not substitute
an aggregate 47.72 without the underlying compatible predictions and setup.
Compare additional scored runs with `turboquant-compare --run full=... --run
kivi_2bit=... --evaluation-set team --dataset "$DATA" --output-dir ...`.
Use `--evaluation-set adaptive` for the held-out 768 comparison.

Report all-layer KV payload bytes, quantizer-state overhead and peak CUDA memory
separately. Nominal 2-bit is not a 10% retention budget. Our reference path
reconstructs tensors; it is suitable for quality and packed-storage experiments,
and its measured runtime must be labelled as reference end-to-end runtime.
It does not establish fused-kernel production speedups.

## 8 Bring back

- `results/environment/`, command logs, errors and the exact source Git commit.
- Numerical gate, tests and real-KV reports.
- All `*.run.json` manifests and raw predictions, including failed-run logs.
- Scored predictions and `.summary.json` files, both 800 and 768 comparison reports.
- Adaptive allocation, calibration provenance and context-study inputs/reports.
- Exact model revision and dataset/split identities.

Keep large calibration tensors locally until the reports and allocation have
been copied successfully. Do not upload Hugging Face tokens or model weights.
Add actual GPU artifacts deliberately in a later GitHub results commit.
