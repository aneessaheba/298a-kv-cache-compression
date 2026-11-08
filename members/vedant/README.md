# Team 6 TurboQuant

**Team repository integration:** read [TEAM_INTEGRATION.md](TEAM_INTEGRATION.md)
before running a cross-method comparison. The shared harness uses BF16 and
protects question tokens; this standalone reference path still needs that integration.

Vedant's TurboQuant reference implementation and reproducible Team 6 experiment
package. Version 0.2.0 prepares source code for GitHub; Llama 3.1 8B RTX 5090
quality, peak-memory and performance results remain pending. No expected score
or speedup is presented as a measured result.

## Implemented

- Deterministic Haar rotations, sphere Lloyd–Max codebooks, MSE and Product/QJL quantizers.
- Physical 1–8 bit packing, norms, a Transformers cache adapter and numerical G3 gate.
- Full-cache and TurboQuant 2/3/4-bit runs, QJL-off ablations, adaptive layer allocation.
- Frozen Team 6 dataset identity checks, preprocessed prompts and chat exceptions.
- Pinned official LongBench metrics with the documented leading-whitespace fix.
- Safe resume manifests, prompt/token hashes, paired comparison and bootstrap intervals.
- Separate 800-record team and 768-record held-out adaptive reports.

## Local installation

Python 3.11 or newer. Use a virtual environment; Conda is not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf,plots,score]'
python -m pytest -q
```

For the RTX 5090 use `bash scripts/lab_setup.sh`. It captures the environment
and executes a CUDA matrix operation to check kernel support. Lab drivers and
the installed PyTorch CUDA build must be compatible; Mac tests cannot prove it.
Run `hf auth login` interactively. Never commit access tokens or model weights.

## Team data

Obtain `processed_longbench_dev.jsonl` from the team. It contains 800 records
(16 tasks × 50), SHA-256:
`fc9d8649f535619c6b3dbe31ad158b112131d85230f92cd253c2e32bb1b7a47c`.
It is a subset of the team's deduplicated 3,731-record LongBench variant.
This is a modified benchmark population, not the entire official benchmark.

The normal runner rejects any other file before loading a model. It uses each
record's `prompt`, preserves `all_classes` and answers, and applies the chat
template except for `trec`, `triviaqa`, `samsum`, `lcc`, and `repobench-p`.
Subset smoke/context runs require explicit `--dataset-scope subset`.

```bash
# Set this to your copy; the group's path is one possible location.
DATA=/path/to/data/processed_longbench/processed_longbench_dev.jsonl
# Check a single prompt before spending time on the matrix.
turboquant-longbench --input "$DATA" --model meta-llama/Llama-3.1-8B-Instruct \
  --method full --device cuda --max-samples 1 --output results/smoke/full.jsonl
```

Use a different output directory for smoke tests with changed settings. Resume
rejects changes to input identity, code, model revision, prompts/template settings,
method, bits, seed, dtype, attention backend and generation limits. Preserve
`*.run.json` beside predictions. Corrupt or legacy files are rejected, not repaired.

## Run and score

```bash
bash scripts/run_r5_matrix.sh "$DATA" meta-llama/Llama-3.1-8B-Instruct results/longbench
bash scripts/score_and_compare.sh "$DATA" results/longbench results/report_v1
```

The standard report includes all 800 records. If an adaptive run is present,
a second report excludes the frozen 32 calibration IDs from **every** method,
including full cache, leaving 768 paired records. Never compare an adaptive
768-record score directly with the group's reported 800-record baseline 47.72.
Reproduce the baseline with identical settings or recompute its paired subset.

The scorer uses official English LongBench v1 metric functions at a pinned
revision; first-line tasks use `lstrip()` before extracting the first line.
Per-record scores are 0–100 and averaged per task, then equally across tasks.
Every output records its scorer policy. See `THIRD_PARTY_NOTICES.md`.

## Adaptive and context studies

```bash
turboquant-split --input "$DATA" --output-dir configs/splits
bash scripts/run_adaptive_calibration.sh configs/splits/adaptive_calibration.jsonl \
  meta-llama/Llama-3.1-8B-Instruct results/adaptive_calibration 16
bash scripts/run_r5_matrix.sh "$DATA" meta-llama/Llama-3.1-8B-Instruct results/adaptive_matrix \
  results/adaptive_calibration/allocation/adaptive_allocation.json
bash scripts/score_and_compare.sh "$DATA" results/adaptive_matrix results/adaptive_report_v1
```

Thirty-two records are calibration-only. The 32 context-study records are a
subset of the 768 evaluation records. Calibration uses 16 prompts, one per task,
and conservatively excludes all 32 calibration-pool IDs. Generate split JSONL
locally; GitHub contains the frozen exclusion IDs, not dataset text.
See `docs/GPU_RUNBOOK.md` for context studies and collection instructions.

## Interpret memory and timing correctly

All layers are compressed. `measured_kv_bytes` counts actual packed codes plus
norm metadata; `quantizer_state_bytes` reports rotations/projections/codebooks
separately. The sum is also recorded. No full-precision layers are hidden.

Nominal bits are not retention percentages. At Llama 3.1 8B FP16, QJL 2/3/4-bit
payload is approximately 13.7%/19.9%/26.2% of full KV. Use actual per-token byte
measurements for memory budgets; the comparison reports rate-quality points
and does not automatically assert equal memory. Methods retaining 10% of tokens
are not necessarily comparable to any of these settings at equal memory.

The reference adapter reconstructs KV for stock attention. Packed storage
savings do not prove peak CUDA memory or latency gains. `tokens_per_second` is
end-to-end generated tokens divided by prefill-plus-decode time, not isolated
decode throughput. Peak allocated CUDA bytes are recorded separately.

## Source and results releases

Source release: code, tests, configs, runbooks, scorer notices and local G3 evidence.
Data, credentials, model weights, generated predictions, tensors and build output
are ignored by Git. Later GPU results should include environment, commands,
paired raw/scored predictions, manifests and measured reports. Check artifacts
before intentionally adding them; do not force-add the entire results directory.

## Primary sources

- [TurboQuant](https://arxiv.org/abs/2504.19874)
- [Official LongBench](https://github.com/THUDM/LongBench)
- [KIVI](https://github.com/jy-yuan/KIVI)

The shared GitHub harness has now been inspected. Dataset and chat exceptions
align; precision, prompt construction, protected question tokens and scorer-scale
integration remain explicit follow-up work in `TEAM_INTEGRATION.md`.
