# Vedant Workstream Status

Updated: 2026-11-07

Source package version: 0.2.0. RTX 5090 model-level results remain pending.

## Completed locally

- Implemented deterministic Haar-random rotations using Gaussian QR decomposition.
- Implemented Lloyd--Max MSE codebooks for the exact unit-sphere coordinate density.
- Implemented TurboQuant-MSE and TurboQuant-Prod/QJL encode/decode paths.
- Implemented physical 1--8 bit packing and exact persistent-byte accounting.
- Implemented a Hugging Face `Cache` adapter that stores compressed keys and
  values and dequantizes them for ordinary attention.
- Implemented real-KV extraction and validation utilities.
- Implemented LongBench generation, resume support, metadata capture, and a
  paired TurboQuant/KIVI/full-precision comparison tool.
- Added the official LongBench v1 generation-length configuration.
- Passed the complete G3 numerical gate from the final workbook rubric.
- Verified the model/cache path end to end with a public tiny Llama checkpoint.
- Implemented and locally validated mixed per-layer cache configurations.
- Added an exact storage-constrained Adaptive TurboQuant allocator, sensitivity
  heatmap, task-stratified calibration isolation, context scaling workflow, and
  paired bootstrap confidence intervals.
- Frozen 32 calibration and 32 context-scaling records across all 16 tasks,
  leaving 768 records for held-out adaptive evaluation.
- Predeclared 16-prompt primary calibration pass, sampling every task once
  while conservatively excluding the complete 32-record pool from evaluation.

## G3 gate result

The authoritative output is in `results/g3/`. The run used 30 independent
seeds and 1,000 vectors per seed at dimensions 128 and 512.

| Dimension | Bits | MSE relative error vs numerical reference | Product mean signed error 95% CI | Result |
|---:|---:|---:|---:|:---:|
| 128 | 2 | 0.183% | [-0.000729, 0.000625] | PASS |
| 128 | 4 | 0.343% | [-0.000374, 0.000114] | PASS |
| 512 | 2 | 0.0035% | [-0.000542, 0.000340] | PASS |
| 512 | 4 | 0.0416% | [-0.000157, 0.000093] | PASS |

The maximum observed relative norm error after rotation was `8.882e-16`, well
inside the required `1e-5` limit. Deterministic scalar fixtures matched an
independent float64 reference exactly.

## Llama 3.1 8B persistent-cache forecast

Assumptions: 32 layers, 8 KV heads, head dimension 128, packed codes, and FP16
norm metadata. Percentages are relative to an FP16 key-plus-value cache and do
not claim peak-memory or latency improvement in the unfused reference path.

| Method | Bits | Bytes/token | % of FP16 | Compression |
|---|---:|---:|---:|---:|
| MSE keys + MSE values | 2 | 17,408 | 13.28% | 7.53x |
| MSE keys + MSE values | 3 | 25,600 | 19.53% | 5.12x |
| MSE keys + MSE values | 4 | 33,792 | 25.78% | 3.88x |
| QJL keys + MSE values | 2 | 17,920 | 13.67% | 7.31x |
| QJL keys + MSE values | 3 | 26,112 | 19.92% | 5.02x |
| QJL keys + MSE values | 4 | 34,304 | 26.17% | 3.82x |

## External evidence still required

These are external validation and experiment tasks still required before claiming model-level completion.

| Evidence | Owner/source | Why it cannot be produced on this Mac |
|---|---|---|
| Llama 3.1 8B real-KV G3 layer results | Vedant on university RTX 5090 | gated checkpoint and CUDA GPU required |
| LongBench full/TurboQuant predictions at 2, 3, and 4 bits | Vedant on RTX 5090 | the 8B long-context runs require CUDA memory and time |
| KIVI 2- and 4-bit prediction files | Manjot/KIVI environment | official KIVI exposes 2/4-bit cache settings, not a genuine 3-bit result |
| Official task scores | Included pinned scorer, executed after GPU prediction runs | no model predictions from RTX 5090 are available locally |
| Evict-then-Quantize integration result | Vedant + Manjot | depends on the eviction implementation and shared model runner |
| Adaptive 32-layer allocation and held-out result | Vedant on RTX 5090 | requires real Llama KV measurements and scored predictions |

The exact commands and handoff format are documented in
`docs/GPU_RUNBOOK.md`.

## Team protocol update

- Re-ran the local suite: 37 tests pass, including offline tiny-Llama generation,
  safe resume, official scoring, paired comparison and prompt exceptions.
- Added the pinned official LongBench scorer and declared whitespace policy.
- Preserved classification labels, raw predictions and prompt/token provenance.
- Default generation verifies the frozen 800-record dataset hash and task counts.
- Team comparisons enforce exact 800 IDs; adaptive comparisons enforce the frozen
  32-record exclusion and compare 768 paired records including a recalculated baseline.
- Packed KV payload, quantizer-state overhead and CUDA peak memory are distinct fields.
- Added Conda-free lab setup with actual CUDA kernel preflight and GitHub source-release instructions.
- Built both Python source distribution and wheel locally.
- The team's reported 47.72 baseline is not yet independently reproduced by this package.
- Shared Drive runner access was unavailable; newer shared harness settings still need cross-checking.

## Shared repository review

The shared harness uses BF16, separate document/question construction and protected
question tokens. The standalone reference path still needs this integration before
claiming cross-method parity. Shared scorer outputs are 0–1, while this package
stores 0–100. See TEAM_INTEGRATION.md for the checked differences.
