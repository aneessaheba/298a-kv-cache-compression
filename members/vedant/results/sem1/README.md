# Semester 1 TurboQuant experiment evidence

This release archives nine fixed settings and one adaptive experiment from the RTX 5090. The existing team full-cache baseline is referenced at `results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl`; it is not a new run by Vedant.

- [Verified tables and chart captions](analysis/RESULTS.md)
- [Machine-readable results and paired intervals](analysis/results.json)
- [Evidence checksums](analysis/checksums.json)
- [Workbook 2 contribution](../../docs/WORKBOOK_2_CONTRIBUTION.md)
- [GPU runtime/package record](evidence/python_packages.txt)
- [Adaptive execution log](evidence/turboquant_adaptive.log)
- [Exact adaptive queue](evidence/run_tq_adaptive.sh)

## What is stored

`fixed/` contains each setting's original prediction JSONL, summary and run manifest (27 files). `adaptive/longbench/` contains the equivalent three files for the adaptive experiment. Scores in raw predictions are 0–1; shared summaries are task-macro percentages. `adaptive/calibration/` contains metadata and all 512 per-layer diagnostic reports. `adaptive/allocation/` contains the chosen configuration, table and original sensitivity plot. `adaptive/split_manifest.json` records all 32 excluded calibration-pool IDs.

Large model weights, raw prompt datasets, tokens/credentials, the 1.6 GB tensor sample and multi-GB archives are not Git artifacts. The retained raw real-KV sample and complete original archives are backed up separately; their archive verification is included. Each diagnostic file records the quantizer seed and source tensor shape. These reports are real-Llama measurements, not the earlier synthetic G3 fixture.

## Frozen experiment protocol

- Source commit: `82d40f72eb8c3a8313c4da95d2dbe22913aaad7a` (not the newer reporting commit).
- Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Python 3.11.13; Torch 2.7.1+cu128; Transformers 5.2.0; kvpress 0.5.5; tokenizers 0.22.2; rapidfuzz 3.14.6; rouge 1.0.1.
- Frozen development file: 800 unique records, 16 tasks × 50; SHA256 `fc9d8649f535619c6b3dbe31ad158b112131d85230f92cd253c2e32bb1b7a47c`.
- BF16 model weights and attention; document-only cache compression. Question and generated tokens are protected. Original document K/V is used for its own prefill attention.
- Task-specific official-style metrics in the team's shared scorer, including its first-line handling and rapidfuzz code similarity. Task-macro scores are not a single classification accuracy.
- Logical packed-cache accounting includes codes, norm metadata, mixed-group selectors and protected question tokens. Actual peak GPU allocation is reported separately.

## Procedure used in the lab

1. Preserve the tested container Torch using the system-site-packages venv. Record GPU, packages and the source commit; verify CUDA/BF16 and authenticate privately for the gated model.
2. Verify the frozen dataset hash and all 800 unique IDs. Run the 32-record smoke separately from full outputs.
3. Run the shared harness for QJL keys at 2, 2.5, 3, 3.5 and 4 bits, then MSE keys at 2.5, 3, 3.5 and 4 bits. Keep every prediction, adjacent run manifest and summary, including poor scores.
4. Build the deterministic 32-record calibration pool (two per task, seed 20260). Use every second record: 16 calibration samples, one per task. Capture real post-RoPE queries and BF16 K/V with the shared prompt format, retaining at most 4,096 head/tail tokens.
5. Evaluate 2/3/4-bit QJL-on/off reconstruction per sample and layer; allocate layer rates under the uniform 3-bit QJL packed budget. All selected layers were 3-bit QJL.
6. Evaluate that configuration with the full 800-record shared harness. Exclude all 32 calibration-pool IDs from every compared method during analysis: 768 held-out records, 48 per task. No training or fine-tuning was performed.
7. Validate IDs, failures, data/code fingerprints and scores; preserve logs and environment; back up archives. Results are now available for CPU-only analysis.

## Regenerate validation, comparisons and figures

From the repository root in the existing analysis or lab environment, with NumPy, Matplotlib, rapidfuzz 3.14.6 and rouge 1.0.1 available:

```bash
python members/vedant/scripts/report_sem1_results.py \
  --data data/processed_longbench/processed_longbench_dev.jsonl
```

This does not generate new model predictions. It verifies all 8,000 TurboQuant and 800 baseline scores, regenerates five figures in PNG/SVG, checks calibration allocation, and computes 2,000-draw paired task-stratified bootstrap intervals (seed 20260). It verifies the scorer matches the archived version; use the recorded source version if a later main changes scoring. Older run fingerprints remain tied to the source used in the lab.

## Interpretation

The adaptive experiment is a null allocation result: every layer remains at 3-bit QJL. Differences from the earlier uniform run are repeated-generation differences, not adaptive improvement. CI overlap with zero does not establish equivalence. Dev-set comparisons do not prove generalization to unseen test examples. The reference cache retains decoded BF16; neither the computed cache reduction nor the elapsed example timings demonstrate a fused low-memory serving implementation.
