# TurboQuant Semester 1 status — 2026-11-11

## Completed and archived

- Reference quantizers, packed representation, protected BF16 prefill, shared kvpress integration, mixed-channel settings, QJL ablations, strict resume and paired comparisons.
- Actual RTX 5090 execution using Llama-3.1-8B-Instruct, BF16, Torch 2.7.1+cu128, Transformers 5.2.0 and kvpress 0.5.5.
- Nine fixed settings × 800 records = 7,200 predictions; one adaptive experiment × 800 = 800 more predictions.
- Sixteen real-query calibration samples × 32 layers = 512 reports, each evaluating 2/3/4-bit QJL-on/off candidates.
- Independent verification of all 8,000 predictions plus the team's 800 baseline predictions: exact IDs/tasks, no OOM or blank outputs, canonical re-scoring, run/data/source hashes and summaries.
- CPU-only regeneration script, paired bootstrap comparisons, five figures (PNG/SVG), experiment procedure and Workbook 2 contribution text.

[Results](results/sem1/analysis/RESULTS.md) · [Evidence and reproduction](results/sem1/README.md) · [Workbook contribution](docs/WORKBOOK_2_CONTRIBUTION.md)

## Findings and limits

The strongest observed fixed means are MSE-key 3-bit (47.90) and MSE-key 3.5-bit (47.84), versus the team's full-cache 47.72. Small mean differences are not evidence of superiority. QJL 2-bit degrades strongly (14.94); QJL 3.5/4-bit score 47.25/47.67. Retain unfavorable results.

Adaptive calibration selected 3-bit QJL for all 32 layers. There was zero calibration-objective reduction and no demonstrated adaptive improvement. Compare all methods on the same 768 held-out IDs after excluding the entire 32-record calibration pool.

The reference implementation retains decoded BF16 cache tensors. Packed representation accounting is not measured physical VRAM reduction. Recorded example times include the pipeline execution and are not isolated decoding throughput.

These are frozen development-set experiments, not full original LongBench test evaluation. The two mixed rates are explicitly documented project adaptations of channel allocation.

## Scope boundaries

The planned nine-setting GPU matrix and adaptive/real-KV experiment are complete. Uniform 2-bit QJL-off, a distinct TurboQuant RULER/context sweep, pure KIVI same-tensor diagnostics, evict-then-quantize integration, fused kernels and production serving are not completed in this evidence release. Do not claim those are done. Broader team Workbook 2 requirements and final semester assessment remain separate from this contribution.
