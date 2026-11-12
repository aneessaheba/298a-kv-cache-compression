# Full team-review audit — 2026-11-08

Audited against the supplied Drive `vedant_turboquant_fixes.md` (all 18 items), the pasted team messages, `turboquant_fixed.zip`, the Drive lab guide and README_FIRST.txt, current shared harness and other members' presses. The source archive was integrated selectively. This branch includes current `aneessaheba` main (`fccf60b`) and the referenced `samruddhichitnis02` main (`b1ba8a1`), verified against both remotes during the audit.

| Supplied note item | Implementation and verification |
|---|---|
| 1. Transformers 5.2 | Cache implements get_max_cache_shape and tensor/int mask API; >=5.2,<6 declared. Full suite runs specifically with 5.2.0. |
| 2. Prefill precision | Original current K/V returned; stored future K/V quantized. Tiny Llama prefill logits match DynamicCache exactly. |
| 3. Repeated unpack | Decoded chunks reused. A 4K prompt plus three decode tokens observes unpack lengths [4096,1,1,1]; retained copy equals full dequantization. |
| 4. Shared harness | All ten uniform/mixed/QJL variants plus adaptive are tested through an actual tiny BF16 kvpress pipeline. Compression occurs only for the document. Shared CLI tests writing, OOM failure, summary and strict resume. |
| 5. Keep lab Torch | System-site venv, runtime constraints, editable install without dependency replacement; actual lab execution is now archived under results/sem1. |
| 6. BF16 defaults | Canonical runner, extractor and experiment scripts use BF16. FP16 vector norm metadata is intentional representation overhead. |
| 7. Correct documentation | 800-record / 47.72 baseline and shared rapidfuzz scorer are canonical; historical standalone fuzzywuzzy policy is separated. |
| 8. Fractional settings | Independent outlier/nonoutlier subgroup quantizers at 2.5/3.5 bits; selectors and both subgroup norm fields counted. Main matrix starts with fractional rates. Exact allocation interpretation is disclosed below. |
| 9. Normalize adaptive objective | Both K/V terms are dimensionless normalized reconstruction error; regression test prevents absolute key error dominating. |
| 10. Real queries | Extract actual post-RoPE Llama queries; preserve GQA heads. All-layer calibration requires saved queries rather than silently using random queries. |
| 11. Prompt/window | Use shared prompt formatting and a longer 4K head/tail window. Bounded-window and position-aligned QK proxy limitations are stated. |
| 12. Representative extraction | Calibration extracts actual dev JSONL records; the five-token runbook example is removed. |
| 13. No nonexistent KIVI files | Shared comparison adapter accepts existing team outputs; no mandatory separate KIVI file. Explicitly converts 0–1 scores for percentage-point paired CIs. |
| 14. Report method weaknesses honestly | Uniform and QJL-off ablations retained. Peer-reported QJL/KIVI error numbers are not presented as reproduced local results. GPU reports must retain unfavorable outcomes. |
| 15. Behavioral/scale tests | 5.2 API, exact prefill, decode reuse, BF16 pipeline, score-scale/policy rejection (including unlabelled rows), packed-rate accounting, query capture, adaptive/CLI/resume tests pass. |
| 16. Standalone OOM recovery | Legacy diagnostic runner now records OOM as failed, continues to next record, preserves resume and reports unknown memory as null. Failure scoring/reporting regression tested. Shared production runner already handles OOM. |
| 17. Calibration process count | All layers are evaluated from one tensor load per sample, instead of starting 32 processes per sample. |
| 18. Name QJL-off correctly | Canonical standalone name is turboquant_mse; scripts use it. The old polarquant spelling remains only a compatibility alias for archived commands. |

## Verified local evidence

- **70 tests passed**, no skipped tests, in 10.15 seconds on Mac/CPU using Python 3.12, Torch 2.14.1, Transformers 5.2.0 and kvpress 0.5.5. One warning concerns the historical standalone fuzzywuzzy implementation's speed, not canonical scoring.
- The downloaded `longbench_dev_data.zip` passed ZIP CRC validation; its dataset is byte-identical to the frozen file tested locally.
- All 800 supplied development records match the frozen SHA256; 16 tasks have exactly 50 records each. Every document/question split and no-chat answer-cue split reconstructs its original prompt.
- All 800 saved baseline predictions re-score with zero mismatches; shared macro score **47.72**. This is scoring verification, not new 8B model generation.
- Paired adapter baseline-versus-itself check gives zero difference and [0,0] confidence interval. It is a reporting check, not a model experiment.
- Shell syntax and Python compilation pass. Source distribution and wheel builds pass.
- Main matrix contains ten bit/QJL settings, including 2-bit QJL-off; smoke outputs stay separate from full-run outputs.

## Qualifications and completed lab evidence

**Fractional allocation:** the paper's section 4.3 prints `(32×3+96×2)/128=2.5`, but that expression is 2.25. It does not specify an unambiguous 3.5-bit split or exact selection algorithm. Our documented strategy ranks original channels by document-only energy and assigns the top 64/128 channels the higher bit width. This gives exact 2.5/3.5 nominal rates; it is an implementation interpretation, not an exact reproduction of an unspecified recipe. Source: https://arxiv.org/html/2504.19874v1#S4.SS3

**Protocol adaptation:** the shared team protocol protects question and generated tokens. The paper quantizes generated tokens as well. Our reported results must be labelled as shared-protocol adaptation rather than literal reproduction of every paper experiment.

**Physical memory:** the press and reference cache retain decoded BF16 buffers. Logical packed bytes count codes, norm fields, selectors and protected question overhead; dense quantizer state is separate. These do not establish physical VRAM savings or faster serving.

**Lab evidence now complete:** actual Python 3.11.13 / Torch 2.7.1+cu128 / Transformers 5.2.0 / kvpress 0.5.5 RTX 5090 execution, nine full fixed runs, adaptive calibration/evaluation, recorded per-example time and measured peak allocation. Source commit: `82d40f72eb8c3a8313c4da95d2dbe22913aaad7a`. All 8,000 TurboQuant predictions re-score correctly. The adaptive result is a null allocation outcome, not an improvement. [Results and verification](../results/sem1/analysis/RESULTS.md). The tenth available fixed variant (uniform 2-bit QJL-off) and separate context/RULER/EtQ studies were not executed for this release.

## Input provenance

- `turboquant_fixed.zip` SHA256: `c78d8c2fdd5d64bee822b6076c76754327b0264af1d58f4cbec0ed3e50dc7428`
- `vedant_turboquant_fixes.md` SHA256: `7058e2f87b503da1809f9c7cbf8c0cc3d0d1ded4c7d54c52f95093b36afe4e7e`
- `longbench_dev_data.zip` SHA256: `1c4bed362fa81b284a37aa17704155c0859c8197d5077887dc435c578e29986d`
