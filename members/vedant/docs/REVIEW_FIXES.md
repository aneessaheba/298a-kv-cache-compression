# Review fixes and evidence — 2026-11-08

The shared Drive `turboquant_fixed.zip` and `vedant_turboquant_fixes.md` were inspected and selectively integrated, preserving other members' changes from current team main. The Drive lab guide was read; its older fork, run matrix and source overlay are superseded by GPU_RUNBOOK.md in this branch.

| Review item | Source change and evidence |
|---|---|
| 1. Transformers 5.2 | Version range >=5.2,<6; cache mask tensor/int API and get_max_cache_shape. Suite tested against 5.2.0. |
| 2. Prefill precision | Cache returns original current K/V; kvpress modifies stored document after attention. Tiny Llama prefill logits equal DynamicCache exactly. |
| 3. Repeated unpack | Retain decoded cache; behavioral counter on 4K prefill plus three decode tokens observes unpack lengths [4096,1,1,1]. Shared press compresses once per layer/document. Dense BF16 remains resident; no speedup claim. |
| 4. Shared harness | TurboQuant press registered; actual tiny kvpress pipeline checks document-only compression, question/decode protection; shared prompts/scorer retained. |
| 5. Torch preservation | System-site venv, runtime constraints, editable install without dependency replacement; CUDA check deferred to lab. |
| 6. BF16 | Canonical runner, extractor and experiment scripts default/select BF16. FP16 norm metadata is intentional representation overhead. |
| 7. Docs/baseline | Saved team baseline is 800/47.72; canonical shared rapidfuzz 0–1 scoring distinguished from historical standalone fuzzywuzzy. |
| 8. Mixed rates | Independent outlier/nonoutlier subgroup quantizers at 2.5/3.5; top 64/128 channels higher precision. Actual bytes, selectors and chunk invariance tested. Paper allocation ambiguity disclosed. |
| 9. Adaptive objective | Both K/V terms normalized; scale-domination regression test. |
| 10. Calibration | Real post-RoPE Q, actual shared formatting, 4K head/tail window; bounded-window and attention-proxy limits stated. |
| 11. KIVI files | Team-run comparison accepts actual shared method outputs; no mandatory unproduced KIVI files. |
| 12. Behavioral tests | Prefill logits, no repeated unpack, shared pipeline, real-Q hooks, mixed payload accounting and shared factory/adaptive tests added. |

Additional safeguards: exact dataset hash, code/runtime/allocation resume fingerprints, separate smoke outputs, OOM visibility and explicit score-scale conversion. Adaptive layer allocations use the same shared press.

Local result: 56 tests passed using Python 3.12, Mac Torch 2.14.1, Transformers 5.2.0 and kvpress 0.5.5 in an isolated dependency target. The target lab stack uses Python 3.11.13 / Torch 2.7.1+cu128 and must still run setup/smoke. No RTX 5090 timing, memory, 8B quality or adaptive gain is asserted.

Re-scored all 800 saved baseline predictions with the current shared scorer: zero score mismatches, macro 47.72. The paired adapter baseline-versus-itself check returned zero difference and [0, 0] confidence interval; this is a reporting validation, not a new model experiment. Calibration regression also checks the press's exact seeds and rounded stored reconstruction.
