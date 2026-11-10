# Vedant — TurboQuant progress and reporting text

Implemented TurboQuant-MSE and the QJL residual product quantizer with bit packing, norm metadata, numerical validation and a Transformers cache. Integrated them as a kvpress press in the team's shared BF16 LongBench pipeline. Prompt prefill attention uses original K/V; only stored document K/V is quantized. Question and answer tokens remain protected to match the team protocol.

Added exact nominal 2.5/3.5-bit settings through independent subgroup quantizers, selecting half the original channels by per-head document energy. The paper motivates outlier groups but its printed channel example is arithmetically inconsistent; our allocation is specified explicitly as an implementation choice, not an exact reproduction of that recipe.

Completed local behavioral tests under Transformers 5.2.0, including identical prefill logits and no repeated prompt unpack. Added run fingerprints, exact 800-record data validation, a shared-score comparison adapter and task-stratified paired bootstrap intervals. The team's saved full-cache baseline is 47.72 on all 800 records; it is not evidence of completed TurboQuant experiments.

The adaptive extension normalizes key and value reconstruction error before budget allocation. Calibration captures real post-RoPE queries and shared-format prompts within a 4K head/tail window. These bounded measurements are proxies; GPU calibration, allocation validation and held-out evaluation remain pending.

The reference pipeline retains reconstructed BF16 tensors. Report computed compressed-cache bytes (codes, norm fields, channel selectors and protected question bytes) separately from measured peak GPU allocation and latency. Dense quantizer state is additional implementation overhead. Do not describe this source release as a deployed low-memory serving kernel or claim GPU performance gains before measurement.

Remaining deliverables: RTX 5090 smoke; complete primary 2.5/3.5-bit runs and ablations; retrieve raw predictions/manifests/environment/logs; compare against the canonical shared baseline and available team method results at observed memory budgets. Optional adaptive/context results require separate execution and explicit calibration exclusions. No new real-model score is reported until those artifacts exist.
