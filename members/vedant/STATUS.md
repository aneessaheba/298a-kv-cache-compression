# Source and evidence status

## Complete locally

- Transformers >=5.2,<6 declared; behavioral checks run specifically with 5.2.0.
- Original K/V used for current prefill attention; only future stored cache quantized.
- Decoded cache reused; new chunks decoded once instead of repeatedly unpacking the prompt.
- Shared kvpress harness registration, BF16, protected questions/answers and canonical scoring.
- Lab setup inherits and constrains the installed Torch/runtime.
- Explicit outlier-aware 2.5/3.5-bit settings and QJL-off/uniform ablations.
- Normalized adaptive objective, real query capture, shared-format bounded calibration and shared adaptive press.
- Dataset hash checks, fingerprinted resume, paired confidence intervals and team-run adapter.
- 56 local tests passed with Transformers 5.2.0 / kvpress 0.5.5; exact lab CUDA stack not executed.

## Required on RTX 5090

1. Environment/CUDA smoke test, model access and 32-record pipeline smoke.
2. Complete 800-record 2.5/3.5-bit runs; remaining uniform/QJL ablations as time allows.
3. Retrieve predictions, manifests, summaries, environment and log; compute paired comparisons against saved baseline.
4. Run optional adaptive calibration/evaluation and context study if claiming that extension experimentally.

No local test completes these GPU deliverables. No real-model TurboQuant score, speedup or CUDA memory saving is claimed. The reference press retains decoded BF16; logical compressed size and measured peak allocation are different metrics.

## Representation forecast (Llama-3.1-8B)

Uniform QJL 2/3/4-bit document bytes per token across 32 layers: 17,920 / 26,112 / 34,304. Mixed QJL 2.5/3.5-bit: 23,552 / 31,744, plus 131,072 bytes of stored channel permutations per document. All add full-precision question bytes. These are computed representation sizes, not measured GPU savings.
