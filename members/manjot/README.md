# RDKV (Manjot)

[RDKV](https://arxiv.org/abs/2605.08317) (Zhang, Guo, Benini, Li, 2026) gives every cached unit a bit-width from
{0, 2, 4, 8, 16}: 0 means evicted, 16 means kept in bf16. V is allocated per token and K per channel, by rate-distortion
bit allocation. No official code is public, so `rdkv_press.py` is our re-implementation as a kvpress press.

| File | What |
|---|---|
| `rdkv_press.py` | `RDKVPress`, registered as `--press rdkv` in `harness/run_longbench.py` |
| `test_rdkv.py` | CPU unit tests (allocator hits each budget, end-to-end on a tiny random Llama). `python -m pytest -q members/manjot/test_rdkv.py` |
| `compare.py` | Paired per-task comparison of any run against the full-cache baseline |
| `kaggle_rdkv.ipynb` | Kaggle 2xT4 notebook (fp16 side check only, not the main table): setup, tests, data rebuild, smoke run, fp16 check, full runs |
| `colab_rdkv.ipynb` | Colab A100 notebook for the reported results: clones this branch, pins `transformers==5.2.0`, bf16 on one GPU (the baseline's setup); results saved to Drive, resumes after disconnects |

## Running

```bash
# lab GPU (same settings as the baseline)
python harness/run_longbench.py --press rdkv --compression-ratio 0.9

# Kaggle 2xT4 (no bf16 on T4, model split evenly over both GPUs)
python harness/run_longbench.py --press rdkv --compression-ratio 0.9 --dtype float16 --device-map auto --max-gpu-mem 8.5
```
Team settings: all 800 records, no `--max-prompt-tokens`, ratios 0.75 / 0.8 / 0.9 / 0.95 (0.75 and 0.9 first).

`--dtype`, `--device-map`, `--max-gpu-mem` and `--out-tag` are new optional harness flags. Their defaults keep the old
behaviour, so existing commands and file names are unchanged.

**T4 note.** On GPUs older than Ampere, torch's SDPA has no fused kernel for grouped-query attention and silently falls
back to the math kernel, which builds the full fp32 attention matrix (37 GB at 17K tokens). The harness now expands
the KV heads first on those GPUs (`avoid_sdpa_gqa_on_old_gpus`) and sets `expandable_segments` for the CUDA allocator, so the memory-efficient kernel runs; the result is
mathematically identical, and it does nothing on the 5090/4090. Check: full-cache fp16 on 2x T4 reproduced the
bf16 baseline's scores exactly on the qasper records tested (0.061 and 0.605). It runs about 7-10x slower than
the 5090, so about 6 h per ratio.

## Memory accounting

RDKV's budget is in bits: each KV head gets (1 - ratio) x T x d x 16 bits for V and the same for K (the paper's r_K = 0.5),
so the payload matches the harness's `kv_cache_gb` formula. **For memory comparisons use `rdkv_kv_cache_gb`**: the context
cache *including* quantizer metadata (fp16 scale/zero-point per quantized token or channel, keep bitmaps) plus the
uncompressed question tokens, counted the same way as the baseline's `kv_cache_gb`. Other per-row fields:
`rdkv_context_gb` (context only, with metadata), `rdkv_context_gb_payload` (context only, without metadata) and
`rdkv_kept_fraction_bits` / `_tokens`.

## How the implementation works (for the report)
Once, at the end of prefill, per layer and KV head:
1. **Scores.** V: per-token attention mass from the last 32 queries, summed over the GQA group, AvgPool1d(5).
   K: per-channel `‖Q[:,c]‖·‖K[:,c]‖/√d`.
2. **V allocation over tokens.** Each token picks b ∈ {0,2,4,8,16} minimising Σ w_t·ε_t(b) under the V budget
   (Lagrangian relaxation + bisection on λ, ≤64 steps, 1 % tolerance). b=0 ⇒ evicted.
3. **K allocation over channels**, only over kept tokens, same solver. b=0 ⇒ channel removed.
Budget per head = (1−r)·T·d·16 bits for V and the same for K (paper's r_K = 0.5). New decode tokens stay FP16.

**Deviations from the paper (state them honestly):**
- ε_u(b) is the exact per-unit RMS quantisation error computed online, not the paper's offline calibrated table.
- The **values** of the last 32 context tokens are always kept at FP16 (SnapKV convention) and charged to the V budget. Their **keys** are not exempt: they go through the per-channel K allocation like every other kept token.
- No TriZone packing / Triton kernels: quantisation is simulated (quantise→dequantise) and eviction is masked, so
  physical GPU memory doesn't drop — same as KVzip/RestoreKV in our comparison; we report computed size.
- Uniform budget per head and layer, as in the paper.
