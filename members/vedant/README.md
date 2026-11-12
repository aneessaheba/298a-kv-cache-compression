# TurboQuant — Team 6 reference implementation

Version 0.3.0. The reference implementation and Semester 1 RTX 5090 experiments are complete: nine fixed settings (7,200 predictions), one adaptive experiment (800 predictions), and 512 real-KV calibration layer reports. All recorded predictions independently re-score without mismatches. See [verified results and charts](results/sem1/analysis/RESULTS.md), [Workbook 2 contribution](docs/WORKBOOK_2_CONTRIBUTION.md), and [evidence/procedure](results/sem1/README.md).

The adaptive allocator selected 3-bit QJL in all 32 layers; no adaptive quality improvement is established. Packed-cache size is computed; reconstructed BF16 remains resident, so these experiments do not demonstrate physical VRAM savings or serving speedups.

## Canonical team run

Use the repository's `harness/run_longbench.py`, BF16, `meta-llama/Llama-3.1-8B-Instruct`, and the frozen 800-record development file (16 tasks, 50 each). Its SHA256 is `fc9d8649f535619c6b3dbe31ad158b112131d85230f92cd253c2e32bb1b7a47c`. The checked-in shared baseline contains all 800 records and scores **47.72** after task macro averaging; it is the team's result, not a TurboQuant result.

From the repository root, inside the team's CUDA container:

```bash
bash members/vedant/scripts/lab_setup.sh
source members/vedant/.venv/bin/activate
hf auth login
bash members/vedant/scripts/run_harness_longbench.sh
```

The script first tests 32 records in a separate smoke directory and stops if any OOM occurs. It then prioritizes `turboquant_2.5bit` and `turboquant_3.5bit`, followed by mixed QJL-off and uniform ablations. To run only one setting:

```bash
python harness/run_longbench.py --press turboquant_2.5bit
```

Setup preserves the container's Torch, Transformers and kvpress versions with `--system-site-packages` and constraints. The target image is Python 3.11.13, Torch 2.7.1+cu128, Transformers 5.2.0, kvpress 0.5.5. See [lab runbook](docs/GPU_RUNBOOK.md).

## What the press does

The document's prefill attention uses the original BF16 keys and values. After each layer's document forward, the press packs and reconstructs that document's K/V. Questions and generated answers remain uncompressed under the shared pipeline. No document unpack occurs at each decoding step.

Uniform settings use TurboQuant-MSE values and either TurboQuant-Prod/QJL keys or MSE keys. Fractional 2.5/3.5-bit settings select the top half of original channels by document-only squared magnitude independently per head for K and V. Two independent subgroup quantizers assign 3/2 or 4/3 bits. This is an explicitly defined implementation of the paper's outlier-group strategy, **not a claim of reproducing its underspecified allocation exactly**. The paper's printed `(32×3+96×2)/128` is 2.25, despite being labelled 2.5. Our 64/64 split gives the stated nominal rate exactly. See [TurboQuant §4.3](https://arxiv.org/html/2504.19874v1).

This shared protocol protects generated tokens; the paper also quantizes generated tokens. Label this a team-protocol adaptation when reporting results.

## Memory and speed

The kvpress path retains decoded BF16 tensors. `kv_cache_gb` is a **computed compressed-representation size**, counting packed codes, subgroup norm metadata, channel permutations, and protected question tokens. It is not allocated CUDA memory. `peak_gpu_gb` is measured separately. Dense rotations/codebooks are implementation state outside the logical cache metric, as are model weights. Do not claim VRAM savings or accelerated decoding from this reference path.

For Llama's head dimension 128, before selectors and protected tokens, QJL K/V representation costs are 17,920 / 23,552 / 26,112 / 31,744 / 34,304 bytes per document token across all layers at 2 / 2.5 / 3 / 3.5 / 4 bits. Fractional settings need two sets of norm fields. Compare actual recorded memory with the team's eviction runs rather than treating bit width as an exactly equal budget.

## Scores and comparisons

The canonical scorer is `harness/longbench.py` (rapidfuzz, **0–1 per record**, leading whitespace stripped before first-line tasks). The shared runner supplies the same document/question split and task-specific chat-template exceptions. Do not re-score canonical results with the standalone scorer: its historical fuzzywuzzy code metric differs.

```bash
python members/vedant/scripts/compare_team_runs.py \
  --data data/processed_longbench/processed_longbench_dev.jsonl \
  --run full=results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl \
  --run tq25=results/longbench/Llama-3.1-8B-Instruct_turboquant_2.5bit_0.0.jsonl \
  --output-dir results/turboquant_comparison
```

This validates all 800 IDs/tasks, explicitly converts scores to percentage points for paired, task-stratified bootstrap confidence intervals, retains OOM failures, and reports computed memory separately. Other methods are optional additional `--run NAME=FILE` arguments; no unproduced KIVI file is required. A changed data/code/runtime/allocation fingerprint prevents unsafe resume. Keep the adjacent `.run.json` files with predictions.

## Adaptive extension and local checks

[Adaptive calibration](docs/ADAPTIVE_TURBOQUANT.md) uses real post-RoPE queries, shared prompt formatting, bounded 4K head/tail windows and a dimensionless normalized reconstruction objective. It is experimental; the bounded window and QK proxy limitations are explicit. Its layer allocation also runs through the shared press.

The earlier isolated local check passed 70 tests with Transformers 5.2.0 and kvpress 0.5.5 on CPU/Mac (Torch 2.14.1, Python 3.12). These cover cache construction, exact prefill logits, no repeated prompt unpack, mixed rates/accounting, actual shared pipeline behavior, adaptive configuration and resume/paired reporting. Those tiny local tests are distinct from the now archived full 8B GPU experiment evidence. Older synthetic G3 artifacts remain synthetic numerical evidence only.

Read [fix audit](docs/REVIEW_FIXES.md) for review-item status. Historical standalone scripts remain diagnostic tools; use the shared runner for team comparisons.
