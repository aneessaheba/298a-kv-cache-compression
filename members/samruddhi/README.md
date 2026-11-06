# TriAttention on LongBench

KV-cache compression with **TriAttention** (Mao et al., ICML 2026, [arXiv:2604.04921](https://arxiv.org/abs/2604.04921);
repo [WeianMao/triattention](https://github.com/WeianMao/triattention)), evaluated on **LongBench v1**
with **Llama-3.1-8B-Instruct** (bf16) on an RTX 5090.

TriAttention shrinks the KV cache by scoring each cached key with a frequency-domain trigonometric
series built from **precomputed per-head query statistics**. It inverts RoPE on the cached keys,
combines them with the calibrated per-head query centers and magnitudes, and keeps the highest-scoring
tokens plus a recent window and the initial sink tokens. It is implemented as a kvpress `ScorerPress`
and runs through the team's shared harness against the full-cache baseline on the same prompts and metrics.

## Faithful to the method

The scoring is the authors' exact algorithm (`triattention/methods/pruning_utils.py`), driven by
statistics from the authors' calibration procedure, reproduced in `calibrate_triattention.py`:
a single forward pass on plain text captures each layer's query states, inverts RoPE, forms
`head_dim/2` complex frequency pairs, and stores per head the complex query mean and mean magnitude
(`calibration/llama31_8b_stats.pt`, 1024 heads).

This matters for finding real flaws. An earlier version estimated the query center online instead of
calibrating it, which both **invented flaws that do not exist** (it reported hotpotqa collapsing to
16.7 at @0.5, where the calibrated method scores **91.7**) and exaggerated the real ones. The results
below use the calibrated statistics, so the weaknesses are TriAttention's own.

## Headline result

Halving the KV cache costs ~2 points; past that, accuracy falls steadily as the cache shrinks toward
13× smaller.

| Run | Macro avg | Mean KV cache (GB) |
|---|---|---|
| Full cache | 45.24 | 0.4654 |
| TriAttention @0.5 | 43.18 | 0.2396 |
| TriAttention @0.75 | 37.23 | 0.1267 |
| TriAttention @0.9 | 32.54 | 0.0589 |
| TriAttention @0.95 | 31.50 | 0.0364 |

![Accuracy vs KV-cache memory](results_out/figures/score_vs_kv_cache.png)

![Accuracy vs memory, animated](results_out/figures/accuracy_vs_memory.gif)

## Per-task behaviour

QA tasks collapse as the budget shrinks; code, few-shot, and summarization hold up far better.

![Per-task scores](results_out/figures/per_task_scores.png)

The same per-task scores, one frame per compression level from full cache to the most aggressive budget:

![Per-task sweep, animated](results_out/figures/per_task_sweep.gif)

![Macro average per run](results_out/figures/macro_average.png)

## The flaw

TriAttention's biggest weakness is **question-answering** — at @0.95, multifieldqa_en drops 58.4 → 19.2,
2wikimqa 52.7 → 17.2, qasper 40.9 → 16.8. These tasks need a few specific evidence tokens scattered in
a long document. The frequency-domain score ranks a key by expected attention averaged over query
distances using the calibrated query geometry — it is not sensitive to the specific question — so
isolated evidence tokens score low and get evicted. Tasks with redundant or structurally regular signal
(code, few-shot, summarization) survive because the information is spread across many tokens.

Category-level damage at @0.95 (mean Δ vs full cache): Single-Doc QA −31.6, Multi-Doc QA −21.9,
Summarization −8.8, Code −7.4, Few-shot −5.5.

Full per-task tables are in `RESULTS.md`; the measured flaw breakdown and root cause are in `FLAWS.md`.

## Files

**Code**
- `triattention_press.py` — TriAttention as a kvpress `ScorerPress` (RoPE inversion + trigonometric
  scoring over calibrated stats; attention-sink + recent-window protection)
- `calibrate_triattention.py` — reproduces the authors' calibration → `calibration/llama31_8b_stats.pt`
- `build_dev_data.py` — builds the LongBench dev set → `data/processed_longbench/processed_longbench_dev.jsonl`
- `run_all.py` — runs the full-cache baseline + the TriAttention compression sweep, then the analysis
- `analyze_results.py` — turns run outputs into CSVs, figures, GIFs, and `RESULTS.md`
- `write_flaws.py` — derives `FLAWS.md` from the measured scores
- `selftest_triattention.py` — plumbing check on a small open model

**Outputs** (`results_out/`)
- `per_record.csv` — every record: task, score, latency, KV-cache size
- `per_task_summary.csv` — mean score per task per run
- `run_summary.csv` — macro average, mean KV cache, mean latency per run
- `run_log.txt` — full harness output
- `figures/` — `per_task_scores.png`, `score_vs_kv_cache.png`, `macro_average.png`,
  `per_task_sweep.gif`, `accuracy_vs_memory.gif`
- `longbench/` — the TriAttention per-ratio run files (predictions + summaries)

**Documents**
- `RESULTS.md` — full results tables and figures
- `FLAWS.md` — TriAttention's weaknesses and root cause, computed from the measured scores

## Setup notes

- Runs cover the LongBench dev prompts under 6K tokens (260 records per run, all 16 tasks); longer
  prompts exceed the 32 GB GPU during the uncompressed prefill.
- Calibration uses a 4096-token plain-text pass; the authors note the statistics are domain-agnostic
  and robust to the calibration text.

## Reproduce

```powershell
huggingface-cli login
C:\Python310\python.exe members/samruddhi/build_dev_data.py
C:\Python310\python.exe members/samruddhi/calibrate_triattention.py --output members/samruddhi/calibration/llama31_8b_stats.pt
C:\Python310\python.exe members/samruddhi/run_all.py --compression-ratios 0.5,0.75,0.9,0.95 --max-prompt-tokens 6000
```
