# RestoreKV (Anees)

[RestoreKV](https://arxiv.org/abs/2608.01247) evicts tokens with KVzip, then adds 8 learned restore tokens that summarize what was dropped. It uses kvpress's `RestoreKVPress` and the released adapter `higokri/RestoreKV-Llama-3.1-8B-Instruct`, with no training. The restore tokens are counted inside the memory budget.

## Setup

| | |
|---|---|
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16 |
| GPU | NVIDIA RTX 5090 (32 GB), SJSU lab, Docker |
| Software | Python 3.11.13, torch 2.7.1+cu128, transformers 5.2.0, kvpress 0.5.5, peft 0.21.2, datasets 5.1.0 |
| Code | Run from team repo commit `c26db9d`, then rescored after the scoring fix below |
| Data | All 800 LongBench dev records (16 tasks, 50 each); RULER 4K, 13 tasks, 50 examples each |

## Commands

```bash
python members/anees/smoke_test.py

python harness/run_longbench.py
python harness/run_longbench.py --press restorekv --compression-ratio 0.75   # also 0.8, 0.9, 0.95

python harness/run_ruler.py --per-task 50
python harness/run_ruler.py --press kvzip --compression-ratio 0.95 --per-task 50
python harness/run_ruler.py --press restorekv --compression-ratio 0.95 --per-task 50
```

The shared 800-record full-cache baseline is in `results/longbench/`, and the RestoreKV and RULER runs are in `results/restorekv/`.

## LongBench results

| Budget (share of document cache kept) | Macro average | Change vs. full cache (95% CI) | Mean KV cache |
|---|---|---|---|
| Full cache | 47.72 | | 1.357 GB |
| 25% | 47.31 | -0.41 (-1.85 to 0.95) | 0.355 GB |
| 20% | 47.63 | -0.09 (-1.80 to 1.63) | 0.288 GB |
| 10% | 43.29 | -4.42 (-6.45 to -2.35) | 0.154 GB |
| 5% | 36.13 | -11.58 (-13.93 to -9.15) | 0.087 GB |

Confidence intervals come from a paired bootstrap over records within each task (5,000 resamples).

| Task | Full | 25% | 20% | 10% | 5% |
|---|---|---|---|---|---|
| narrativeqa | 35.47 | 36.80 | 40.69 | 31.69 | 28.58 |
| qasper | 41.63 | 43.04 | 45.73 | 47.47 | 40.62 |
| multifieldqa_en | 54.85 | 53.34 | 53.43 | 53.83 | 42.49 |
| hotpotqa | 52.72 | 49.48 | 51.17 | 48.87 | 45.90 |
| 2wikimqa | 45.53 | 46.28 | 45.08 | 41.18 | 33.17 |
| musique | 26.50 | 23.83 | 25.62 | 20.28 | 16.02 |
| gov_report | 33.88 | 33.52 | 33.06 | 32.11 | 27.54 |
| qmsum | 25.06 | 26.09 | 26.65 | 25.90 | 25.32 |
| multi_news | 27.01 | 26.99 | 27.49 | 26.76 | 24.98 |
| trec | 70.00 | 66.00 | 68.00 | 62.00 | 48.00 |
| triviaqa | 93.20 | 89.20 | 88.53 | 93.20 | 94.19 |
| samsum | 49.31 | 48.19 | 48.35 | 45.34 | 38.47 |
| passage_count | 4.50 | 4.67 | 6.00 | 8.00 | 4.00 |
| passage_retrieval_en | 100.00 | 98.00 | 92.00 | 52.00 | 14.00 |
| lcc | 54.46 | 55.68 | 55.29 | 50.05 | 37.50 |
| repobench-p | 49.33 | 55.79 | 54.96 | 54.01 | 57.32 |

Time per record: median 1.78 s for full cache and about 7.3 s for RestoreKV.

## RULER results (4K, 5% budget, 650 examples per method)

| Method | Average | Median KV cache | Median time |
|---|---|---|---|
| Full cache | 96.43 | 0.495 GB | 0.75 s |
| KVzip | 62.18 | 0.029 GB | 3.18 s |
| RestoreKV | 78.89 | 0.029 GB | 3.25 s |

## Notes

- **Scoring fix.** On trec, triviaqa and samsum the score uses the first line of the output. Llama often starts its answer with " \n", which made that line blank, so correct answers scored 0. `score()` now strips all leading whitespace on those tasks. LongBench's own `eval.py` strips only newlines, so this differs slightly from the official scorer. It changes full-cache triviaqa from 8.00 to 93.20 and the macro average from 42.39 to 47.72.
- **Why the fix matters here.** The bug hit full cache harder than compressed runs. Before the fix, RestoreKV at 20% and 25% appeared to beat full cache.
- **Memory.** kvpress KVzip and RestoreKV mask evicted pairs and leave them in GPU memory, so peak GPU memory stays around 24 GB on LongBench. Compare methods on the computed `kv_cache_gb`, which includes the uncompressed question.
- **Interruptions.** The lab machine shut down twice, during the RULER KVzip run and the LongBench 10% run. Both resumed from saved records, and every result file has one row per record with no duplicates.
- **Still missing.** LongBench KVzip and SnapKV haven't been run, so these tables compare RestoreKV only with full cache. The comparison with published LongBench and RestoreKV numbers is also still to do.
