# Progress Report 3: Draft

Team 6, KV-cache compression on Llama-3.1-8B-Instruct. Draft started Oct 22, 2026 (SAM-133). Sections marked *pending* get filled from the RTX 5090 runs before the Oct 26 review (SAM-125).

## 1. What changed since Progress Report 2

- Narrowed the method list to one 2026-era method per member, plus SnapKV and KIVI as shared baselines. The whole team agreed on Oct 22 (SAM-111).
- Data pipeline finished: 3,731 cleaned LongBench records and an 800-record dev set, 50 per task (SAM-64).
- Shared evaluation harness built and connected to the full-cache baseline (SAM-124, PR #47).

| Member | Method | Type | Code |
|---|---|---|---|
| Anees | RestoreKV (arXiv 2608.01247) | Eviction + learned restoration | kvpress `RestoreKVPress`, released Llama-3.1-8B adapter |
| Vedant | TurboQuant (arXiv 2504.19874, ICLR 2026) | Quantization | Community implementations |
| Samruddhi | TriAttention (arXiv 2604.04921, ICML 2026); KVTC in spring | Token selection (pre-RoPE scoring) | github.com/WeianMao/triattention |
| Charvee | ResidualKV, formerly DeltaKV (arXiv 2602.08005) | Residual compression | github.com/CURRENTF/DeltaKV (repo keeps the old name), released Llama-3.1-8B compressor |
| Manjot | RDKV (arXiv 2605.08317) | Joint eviction + quantization (rate-distortion bit allocation) | No public code yet; fallback HqeKV (ACL 2026 Findings) |

Excluded or deferred: KVTC moves to spring (no public code; it compresses the cache for storage, not at runtime). DirectKV is kept as a systems reference (it needs GH200 hardware and is lossless).

## 2. Experimental setup

**Model and hardware.** Llama-3.1-8B-Instruct in bf16 on the SJSU lab RTX 5090 (32 GB), using kvpress 0.5.5.

**Data.** The LongBench v1 English dev set: 16 tasks, 50 records each, 800 in total. Prompts use the official LongBench templates and are capped at 32K tokens with the Llama tokenizer, keeping the head and tail. 21 of the 800 dev prompts are truncated. RULER uses kvpress's build: 13 tasks at 4K, 8K and 16K tokens.

**One shared harness for every method** (`harness/`):
- Each prompt is split into the document, which a method may compress, and the instruction and question, which are never compressed. Every method therefore sees the same input and only the document is compressed.
- Generation is greedy, with LongBench's per-task output lengths.
- trec, triviaqa, samsum, lcc and repobench-p run without the chat template, following LongBench.

**Metrics.**
- LongBench: the official per-task metrics (QA F1, ROUGE-L, classification accuracy, retrieval/count accuracy, code edit similarity), reported per task and as the macro average.
- RULER: kvpress's string-match scores.
- Systems metrics are reported separately (SAM-126): time per record and peak GPU memory.

**Memory accounting.** KVzip and RestoreKV in kvpress mask evicted key/value pairs during attention but leave them in GPU memory, so measured peak memory does not fall as the budget shrinks. We therefore compare methods on the computed KV-cache size: kept tokens × 0.125 MiB/token for Llama-3.1-8B in bf16. A full 32K-token cache is 4.2 GB. RestoreKV's 8 restore tokens are counted inside the budget.

## 3. Baseline results (full cache)

All 800 dev records on the RTX 5090.

| Task group | Tasks | Score |
|---|---|---|
| Single-doc QA | narrativeqa, qasper, multifieldqa_en | 43.98 |
| Multi-doc QA | hotpotqa, 2wikimqa, musique | 41.58 |
| Summarization | gov_report, qmsum, multi_news | 28.65 |
| Few-shot | trec, triviaqa, samsum | 70.84 |
| Synthetic | passage_count, passage_retrieval_en | 52.25 |
| Code | lcc, repobench-p | 51.90 |
| **Average over 16 tasks** | | **47.72** |

Time per record: 1.78 s (median). Peak GPU memory: 24.09 GB. Computed KV cache: 1.357 GB on average.

**Scoring note.** trec, triviaqa and samsum are scored on the first line of the output. LongBench's `eval.py` strips only leading newlines before taking that line. Llama often starts these answers with a space and then a newline, which leaves the first line blank, so we strip all leading whitespace instead. This changes only triviaqa on the full cache, from 8.00 to 93.20, and moves the average from 42.39 to 47.72.

**Comparison with published numbers (G1, SAM-69).** Two 2024–2026 papers report full-cache LongBench scores for Llama-3.1-8B-Instruct on all 16 English tasks:

| Source | Average | Difference from ours |
|---|---|---|
| Ours (800-record dev set, 32K cap) | 47.72 (95% CI 45.76 to 49.85) | |
| Ada-KV, Feng et al. (arXiv 2407.11550), Table 5 | 49.20 | -1.48 |
| LKV (arXiv 2605.06676), Table 8 | 47.48 | +0.24 |

Our average is within 2 points of both, and both published averages fall inside our confidence interval, so the baseline passes the G1 gate. Per task, most differences are within our sampling error: with 50 records per task, the 95% interval is about ±10 points on the QA tasks and ±2 on summarization. Three tasks stand out:

- **samsum** is higher for us: 49.31, compared with 43.57 and 36.99. This is likely our leading-whitespace fix, together with running samsum without the chat template, as LongBench does.
- **lcc and repobench-p** are lower than Ada-KV (by 8.4 and 7.1) but close to LKV. The two papers differ from each other by 12 points on lcc, so code scores depend heavily on the setup.
- **passage_count** is 4.50, compared with 9.41 and 7.45. It is near zero for every source, and our interval is ±5.5.

Our dev set covers 50 records per task, not the full test set, and prompts are capped at 32K tokens. The published numbers use the full set, so differences of a few points on single tasks are expected.

## 4. Method results

### 4.1 RestoreKV (Anees)

RestoreKV adds 8 learned restore tokens on top of KVzip eviction. Before eviction, the restore tokens read the full cache once through a small LoRA adapter, then take 8 slots per layer and head inside the same budget. Only the restore embeddings and LoRA weights (~0.4% of parameters) are trained, and the released Llama-3.1-8B-Instruct adapter is used as is. The adapter was trained at keep ratios of 2.5–25%, so our main budgets stay in that range.

**RULER 4K, 5% budget, 50 examples per task (SAM-94).**

| Method | Avg | NIAH tasks | VT | CWE / FWE | QA 1 / QA 2 |
|---|---|---|---|---|---|
| Full cache | 96.43 | 99.69 | 99.6 | 99.2 / 95.3 | 90 / 72 |
| KVzip | 62.18 | 72.56 | 98.8 | 3.0 / 60.0 | 28 / 38 |
| RestoreKV | 78.89 | 92.94 | 99.6 | 5.8 / 70.7 | 60 / 46 |

At the same memory budget, RestoreKV scores 16.71 points higher than KVzip on average and at least as high on all 13 tasks, for about 0.07 s more per example. Neither method keeps CWE, which requires counting across the whole context.

**LongBench, all 800 records (SAM-95).**

| Share of document cache kept | Average | Change vs. full cache (95% CI) | Mean KV cache |
|---|---|---|---|
| Full cache | 47.72 | | 1.357 GB |
| 25% | 47.31 | -0.41 (-1.85 to 0.95) | 0.355 GB |
| 20% | 47.63 | -0.09 (-1.80 to 1.63) | 0.288 GB |
| 10% | 43.29 | -4.42 (-6.45 to -2.35) | 0.154 GB |
| 5% | 36.13 | -11.58 (-13.93 to -9.15) | 0.087 GB |

Confidence intervals are from a paired bootstrap over records within each task. At 20% and 25% budgets the difference from the full cache is within noise, while the losses at 10% and 5% are clear. The largest single loss is passage_retrieval_en: 100 at full cache, 52 at 10% and 14 at 5%. Median time per record is about 7.3 s, compared with 1.78 s for the full cache.

**Still to run.** LongBench KVzip and SnapKV at the same budgets, so that this table compares methods rather than only RestoreKV against the full cache.

### 4.2 to 4.5 Other methods

*Each owner adds their method summary and first result here.*

## 5. Next steps (to the Nov 13 results freeze)

- All methods through the harness on the 800-record dev set at equal computed KV size (Nov 2 gate).
- RestoreKV with 4-bit and 2-bit quantization of the kept cache, at equal memory (N1, SAM-96).
- Length sweep from 4K to 32K on RULER and NIAH (N2, SAM-97).
- Follow-up questions answered on one compressed cache (N3, SAM-98).
