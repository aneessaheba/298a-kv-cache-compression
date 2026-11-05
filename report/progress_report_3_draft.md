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

*Pending: RTX 5090 smoke test, 2 records per task (`python harness/run_longbench.py --per-task 2`).*

| Task group | Tasks | Score |
|---|---|---|
| Single-doc QA | narrativeqa, qasper, multifieldqa_en | |
| Multi-doc QA | hotpotqa, 2wikimqa, musique | |
| Summarization | gov_report, qmsum, multi_news | |
| Few-shot | trec, triviaqa, samsum | |
| Synthetic | passage_count, passage_retrieval_en | |
| Code | lcc, repobench-p | |
| **Average** | | |

Time per record: ___ s (median). Peak GPU memory: ___ GB.

## 4. Method results

### 4.1 RestoreKV (Anees)

RestoreKV adds 8 learned restore tokens on top of KVzip eviction. Before eviction, the restore tokens read the full cache once through a small LoRA adapter, then take 8 slots per layer and head inside the same budget. Only the restore embeddings and LoRA weights (~0.4% of parameters) are trained, and the released Llama-3.1-8B-Instruct adapter is used as is. The adapter was trained at keep ratios of 2.5–25%, so our main budgets stay in that range.

**RULER 4K, 5% budget, 50 examples per task (SAM-94).** *Pending.*

| Method | Avg | NIAH tasks | VT | CWE/FWE | QA |
|---|---|---|---|---|---|
| Full cache | | | | | |
| KVzip | | | | | |
| RestoreKV | | | | | |

**Planned for the Nov 2 gate (SAM-95).** RestoreKV vs KVzip vs SnapKV at 5, 10, 20 and 25% budgets on the 800-record LongBench dev set.

### 4.2 to 4.5 Other methods

*Each owner adds their method summary and first result here.*

## 5. Next steps (to the Nov 13 results freeze)

- All methods through the harness on the 800-record dev set at equal computed KV size (Nov 2 gate).
- RestoreKV with 4-bit and 2-bit quantization of the kept cache, at equal memory (N1, SAM-96).
- Length sweep from 4K to 32K on RULER and NIAH (N2, SAM-97).
- Follow-up questions answered on one compressed cache (N3, SAM-98).
