# KV-Cache Compression at Equal Memory

Team 6 capstone for DATA 298A (Fall 2026) and DATA 298B (Spring 2027) at San José State University.

During generation, a large language model stores a key and a value for every earlier token. This KV cache grows with the length of the input and takes up a large share of GPU memory on long documents. Recent papers shrink it in different ways: dropping tokens, quantizing them, or storing them more compactly. These papers rarely compare under the same conditions. **This project compares five 2026-era KV-cache compression methods on one model, one set of benchmarks and one memory budget**, so that the differences in accuracy come from the methods themselves.

- **Model:** Llama-3.1-8B-Instruct (bf16)
- **Benchmarks:** LongBench v1 (English, 16 tasks), RULER (4K–16K), needle-in-a-haystack
- **Hardware:** SJSU lab RTX 5090 (32 GB) and RTX 4090
- **Comparison rule:** every method gets the same KV-cache size

## Methods

| Member | Method | How it compresses | Code |
|---|---|---|---|
| Anees | [RestoreKV](https://arxiv.org/abs/2608.01247) (2026) | Drops tokens (KVzip), then adds 8 learned "restore" tokens that summarize what was dropped | kvpress `RestoreKVPress` |
| Vedant | [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) | Quantizes keys and values to 2.5–3.5 bits | Community implementations |
| Samruddhi | [TriAttention](https://arxiv.org/abs/2604.04921) (ICML 2026) | Keeps the tokens a stable pre-RoPE attention score ranks highest | [WeianMao/triattention](https://github.com/WeianMao/triattention) |
| Charvee | [ResidualKV](https://arxiv.org/abs/2602.08005) (2026, formerly DeltaKV) | Stores each token as a small difference from a similar earlier token | [CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV) |
| Manjot | [RDKV](https://arxiv.org/abs/2605.08317) (2026) | Gives each token a bit-width from 0 (dropped) to 16 using rate-distortion theory | No public code yet |

**Shared baselines:** full cache (no compression), SnapKV (eviction) and KIVI (quantization).

**Deferred to spring:** KVTC (transform coding, Samruddhi).

## Repository layout

```
Data_pipeline/   LongBench download, EDA and preprocessing notebooks, cleaning logs
harness/         Shared evaluation code: data loading, metrics, runners
baselines/       Shared baseline methods (SnapKV, KIVI)
members/<name>/  Each member's method code and scratch outputs
results/         Result files from the runners (one .jsonl per run plus a .summary.json)
report/          Progress reports and the final report
docs/            Meeting notes and design decisions
```

## Setup

Python 3.11, with a CUDA GPU for real runs. The code also runs on Apple Silicon (mps) for quick checks.

```bash
git clone https://github.com/samruddhichitnis02/298a-kv-cache-compression.git
cd 298a-kv-cache-compression
pip install kvpress rouge rapidfuzz datasets
huggingface-cli login   # Llama-3.1-8B-Instruct needs the accepted Meta license
```

**Data.** The processed LongBench files are too large for git. Get `processed_longbench.zip` from Samruddhi and unzip it at the repo root, so that this file exists:

```
data/processed_longbench/processed_longbench_dev.jsonl
```

`data/` is git-ignored. The dev set has 800 records: 50 per task, prompts in the official LongBench templates, capped at 32K Llama tokens with the head and tail kept. `Data_pipeline/Scripts/02_LongBench_Data_Preprocessing.ipynb` rebuilds it from scratch. RULER downloads itself from Hugging Face (`simonjegou/ruler`) on first use.

## Running the harness

```bash
# LongBench: full cache, 2 records per task (quick smoke test)
python harness/run_longbench.py --per-task 2

# LongBench: a method at a given compression ratio (0.9 = drop 90%, keep 10%)
python harness/run_longbench.py --press restorekv --compression-ratio 0.9

# RULER at 4K tokens, 50 examples per task
python harness/run_ruler.py --press kvzip --compression-ratio 0.95 --per-task 50
```

Useful options:
- `--tasks qasper,trec`: only these tasks
- `--max-prompt-tokens 6000`: skip long prompts on a laptop
- `--length 8192` or `16384`: RULER context length
- `--model`: any Hugging Face model id

Built-in presses: `none`, `snapkv`, `kvzip`, `restorekv`. A run that stops halfway picks up where it left off when you rerun the same command.

**Output.**
- One line per example in `results/longbench/<model>_<press>_<ratio>.jsonl`: prediction, score, seconds, computed KV-cache size and peak GPU memory.
- Per-task scores and their average in the matching `.summary.json`.

## How the comparison is kept fair

- **Same input split.** Each prompt is split into the document, which a method may compress, and the instruction plus question, which is never compressed. Every method sees exactly the same text.
- **Official metrics.** LongBench is scored with the official per-task metrics (QA F1, ROUGE-L, classification, retrieval, count, code similarity) and RULER with kvpress's string match. trec, triviaqa, samsum, lcc and repobench-p run without the chat template, as in LongBench.
- **Greedy decoding** with LongBench's per-task output lengths.
- **Memory is computed, not measured.** Some kvpress methods (KVzip, RestoreKV) mask dropped tokens instead of freeing them, so measured GPU memory doesn't fall. Each result therefore records `kv_cache_gb`: tokens kept × 0.125 MiB per token for Llama-3.1-8B in bf16. A full 32K-token cache is about 4.2 GB.
- **Quality and systems metrics are reported separately** (accuracy vs. time and memory).

## Adding your method

1. Put your code in `members/<your-name>/`.
2. If your method is a kvpress press, add it to `make_press()` in `harness/run_longbench.py`, and it then works with both runners.
3. Otherwise, write a runner that reads the same records with `harness/longbench.py`:
   - `load_records()` reads the data and `split_prompt()` splits each prompt.
   - Score with `score()` and use the same output fields, so results can be compared side by side.
4. Report `kv_cache_gb` for the memory your method actually keeps.

## Timeline

| Date | Milestone |
|---|---|
| Nov 2, 2026 | Baseline gate: every method runs on the 800-record dev set |
| Nov 13, 2026 | Results freeze |
| Nov 20, 2026 | Fall report draft (team target) |
| Spring 2027 | Full benchmarks, 64K context, KVTC, combined methods |

Tasks are tracked in Linear (project *KV-Cache Optimization (Team 6)*); each Linear issue is mirrored as a GitHub issue here.
