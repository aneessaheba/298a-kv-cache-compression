"""Run a model, with or without a KV-cache press, on the processed LongBench dev set (SAM-124).

Each record's document is passed as the compressible context and the rest of the
prompt as the uncompressed question, so every method is scored on the same split.
Predictions go to results/longbench/<model>_<press>_<ratio>.jsonl, one line per
record; rerunning the same command skips records that are already there.

    # Smoke test: full cache, 2 records per task
    python harness/run_longbench.py --per-task 2

    # Full dev set (800 records) with RestoreKV keeping 10% of the cache
    python harness/run_longbench.py --press restorekv --compression-ratio 0.9

The data is not in git; by default it is read from data/processed_longbench/
at the repo root (unzip Samruddhi's processed_longbench.zip there).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import pipeline

import sys

import longbench
from kvpress import KVzipPress, RestoreKVPress, SnapKVPress  # also registers the kv-press pipeline

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"
OUT_DIR = REPO / "results" / "longbench"

# Samruddhi's TriAttention press lives under members/ rather than in kvpress.
sys.path.insert(0, str(REPO / "members" / "samruddhi"))
from triattention_press import TriAttentionPress  # noqa: E402
from chunked_prefill import chunked_prefill  # noqa: E402


def generate_chunked(pipe, context, question, answer_prefix, press, ratio, max_new_tokens, chunk_size):
    """Chunked-prefill variant of the pipeline: build a compressed cache in chunks, then decode.

    Reuses the pipeline's preprocess (tokenisation + chat template) and greedy decoder, but
    replaces the single-shot prefill with chunked_prefill so long prompts fit in GPU memory.
    """
    inputs = pipe.preprocess(context, questions=[question], answer_prefix=answer_prefix,
                             max_context_length=min(pipe.tokenizer.model_max_length, int(1e10)))
    context_ids = inputs["context_ids"]
    n_ctx = context_ids.shape[1]
    budget = max(1, round((1 - ratio) * n_ctx)) if (press is not None and ratio > 0) else None
    cache = chunked_prefill(pipe.model, context_ids, press, chunk_size, budget)
    context_length = cache.get_seq_length()
    answer = pipe.generate_answer(question_ids=inputs["questions_ids"][0].to(pipe.model.device),
                                  cache=cache, context_length=context_length, max_new_tokens=max_new_tokens)
    return answer

# RestoreKV + quantization (SAM-96); --compression-ratio is the share of bf16 cache bytes removed.
sys.path.insert(0, str(REPO / "members" / "anees"))
from quantized_restorekv_press import QuantizedRestoreKVPress  # noqa: E402


def make_press(name, compression_ratio):
    if name == "none":
        return None
    if name in ("restorekv_4bit", "restorekv_2bit"):
        return QuantizedRestoreKVPress(compression_ratio=compression_ratio, bits=int(name.split("_")[1][0]))
    presses = {
        "snapkv": SnapKVPress,
        "kvzip": KVzipPress,
        "restorekv": RestoreKVPress,
        "triattention": TriAttentionPress,
    }
    if name not in presses:
        raise ValueError(f"unknown press: {name}")
    return presses[name](compression_ratio=compression_ratio)


def kv_bytes_per_token(config, bytes_per_value=2):
    """Size of one token's keys and values across all layers (bf16 by default)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * bytes_per_value


def kv_cache_gb(pipe, context, question, ratio):
    """KV cache kept after compression: the compressed context plus the uncompressed question.

    Computed rather than measured, because some presses (KVzip, RestoreKV) only
    mask evicted pairs and leave them in GPU memory.
    """
    n_context = len(pipe.tokenizer(context, add_special_tokens=False).input_ids)
    n_question = len(pipe.tokenizer(question, add_special_tokens=False).input_ids)
    kept = round((1 - ratio) * n_context) + n_question
    return round(kept * kv_bytes_per_token(pipe.model.config) / 1e9, 4)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--press", default="none", help="none, snapkv, kvzip, restorekv, restorekv_4bit or restorekv_2bit")
    parser.add_argument("--compression-ratio", type=float, default=0.0, help="fraction of the KV cache removed")
    parser.add_argument("--tasks", default="", help="comma-separated task names (default: all 16)")
    parser.add_argument("--per-task", type=int, default=None, help="records per task (default: all)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="skip longer records, for quick runs on a laptop")
    parser.add_argument("--chunk-prefill", type=int, default=None,
                        help="prefill the context in chunks of this many tokens so long prompts fit in "
                             "GPU memory (enables the full dev set). Compression is applied after each chunk.")
    args = parser.parse_args()

    tasks = set(args.tasks.split(",")) if args.tasks else None
    records = longbench.load_records(args.data, tasks=tasks, per_task=args.per_task,
                                     max_prompt_tokens=args.max_prompt_tokens)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ratio = args.compression_ratio if args.press != "none" else 0.0
    out_file = OUT_DIR / f"{args.model.split('/')[-1]}_{args.press}_{ratio}.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(line)["record_id"] for line in out_file.open()}
    todo = [r for r in records if r["record_id"] not in done]

    device = pick_device()
    print(f"device={device}  model={args.model}  press={args.press}  ratio={ratio}  "
          f"records={len(records)} ({len(done)} already done)")
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
    press = make_press(args.press, ratio)
    if args.press.startswith("restorekv"):
        # Load the LoRA adapter now so it lands on the model's device (see members/anees/smoke_test.py)
        press.post_init_from_model(pipe.model)
        pipe.model.to(device)

    chat_template = pipe.tokenizer.chat_template
    with out_file.open("a") as out:
        for i, record in enumerate(todo, 1):
            context, question = longbench.split_prompt(record)
            # Without a chat template the pipeline feeds the raw prompt after the BOS token
            # (kvpress needs one; Llama has it, some small test models such as Qwen do not)
            no_template = record["task"] in longbench.NO_CHAT_TEMPLATE_TASKS and pipe.tokenizer.bos_token
            pipe.tokenizer.chat_template = None if no_template else chat_template
            answer_prefix = ""
            if no_template:
                question, answer_prefix = longbench.split_answer_cue(question)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            fatal = False
            try:
                if args.chunk_prefill:
                    prediction = generate_chunked(
                        pipe, context, question, answer_prefix, press, ratio,
                        longbench.MAX_NEW_TOKENS[record["task"]], args.chunk_prefill)
                else:
                    prediction = pipe(context, question=question, answer_prefix=answer_prefix, press=press,
                                      max_new_tokens=longbench.MAX_NEW_TOKENS[record["task"]])["answer"]
                oom = False
            # torch.AcceleratorError (torch >= 2.8) is a RuntimeError, so RuntimeError covers it;
            # naming it directly breaks every OOM on the lab image's torch 2.7.1, which lacks it.
            except RuntimeError as e:
                # Record the failure instead of crashing the whole run. A plain OutOfMemoryError
                # is recoverable in-process; a harder CUDA AcceleratorError corrupts the context,
                # so we re-raise it and let the orchestrator (run_all.py) restart this ratio and
                # resume past the failed record (which is already written below as oom).
                msg = str(e)
                is_oom = "out of memory" in msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError)
                if not is_oom:
                    raise
                prediction = ""
                oom = True
                print(f"[{i}/{len(todo)}] {record['task']:20s} OOM at {record['prompt_tokens_after']} tok "
                      f"-- recorded as failure")
                fatal = not isinstance(e, torch.cuda.OutOfMemoryError)
            finally:
                if device == "cuda":
                    try:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass  # a corrupted context can make cleanup itself throw; ignore
            seconds = time.perf_counter() - start
            # On a fatal CUDA error the context is dead, so skip GPU-touching metrics.
            try:
                kvgb = kv_cache_gb(pipe, context, question + answer_prefix, ratio) if not fatal else None
            except Exception:
                kvgb = None
            try:
                peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if (device == "cuda" and not fatal) else None
            except Exception:
                peak = None
            row = {
                "record_id": record["record_id"],
                "task": record["task"],
                "press": args.press,
                "compression_ratio": ratio,
                "prompt_tokens": record["prompt_tokens_after"],
                "prediction": prediction,
                "score": 0.0 if oom else longbench.score(record["task"], prediction,
                                                         record["answers"], record["all_classes"]),
                "oom": oom,
                "seconds": round(seconds, 2),
                "kv_cache_gb": kvgb,
                "peak_gpu_gb": peak,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            if not oom:
                print(f"[{i}/{len(todo)}] {record['task']:20s} score={row['score']:.3f}  "
                      f"{row['seconds']:6.1f}s  {record['prompt_tokens_after']} tok")
            if fatal:
                # Context is corrupted; exit so the orchestrator restarts and resumes past this record.
                print(f"fatal CUDA error after {record['prompt_tokens_after']} tok; exiting for restart", flush=True)
                raise SystemExit(99)

    rows = [json.loads(line) for line in out_file.open()]
    rows = [r for r in rows if r["record_id"] in {rec["record_id"] for rec in records}]
    summary = longbench.summarize(rows)
    summary_file = out_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps({"args": vars(args), "device": device, "n_records": len(rows),
                                        "scores": summary}, indent=2))
    for task, value in summary.items():
        print(f"{task:22s} {value:6.2f}")
    print(f"saved {out_file} and {summary_file}")


if __name__ == "__main__":
    main()
