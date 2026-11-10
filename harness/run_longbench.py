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
import os
import re
import time
from pathlib import Path

# Let the CUDA allocator grow segments instead of fragmenting: on 15 GB T4s, ~2 GB was stranded as
# "reserved but unallocated" and long prompts ran out of memory. Must be set before torch touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
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

# Manjot's RDKV press (joint eviction + mixed-precision quantization, no public code).
sys.path.insert(0, str(REPO / "members" / "manjot"))
from rdkv_press import RDKVPress  # noqa: E402

# Vedant's TurboQuant: uniform/mixed rates, QJL-off ablations, or calibrated layer rates.
# It quantizes the whole document, so --compression-ratio is not used.
TURBOQUANT = re.compile(r"turboquant_(mse_)?(2(?:\.5)?|3(?:\.5)?|4)bit")


def make_turboquant_press(name, layer_config=None):
    match = TURBOQUANT.fullmatch(name)
    if match is None and name != "turboquant_adaptive":
        raise ValueError(f"unknown TurboQuant press: {name} (use turboquant_[mse_]{{2,2.5,3,3.5,4}}bit)")
    # Imported only when asked for, so the other methods' runs never depend on members/vedant.
    sys.path.insert(0, str(REPO / "members" / "vedant"))
    from turboquant_press import TurboQuantPress

    if name == "turboquant_adaptive":
        if layer_config is None:
            raise ValueError("turboquant_adaptive requires --turboquant-layer-config")
        return TurboQuantPress(bits=3, layer_configs=json.loads(Path(layer_config).read_text())["layers"])
    return TurboQuantPress(bits=float(match.group(2)), qjl_keys=match.group(1) is None)


def make_press(name, compression_ratio, layer_config=None):
    if name == "none":
        return None
    if name in ("restorekv_4bit", "restorekv_2bit"):
        return QuantizedRestoreKVPress(compression_ratio=compression_ratio, bits=int(name.split("_")[1][0]))
    if name.startswith("turboquant_"):
        return make_turboquant_press(name, layer_config)
    presses = {
        "snapkv": SnapKVPress,
        "kvzip": KVzipPress,
        "restorekv": RestoreKVPress,
        "triattention": TriAttentionPress,
        "rdkv": RDKVPress,
    }
    if name not in presses:
        raise ValueError(f"unknown press: {name}")
    return presses[name](compression_ratio=compression_ratio)


def avoid_sdpa_gqa_on_old_gpus():
    """On GPUs older than Ampere (e.g. T4), torch's SDPA has no fused kernel for grouped-query attention
    (enable_gqa=True), so it silently falls back to the math kernel, which builds the full fp32 attention
    matrix (~37 GB at 17K tokens) and runs out of memory. Expanding the KV heads first (repeat_kv) lets the
    memory-efficient kernel run instead; the result is mathematically the same."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 8:
        return
    try:
        import transformers.integrations.sdpa_attention as sdpa_attention
    except ImportError:
        return
    if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
        print("pre-Ampere GPU: SDPA grouped-query fast path disabled (KV heads expanded instead)")


def kv_bytes_per_token(config, bytes_per_value=2):
    """Size of one token's keys and values across all layers (bf16 by default)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * bytes_per_value


def kv_cache_gb(pipe, context, question, ratio, press=None, answer_prefix=""):
    """KV cache kept after compression: the compressed context plus the uncompressed question.

    Computed rather than measured, because some presses (KVzip, RestoreKV) only
    mask evicted pairs and leave them in GPU memory. A quantizing press such as
    TurboQuant gives its packed bytes per document token instead.
    """
    n_context = len(pipe.tokenizer(context, add_special_tokens=False).input_ids)
    n_question = len(pipe.tokenizer(question, add_special_tokens=False).input_ids)
    full = kv_bytes_per_token(pipe.model.config)
    if hasattr(press, "context_bytes_per_token"):
        # Obtain the exact protected question + chat suffix tokens from the same
        # pipeline preprocessing path used for generation. Document bytes include
        # channel selectors and both subgroup norm fields, counted by the press.
        encoded = pipe.preprocess(context, questions=[question], answer_prefix=answer_prefix,
                                  max_context_length=min(pipe.tokenizer.model_max_length, int(1e10)))
        n_question = encoded["questions_ids"][0].numel()
        return round((press.stats["packed_bytes"] + n_question * full) / 1e9, 6)
    kept = round((1 - ratio) * n_context) + n_question
    return round(kept * full / 1e9, 4)


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
    parser.add_argument("--press", default="none",
                        help="none, snapkv, kvzip, restorekv, restorekv_4bit, restorekv_2bit, triattention, rdkv, "
                             "turboquant_[mse_]{2,2.5,3,3.5,4}bit or turboquant_adaptive")
    parser.add_argument("--turboquant-layer-config", type=Path)
    parser.add_argument("--compression-ratio", type=float, default=0.0, help="fraction of the KV cache removed")
    parser.add_argument("--tasks", default="", help="comma-separated task names (default: all 16)")
    parser.add_argument("--per-task", type=int, default=None, help="records per task (default: all)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="skip longer records, for quick runs on a laptop")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="model dtype; use float16 on GPUs without bf16 (e.g. T4)")
    parser.add_argument("--device-map", default=None,
                        help="e.g. 'auto' to split the model over several GPUs (Kaggle 2xT4); default: one device")
    parser.add_argument("--max-gpu-mem", type=float, default=None,
                        help="with --device-map: GiB of weights allowed per GPU, leaving the rest for activations "
                             "(e.g. 8.5 on 2x T4)")
    parser.add_argument("--out-tag", default="",
                        help="suffix for the output file name, e.g. _fp16check, so test runs don't mix with real ones")
    parser.add_argument("--chunk-prefill", type=int, default=None,
                        help="prefill the context in chunks of this many tokens so long prompts fit in "
                             "GPU memory (enables the full dev set). Compression is applied after each chunk.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.press.startswith("turboquant_") and args.compression_ratio != 0:
        parser.error("TurboQuant rate is selected by its bit setting; --compression-ratio must be 0")
    if (args.press == "turboquant_adaptive") != (args.turboquant_layer_config is not None):
        parser.error("--turboquant-layer-config must be used exactly with --press turboquant_adaptive")

    tasks = set(args.tasks.split(",")) if args.tasks else None
    records = longbench.load_records(args.data, tasks=tasks, per_task=args.per_task,
                                     max_prompt_tokens=args.max_prompt_tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratio = args.compression_ratio if args.press != "none" else 0.0
    out_file = args.output_dir / f"{args.model.split('/')[-1]}_{args.press}_{ratio}{args.out_tag}.jsonl"
    run_fingerprint = None
    if args.press.startswith("turboquant_"):
        import hashlib
        from importlib.metadata import version
        sys.path.insert(0, str(REPO / "members" / "vedant" / "src"))
        from turboquant.protocol import prepare_output, sha256, validate_dataset
        # Full team dataset is validated even when a smoke subset is selected.
        validate_dataset(Path(args.data))
        specification = {"model": args.model, "dtype": "bfloat16", "press": args.press,
                         "dataset_sha256": sha256(Path(args.data)), "tasks": sorted(tasks or []),
                         "per_task": args.per_task, "max_prompt_tokens": args.max_prompt_tokens,
                         "runtime": {name: version(name) for name in
                                     ("torch", "transformers", "kvpress", "tokenizers", "rapidfuzz", "rouge")},
                         "allocation_sha256": sha256(args.turboquant_layer_config) if args.turboquant_layer_config else None,
                         "scorer_sha256": sha256(REPO / "harness" / "longbench.py"),
                         "runner_sha256": sha256(Path(__file__)),
                         "implementation_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in
                             sorted([REPO / "members" / "vedant" / "turboquant_press.py"] +
                                    list((REPO / "members" / "vedant" / "src").rglob("*.py"))))).hexdigest()}
        run_fingerprint = prepare_output(out_file, specification, {r["record_id"] for r in records})
    done = set()
    if out_file.exists():
        done = {json.loads(line)["record_id"] for line in out_file.open()}
    todo = [r for r in records if r["record_id"] not in done]

    device = pick_device()
    avoid_sdpa_gqa_on_old_gpus()
    print(f"device={device}  model={args.model}  press={args.press}  ratio={ratio}  "
          f"records={len(records)} ({len(done)} already done)")
    dtype = getattr(torch, args.dtype)
    if args.device_map:
        # memory-efficient attention; without it some setups (e.g. Kaggle T4) fall back to eager attention,
        # which builds the full attention matrix and runs out of memory on long prompts
        model_kwargs = {"attn_implementation": "sdpa"}
        if args.max_gpu_mem:
            model_kwargs["max_memory"] = {i: f"{args.max_gpu_mem}GiB" for i in range(torch.cuda.device_count())}
        pipe = pipeline("kv-press-text-generation", model=args.model, device_map=args.device_map, dtype=dtype,
                        model_kwargs=model_kwargs)
    else:
        pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=dtype)
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print("attention implementation:", getattr(pipe.model.config, "_attn_implementation", "?"))
    if done and args.press.startswith("turboquant_"):
        saved_revisions = {json.loads(line).get("model_revision") for line in out_file.open()}
        if saved_revisions != {getattr(pipe.model.config, "_commit_hash", None)}:
            raise ValueError("loaded model revision differs from saved predictions; choose a new output directory")
    press = make_press(args.press, ratio, args.turboquant_layer_config)
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
            for d in range(n_gpus):
                torch.cuda.reset_peak_memory_stats(d)
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
                print("OOM detail:", msg.splitlines()[0][:300] if msg else type(e).__name__)
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
                if fatal or (oom and args.press.startswith("turboquant_")):
                    kvgb = None  # TurboQuant's packed bytes are only known for a completed record
                elif args.press.startswith("turboquant_"):
                    kvgb = kv_cache_gb(pipe, context, question, ratio, press, answer_prefix)
                else:
                    kvgb = kv_cache_gb(pipe, context, question + answer_prefix, ratio, press)
            except Exception:
                kvgb = None
            try:
                peak = (round(sum(torch.cuda.max_memory_allocated(d) for d in range(n_gpus)) / 1e9, 2)
                        if (device == "cuda" and not fatal) else None)  # summed over GPUs with --device-map
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
            if args.press.startswith("turboquant_"):
                row["turboquant_stats"] = {} if oom else dict(press.stats)
                row["dtype"] = "bfloat16"
                row["score_scale"] = "0-1"
                row["scoring_policy"] = "team-shared-harness-rapidfuzz"
                row["run_fingerprint"] = run_fingerprint
                row["model_revision"] = getattr(pipe.model.config, "_commit_hash", None)
            if args.press == "rdkv" and not oom:
                # RDKV's own accounting: payload hits the ratio exactly in bits; this adds the
                # quantizer metadata (scales/zero-points, keep bitmaps) on top. Context only.
                s = press.summary()
                row["rdkv_context_gb"] = round(s["kv_cache_gb"], 4)
                row["rdkv_context_gb_payload"] = round(s["kv_cache_gb_payload"], 4)
                row["rdkv_kept_fraction_bits"] = round(s["kept_fraction_bits"], 4)
                row["rdkv_kept_fraction_tokens"] = round(s["kept_fraction_tokens"], 4)
                # Like kv_cache_gb (compressed context + uncompressed question), but with RDKV's real
                # context size including metadata: the number to compare against the baseline.
                n_question = len(pipe.tokenizer(question + answer_prefix, add_special_tokens=False).input_ids)
                row["rdkv_kv_cache_gb"] = round(s["kv_cache_gb"]
                                                + n_question * kv_bytes_per_token(pipe.model.config) / 1e9, 4)
            if args.dtype != "bfloat16" or args.device_map:
                row["run_config"] = f"{args.dtype},{args.device_map or device}"
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
    summary_file.write_text(json.dumps({"args": {**vars(args), "output_dir": str(args.output_dir),
                                        "turboquant_layer_config": str(args.turboquant_layer_config) if args.turboquant_layer_config else None}, "device": device, "n_records": len(rows),
                                        "scores": summary}, indent=2))
    for task, value in summary.items():
        print(f"{task:22s} {value:6.2f}")
    print(f"saved {out_file} and {summary_file}")


if __name__ == "__main__":
    main()
