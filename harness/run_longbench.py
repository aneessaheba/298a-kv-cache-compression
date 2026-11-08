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
import re
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

# RestoreKV + quantization (SAM-96); --compression-ratio is the share of bf16 cache bytes removed.
sys.path.insert(0, str(REPO / "members" / "anees"))
from quantized_restorekv_press import QuantizedRestoreKVPress  # noqa: E402

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
    }
    if name not in presses:
        raise ValueError(f"unknown press: {name}")
    return presses[name](compression_ratio=compression_ratio)


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
                        help="none, snapkv, kvzip, restorekv, restorekv_4bit, restorekv_2bit, "
                             "turboquant_[mse_]{2,2.5,3,3.5,4}bit or turboquant_adaptive")
    parser.add_argument("--turboquant-layer-config", type=Path)
    parser.add_argument("--compression-ratio", type=float, default=0.0, help="fraction of the KV cache removed")
    parser.add_argument("--tasks", default="", help="comma-separated task names (default: all 16)")
    parser.add_argument("--per-task", type=int, default=None, help="records per task (default: all)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="skip longer records, for quick runs on a laptop")
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
    out_file = args.output_dir / f"{args.model.split('/')[-1]}_{args.press}_{ratio}.jsonl"
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
    print(f"device={device}  model={args.model}  press={args.press}  ratio={ratio}  "
          f"records={len(records)} ({len(done)} already done)")
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
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
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                prediction = pipe(context, question=question, answer_prefix=answer_prefix, press=press,
                                  max_new_tokens=longbench.MAX_NEW_TOKENS[record["task"]])["answer"]
                oom = False
            except torch.cuda.OutOfMemoryError:
                # Record the OOM instead of crashing the whole run: the kvpress prefill
                # computes full attention over the uncompressed prompt, so peak memory
                # scales with prompt length regardless of the compression ratio.
                torch.cuda.empty_cache()
                prediction = ""
                oom = True
                print(f"[{i}/{len(todo)}] {record['task']:20s} OOM at {record['prompt_tokens_after']} tok "
                      f"-- recorded as failure")
            seconds = time.perf_counter() - start
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
                "kv_cache_gb": (None if oom and args.press.startswith("turboquant_") else
                                kv_cache_gb(pipe, context, question, ratio, press, answer_prefix)
                                if args.press.startswith("turboquant_") else
                                kv_cache_gb(pipe, context, question + answer_prefix, ratio, press)),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            }
            if args.press.startswith("turboquant_"):
                row["turboquant_stats"] = {} if oom else dict(press.stats)
                row["dtype"] = "bfloat16"
                row["score_scale"] = "0-1"
                row["scoring_policy"] = "team-shared-harness-rapidfuzz"
                row["run_fingerprint"] = run_fingerprint
                row["model_revision"] = getattr(pipe.model.config, "_commit_hash", None)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if not oom:
                print(f"[{i}/{len(todo)}] {record['task']:20s} score={row['score']:.3f}  "
                      f"{row['seconds']:6.1f}s  {record['prompt_tokens_after']} tok")

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
