"""Generate LongBench predictions with full KV, PolarQuant, or TurboQuant/QJL."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

import torch

from .kv_cache import TransformersTurboQuantCache
from .protocol import DATA_DIR, PROTOCOL_VERSION, prepare_output, sha256, validate_dataset


NO_CHAT_TEMPLATE_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _middle_truncate(tokenizer: Any, prompt: str, max_tokens: int) -> str:
    token_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if token_ids.numel() <= max_tokens:
        return prompt
    left = max_tokens // 2
    right = max_tokens - left
    return tokenizer.decode(token_ids[:left], skip_special_tokens=True) + tokenizer.decode(
        token_ids[-right:], skip_special_tokens=True
    )


def _format_prompt(tokenizer: Any, row: dict[str, Any], max_input_tokens: int) -> tuple[str, bool]:
    prompt = _middle_truncate(tokenizer, row["prompt"], max_input_tokens)
    if row["task"] in NO_CHAT_TEMPLATE_TASKS:
        return prompt, True
    return (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        False,
    )


def _dynamic_cache_bytes(cache: Any) -> int:
    total = 0
    for layer in cache.layers:
        if layer.keys is not None:
            total += layer.keys.numel() * layer.keys.element_size()
        if layer.values is not None:
            total += layer.values.numel() * layer.values.element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-scope", choices=["team", "subset"], default="team")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method", choices=["full", "polarquant", "turboquant", "adaptive"], required=True
    )
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument(
        "--adaptive-config",
        type=Path,
        help="adaptive_allocation.json produced by turboquant-adaptive",
    )
    parser.add_argument("--max-input-tokens", type=int, default=32_000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--task", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument(
        "--max-new-tokens-config",
        type=Path,
        default=DATA_DIR / "longbench_v1_max_new_tokens.json",
    )
    args = parser.parse_args()
    if args.max_input_tokens < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("token cap and sample limit must be positive")
    try:
        input_rows = validate_dataset(args.input, scope=args.dataset_scope)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    tasks = set(args.task or [])
    if tasks - {row["task"] for row in input_rows}:
        parser.error("requested task is not present in the dataset")
    input_rows = [row for row in input_rows if not tasks or row["task"] in tasks]
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    if args.method == "adaptive" and args.adaptive_config is None:
        parser.error("--adaptive-config is required for --method adaptive")
    if args.method != "adaptive" and args.adaptive_config is not None:
        parser.error("--adaptive-config is only valid for --method adaptive")
    adaptive_allocation = None
    adaptive_digest = None
    if args.adaptive_config is not None:
        adaptive_bytes = args.adaptive_config.read_bytes()
        adaptive_digest = hashlib.sha256(adaptive_bytes).hexdigest()
        adaptive_allocation = json.loads(adaptive_bytes)
        if adaptive_allocation.get("schema_version") != 1:
            parser.error("unsupported adaptive allocation schema")
        adaptive_layers = sorted(adaptive_allocation.get("layers", []), key=lambda row: row["layer"])
        if [row["layer"] for row in adaptive_layers] != list(range(len(adaptive_layers))):
            parser.error("adaptive allocation layers must be contiguous from zero")
        adaptive_allocation["layers"] = adaptive_layers
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type in {"cpu", "mps"} and dtype == torch.float16:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        token=args.token,
        revision=args.revision,
        attn_implementation="sdpa",
    ).to(device).eval()
    max_new_tokens = json.loads(args.max_new_tokens_config.read_text())
    if {row["task"] for row in input_rows} - set(max_new_tokens):
        parser.error("generation-length config lacks one or more input tasks")
    specification = {
        "protocol": PROTOCOL_VERSION, "input_sha256": sha256(args.input),
        "dataset_scope": args.dataset_scope, "source_records": len(input_rows),
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "method": args.method, "bits": args.bits if args.method != "full" else None,
        "adaptive_config_sha256": adaptive_digest, "max_input_tokens": args.max_input_tokens,
        "tasks": sorted(tasks), "seed": args.seed, "dtype": str(dtype), "device": str(device),
        "generation_limits": max_new_tokens, "decoding": "greedy", "attention": "sdpa",
        "transformers": version("transformers"), "torch": torch.__version__,
        "implementation_sha256": hashlib.sha256(b"".join(
            p.read_bytes() for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
        )).hexdigest(),
    }
    try:
        run_id = prepare_output(args.output, specification, {row["record_id"] for row in input_rows})
    except ValueError as error:
        parser.error(str(error))
    completed = set()
    if args.output.exists():
        completed = {row["record_id"] for row in _iter_jsonl(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        row for row in input_rows if row["record_id"] not in completed
    )
    count = 0
    with args.output.open("a", encoding="utf-8") as output:
        for row in rows:
            if args.max_samples is not None and count >= args.max_samples:
                break
            prompt, add_special_tokens = _format_prompt(tokenizer, row, args.max_input_tokens)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if args.method == "full":
                cache = DynamicCache(config=model.config)
            elif args.method == "adaptive":
                assert adaptive_allocation is not None
                cache = TransformersTurboQuantCache(
                    model.config,
                    layer_configs=adaptive_allocation["layers"],
                    seed=args.seed,
                )
            else:
                cache = TransformersTurboQuantCache(
                    model.config,
                    args.bits,
                    qjl_keys=args.method == "turboquant",
                    seed=args.seed,
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None or not 0 <= pad_token_id < model.config.vocab_size:
                    pad_token_id = tokenizer.eos_token_id
                if pad_token_id is None or not 0 <= pad_token_id < model.config.vocab_size:
                    pad_token_id = 0
                generated = model.generate(
                    **encoded,
                    past_key_values=cache,
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens[row["task"]],
                    pad_token_id=pad_token_id,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            prediction_ids = generated[0, encoded["input_ids"].shape[-1] :]
            raw_prediction = tokenizer.decode(prediction_ids, skip_special_tokens=True)
            prediction = raw_prediction.strip()
            if args.method == "full":
                cache_bytes = _dynamic_cache_bytes(cache)
                full_bytes = cache_bytes
                quantizer_bytes = 0
            else:
                stats = cache.storage_stats()
                cache_bytes = stats.compressed_bytes
                full_bytes = stats.full_precision_bytes
                quantizer_bytes = cache.quantizer_state_bytes()
            cache_tokens = cache.get_seq_length()
            gpu_type = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
            result = {
                "method": args.method,
                "bits": (
                    torch.empty((), dtype=dtype).element_size() * 8
                    if args.method == "full"
                    else adaptive_allocation["summary"]["average_nominal_bits"]
                    if adaptive_allocation is not None
                    else args.bits
                ),
                "qjl": None if args.method == "adaptive" else args.method == "turboquant",
                "task": row["task"],
                "record_id": row["record_id"],
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "answers": row["answers"],
                "all_classes": row.get("all_classes", []),
                "length": row.get("length"),
                "metric": row["metric"],
                "score": None,
                "input_tokens": int(encoded["input_ids"].shape[-1]),
                "generated_tokens": int(prediction_ids.numel()),
                "cache_tokens": cache_tokens,
                "measured_kv_bytes": cache_bytes,
                "measured_kv_bytes_per_token": cache_bytes / cache_tokens if cache_tokens else 0.0,
                "persistent_storage_ratio": cache_bytes / full_bytes if full_bytes else 0.0,
                "quantizer_state_bytes": quantizer_bytes,
                "persistent_cache_plus_quantizer_bytes": cache_bytes + quantizer_bytes,
                "memory_definition": "packed_codes_and_norm_metadata; quantizer_state_reported_separately",
                "full_precision_kv_bytes": full_bytes,
                "elapsed_seconds": elapsed,
                "tokens_per_second": prediction_ids.numel() / elapsed,
                "throughput_definition": "generated_tokens / end_to_end_prefill_and_decode_seconds",
                "gpu_type": gpu_type,
                "model": args.model,
                "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
                "tokenizer_revision": getattr(tokenizer, "_commit_hash", None) or args.revision,
                "seed": args.seed,
                "dtype": str(dtype),
                "dataset_scope": args.dataset_scope,
                "dataset_sha256": specification["input_sha256"],
                "run_fingerprint": run_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(encoded["input_ids"][0].tolist()).encode()
                ).hexdigest(),
                "protocol_version": PROTOCOL_VERSION,
            }
            if adaptive_allocation is not None:
                result["adaptive_config_sha256"] = adaptive_digest
                result["adaptive_qjl_layers"] = adaptive_allocation["summary"]["qjl_layers"]
                result["adaptive_layer_bits"] = [
                    int(layer["bits"]) for layer in adaptive_allocation["layers"]
                ]
            if device.type == "cuda":
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            count += 1
            print(f"{row['task']} {row['record_id'][:8]} {args.method}: {prediction[:80]!r}")


if __name__ == "__main__":
    main()
