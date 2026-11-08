"""Extract real Llama KV tensors for the real-distribution TurboQuant check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    source.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/real_kv/llama_kv.safetensors"))
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--token")
    args = parser.parse_args()

    device = _device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type in {"cpu", "mps"} and dtype == torch.float16:
        dtype = torch.float32
    if args.prompt is not None:
        prompt = args.prompt
        source_record_id = None
    elif args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
        source_record_id = None
    else:
        rows = [
            json.loads(line)
            for line in args.prompt_jsonl.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not 0 <= args.record_index < len(rows):
            raise SystemExit(f"--record-index must be between 0 and {len(rows) - 1}")
        prompt = rows[args.record_index]["prompt"]
        source_record_id = rows[args.record_index].get("record_id")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        token=args.token,
        attn_implementation="sdpa",
    ).to(device).eval()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        model(**encoded, past_key_values=cache, use_cache=True, logits_to_keep=1)
    tensors = {}
    for layer_idx, layer in enumerate(cache.layers):
        tensors[f"layer_{layer_idx}.keys"] = layer.keys.detach().cpu().contiguous()
        tensors[f"layer_{layer_idx}.values"] = layer.values.detach().cpu().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    metadata = {
        "model": args.model,
        "input_tokens": int(encoded["input_ids"].shape[-1]),
        "layers": len(cache.layers),
        "tensor_names": sorted(tensors),
        "dtype": str(dtype),
        "device": str(device),
        "source_record_id": source_record_id,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
