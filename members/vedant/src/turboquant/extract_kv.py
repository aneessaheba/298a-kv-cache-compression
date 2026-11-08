"""Extract real Llama KV tensors for the real-distribution TurboQuant check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def attach_query_hooks(model, tensors):
    """Capture actual Llama queries after RoPE, without averaging away GQA heads."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    if model.config.model_type != "llama":
        raise ValueError("real-query capture currently supports Llama models only")

    def capture(module, args, kwargs):
        hidden = kwargs["hidden_states"]
        q = module.q_proj(hidden).view(*hidden.shape[:-1], -1, module.head_dim).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
        tensors[f"layer_{module.layer_idx}.queries"] = rotated.detach().cpu().contiguous()

    return [layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
            for layer in model.model.layers]


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
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--token")
    parser.add_argument("--save-queries", action="store_true")
    parser.add_argument("--shared-prompt", action="store_true", help="use the team's kvpress preprocessing for a JSONL record")
    args = parser.parse_args()
    if args.shared_prompt and args.prompt_jsonl is None:
        parser.error("--shared-prompt requires --prompt-jsonl")

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
    if args.shared_prompt:
        repo = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(repo / "harness"))
        import longbench as shared
        from kvpress import KVPressTextGenerationPipeline
        row = rows[args.record_index]
        context, question = shared.split_prompt(row)
        answer_prefix = ""
        if row["task"] in shared.NO_CHAT_TEMPLATE_TASKS and tokenizer.bos_token:
            tokenizer.chat_template = None
            question, answer_prefix = shared.split_answer_cue(question)
        pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
        prepared = pipe.preprocess(context, questions=[question], answer_prefix=answer_prefix,
                                   max_context_length=32000)
        ids = torch.cat([prepared["context_ids"], prepared["questions_ids"][0]], dim=-1)
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Keep both document ends and question/instruction suffix instead of only the beginning.
    original_tokens = ids.shape[-1]
    if original_tokens > args.max_input_tokens:
        left = args.max_input_tokens // 2
        ids = torch.cat([ids[:, :left], ids[:, -(args.max_input_tokens-left):]], dim=-1)
    encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    encoded = {key: value.to(device) for key, value in encoded.items()}
    cache = DynamicCache(config=model.config)
    tensors = {}
    hooks = attach_query_hooks(model, tensors) if args.save_queries else []
    try:
        with torch.inference_mode():
            model(**encoded, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
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
        "query_source": "model_post_rope" if args.save_queries else "not_saved",
        "prompt_format": "shared_kvpress" if args.shared_prompt else "raw_text",
        "original_tokens": original_tokens,
        "window": "middle_keep_head_tail",
        "limitation": "bounded calibration window; position-aligned QK error is a proxy, not full attention-matrix distortion",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
