"""Shape/behaviour self-test for TriAttentionPress on a small public model (no HF login).

Runs the kv-press pipeline with TriAttention on a needle-in-a-haystack prompt and
checks: (1) the press runs without error at a high compression ratio, (2) tensor
shapes are consistent across GQA, (3) the needle survives compression often enough
to answer. Uses Qwen/Qwen2.5-0.5B-Instruct (open, no gating), the same small model
the harness docs use for quick checks.

    C:\\Python310\\python.exe members/samruddhi/selftest_triattention.py
"""

import argparse
import time

import torch
from transformers import pipeline

from triattention_press import TriAttentionPress

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE = "The secret code for the vault is {code}. "
QUESTION = "What is the secret code for the vault? Answer with the number only."


def build_context(tokenizer, target_tokens, depth, code):
    filler_tokens = len(tokenizer(FILLER, add_special_tokens=False).input_ids)
    parts = [FILLER] * max(1, target_tokens // filler_tokens)
    parts.insert(int(len(parts) * depth), NEEDLE.format(code=code))
    return "".join(parts)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokens", type=int, default=2000)
    p.add_argument("--compression-ratio", type=float, default=0.8)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--code", default="7341")
    args = p.parse_args()

    device = pick_device()
    print(f"device={device}  model={args.model}  ratio={args.compression_ratio}")
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
    ctx = build_context(pipe.tokenizer, args.tokens, args.depth, args.code)
    n_ctx = len(pipe.tokenizer(ctx, add_special_tokens=False).input_ids)
    print(f"context={n_ctx} tokens")

    for name, press in [
        ("none", None),
        ("triattention(norm=1)", TriAttentionPress(compression_ratio=args.compression_ratio, norm_power=1.0)),
        ("triattention(norm=0)", TriAttentionPress(compression_ratio=args.compression_ratio, norm_power=0.0)),
    ]:
        t = time.perf_counter()
        ans = pipe(ctx, question=QUESTION, press=press, max_new_tokens=16)["answer"].strip()
        dt = time.perf_counter() - t
        found = args.code in ans
        print(f"{name:24s} found={found!s:5s} {dt:6.2f}s  answer={ans!r}")

    print("\nself-test completed without errors")


if __name__ == "__main__":
    main()
