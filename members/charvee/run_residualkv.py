"""ResidualKV (formerly DeltaKV, arXiv 2602.08005) on the shared LongBench dev set.

Runs the authors' code on the same 800 records, prompts and scorer as every other
method, and writes results in the shared format so they compare directly with the
full-cache baseline in results/longbench/. Settings follow Charvee's configs.

Setup on the lab machine, nothing to install:

    git clone https://github.com/CURRENTF/DeltaKV ../DeltaKV
    git -C ../DeltaKV checkout 11a837b21acfa475957210c5b2d339b89c1b7e3b

    # quick check, 2 records per task (about 10 minutes)
    python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2

    # full runs, 0.75 and 0.9 first
    python members/charvee/run_residualkv.py --compression-ratio 0.9
    python members/charvee/run_residualkv.py --compression-ratio 0.75

Predictions go to results/residualkv/longbench/<model>_residualkv_<ratio>.jsonl;
rerunning the same command skips records already there.

Memory: ResidualKV keeps 5 layers whole (4-bit KIVI) and stores compressed
residuals for the tokens it drops, so `--compression-ratio 0.9` does not mean a 10%
cache. Each record therefore logs the bytes actually held in its KV cache after
generation (`kv_cache_gb`), next to the bf16 size of a full cache for the same
tokens (`full_kv_cache_gb`). Compare methods on those, not on the ratio.
"""

import argparse
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "harness"))

import longbench  # noqa: E402

DEFAULT_DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"
OUT_DIR = REPO / "results" / "residualkv" / "longbench"
COMPRESSOR = "JitaiHao/Llama-3.1-8B-Instruct-Compressor"

# Charvee's configs (llama31_8b_residualkv_cr*.json); only the keep share changes with the ratio.
BASE_CONFIG = {
    "sparse_method": "delta_compressed_quant_kivi_full_fp8_ref",
    "hf_prefill_chunk_size": 2048000,
    "full_attention_layers": "0,1,2,8,18",
    "sink_keep_tokens": 8,
    "recent_keep_tokens": 128,
    "deltakv_latent_dim": 512,
    "deltakv_center_ratio": 0.1,
    "stride_alpha": 0.001,
    "deltakv_neighbor_count": 4,
    "deltakv_latent_quant_bits": 4,
    "full_layer_kv_quant_bits": 4,
    "full_layer_kivi_group_size": 32,
    "full_layer_kivi_residual_length": 32,
    "cluster_metric": "l2",
    "use_cluster": True,
    "use_compression": True,
    "chunk_prefill_accel_omnikv": False,
    "pool_kernel_size": 1,
}


def import_deltakv(deltakv_repo):
    """Put the DeltaKV source on the path; fall back to the standard logger if loguru is missing."""
    src = Path(deltakv_repo).resolve() / "src"
    if not (src / "deltakv").is_dir():
        sys.exit(f"DeltaKV source not found at {src}; clone it first (see the top of this file)")
    sys.path.insert(0, str(src))
    try:
        import loguru  # noqa: F401
    except ImportError:
        shim = types.ModuleType("loguru")
        shim.logger = logging.getLogger("deltakv")
        sys.modules["loguru"] = shim
    # The lab image has no flash-attn; DeltaKV reads this to pick the attention kernel.
    os.environ.setdefault("DELTAKV_HF_ATTN_IMPLEMENTATION", "sdpa")
    from deltakv.get_chat_api import get_generate_api
    return get_generate_api


def tensor_bytes(obj, seen=None):
    """Bytes of all tensors reachable from a cache object, counting shared storage once."""
    seen = set() if seen is None else seen
    if torch.is_tensor(obj):
        key = (obj.untyped_storage().data_ptr(), obj.device)
        if key in seen:
            return 0
        seen.add(key)
        return obj.untyped_storage().nbytes()
    if isinstance(obj, torch.nn.Module) or id(obj) in seen:
        return 0  # model weights, not cache
    seen.add(id(obj))
    if isinstance(obj, dict):
        return sum(tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(tensor_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(tensor_bytes(v, seen) for v in vars(obj).values())
    return 0


def build_prompt(tokenizer, record):
    """The same prompt the shared harness gives the model."""
    if record["task"] in longbench.NO_CHAT_TEMPLATE_TASKS:
        return tokenizer.bos_token + record["prompt"]
    return tokenizer.apply_chat_template([{"role": "user", "content": record["prompt"]}],
                                         add_generation_prompt=True, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--compressor", default=COMPRESSOR)
    parser.add_argument("--deltakv-repo", default=str(REPO.parent / "DeltaKV"))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--compression-ratio", type=float, default=0.9,
                        help="share of tokens ResidualKV drops from the 27 compressed layers")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--per-task", type=int)
    args = parser.parse_args()

    tasks = set(args.tasks.split(",")) if args.tasks else None
    records = longbench.load_records(args.data, tasks=tasks, per_task=args.per_task)
    ratio = args.compression_ratio
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{args.model.split('/')[-1]}_residualkv_{ratio}.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(line)["record_id"] for line in out_file.open()}
    todo = [r for r in records if r["record_id"] not in done]
    print(f"residualkv ratio={ratio} records={len(records)} ({len(done)} already done)", flush=True)

    get_generate_api = import_deltakv(args.deltakv_repo)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(args.model)
    compressor_path = snapshot_download(args.compressor)
    config = dict(BASE_CONFIG, decode_keep_tokens=1 - ratio, prefill_keep_tokens=1 - ratio)
    generate, model = get_generate_api(model_path, config, deltakv_checkpoint_path=compressor_path,
                                       tokenizer_path=model_path, backend="hf",
                                       return_kv_cache=True, return_model=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    bf16_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2

    with out_file.open("a") as out:
        for i, record in enumerate(todo, 1):
            prompt = build_prompt(tokenizer, record)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                prediction, cache = generate(prompt, max_new_tokens=longbench.MAX_NEW_TOKENS[record["task"]],
                                             do_sample=False)
                oom = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                prediction, cache, oom = "", None, True
            seconds = time.perf_counter() - start
            tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            row = {
                "record_id": record["record_id"],
                "task": record["task"],
                "press": "residualkv",
                "compression_ratio": ratio,
                "prompt_tokens": record["prompt_tokens_after"],
                "prediction": prediction,
                "score": 0.0 if oom else longbench.score(record["task"], prediction,
                                                         record["answers"], record["all_classes"]),
                "oom": oom,
                "seconds": round(seconds, 2),
                "kv_cache_gb": round(tensor_bytes(cache) / 1e9, 4) if cache is not None else None,
                "full_kv_cache_gb": round(tokens * bf16_bytes_per_token / 1e9, 4),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
            del cache
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {record['task']:20s} score={row['score']:.3f} {row['seconds']:6.1f}s "
                  f"kv={row['kv_cache_gb']} GB of {row['full_kv_cache_gb']} GB", flush=True)

    rows = [json.loads(line) for line in out_file.open()]
    rows = [r for r in rows if r["record_id"] in {rec["record_id"] for rec in records}]
    summary = longbench.summarize(rows)
    out_file.with_suffix(".summary.json").write_text(json.dumps(
        {"args": vars(args), "config": config, "n_records": len(rows), "scores": summary}, indent=2))
    for task, value in summary.items():
        print(f"{task:22s} {value:6.2f}")
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
