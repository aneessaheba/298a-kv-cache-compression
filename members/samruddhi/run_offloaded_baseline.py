"""Full, uncompressed KV-cache baseline on all 800 LongBench records via CPU offloading.

A TRUE naive/full-cache baseline for Llama-3.1-8B-Instruct: the COMPLETE KV cache is kept
(no eviction, pruning, quantization, or TriAttention). GPU OOM is avoided ONLY by storing
the full KV cache in CPU RAM and paging each layer to the GPU when it is computed.

Uses the native transformers path with a PLAIN model (no kvpress patch, which is incompatible
with offloading):
    model.generate(..., cache_implementation="offloaded", prefill_chunk_size=N)
- cache_implementation="offloaded" -> DynamicCache(offloading=True): full cache, CPU-resident,
  per-layer async prefetch/offload.
- prefill_chunk_size -> chunked prefill, so no full-sequence attention matrix is built.
If a record still OOMs, prefill_chunk_size is reduced and the record retried (never by
dropping cache).

Validation (first): short records normal-cache vs offloaded-cache must match; then one long
record that previously OOMed.

Outputs: experiment2_full_800/offloaded_full_baseline/
  per_record.csv, task_summary.csv, overall.json, oom_report.csv, memory_runtime.json

    C:\\Python310\\python.exe members/samruddhi/run_offloaded_baseline.py --validate-only
    C:\\Python310\\python.exe members/samruddhi/run_offloaded_baseline.py
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(HERE))
import longbench  # noqa: E402

DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"
OUT_DIR = HERE / "experiment2_full_800" / "offloaded_full_baseline"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def cpu_ram_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return None


def build_prompt(tokenizer, record):
    """Reproduce the pipeline's prompt: chat template for most tasks, raw+BOS for no-template tasks."""
    prompt = record["prompt"]
    no_template = record["task"] in longbench.NO_CHAT_TEMPLATE_TASKS and tokenizer.bos_token
    if no_template or tokenizer.chat_template is None:
        text = (tokenizer.bos_token or "") + prompt
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    return text


@torch.no_grad()
def generate_offloaded(model, tokenizer, record, chunk_size, max_new_tokens, offloaded=True):
    text = build_prompt(tokenizer, record)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    n_in = input_ids.shape[1]
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                      pad_token_id=tokenizer.eos_token_id)
    if offloaded:
        gen_kwargs["cache_implementation"] = "offloaded"
        gen_kwargs["prefill_chunk_size"] = chunk_size
    out = model.generate(input_ids, **gen_kwargs)
    answer = tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
    return answer, n_in


def run_record(model, tokenizer, record, chunk_sizes):
    max_new = longbench.MAX_NEW_TOKENS[record["task"]]
    last = None
    for cs in chunk_sizes:
        if torch.cuda.is_available():
            import gc
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            answer, n_in = generate_offloaded(model, tokenizer, record, cs, max_new, offloaded=True)
            seconds = time.perf_counter() - start
            score = longbench.score(record["task"], answer, record["answers"], record["all_classes"])
            return {
                "task": record["task"], "record_id": record["record_id"],
                "input_tokens": n_in, "answer": answer,
                "reference": record["answers"][0] if record["answers"] else "",
                "score": round(score, 4),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
                "cpu_ram_gb": cpu_ram_gb(), "seconds": round(seconds, 2),
                "chunk_size": cs, "status": "ok", "oom_source": None,
            }
        except torch.cuda.OutOfMemoryError as e:
            last = str(e)[:100]
            import gc
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            continue
    return {
        "task": record["task"], "record_id": record["record_id"],
        "input_tokens": record["prompt_tokens_after"], "answer": "",
        "reference": record["answers"][0] if record["answers"] else "",
        "score": None, "peak_gpu_gb": None, "cpu_ram_gb": cpu_ram_gb(), "seconds": None,
        "chunk_size": chunk_sizes[-1], "status": "oom",
        "oom_source": f"prefill/attention temp memory even at smallest chunk ({last})",
    }


def validate(model, tokenizer, records, chunk_sizes):
    print("\n=== VALIDATION ===", flush=True)
    shorts = sorted([r for r in records if r["prompt_tokens_after"] < 3000],
                    key=lambda r: r["prompt_tokens_after"])[:3]
    ok = True
    for r in shorts:
        max_new = longbench.MAX_NEW_TOKENS[r["task"]]
        a_norm, _ = generate_offloaded(model, tokenizer, r, chunk_sizes[0], max_new, offloaded=False)
        a_off, _ = generate_offloaded(model, tokenizer, r, chunk_sizes[0], max_new, offloaded=True)
        s_norm = longbench.score(r["task"], a_norm, r["answers"], r["all_classes"])
        s_off = longbench.score(r["task"], a_off, r["answers"], r["all_classes"])
        match = abs(s_norm - s_off) < 0.02
        ok = ok and match
        print(f"  {r['task']:16s} {r['prompt_tokens_after']:5d}tok  normal={s_norm:.3f}  "
              f"offloaded={s_off:.3f}  {'MATCH' if match else 'MISMATCH'}", flush=True)
    longest = sorted(records, key=lambda r: -r["prompt_tokens_after"])[0]
    res = run_record(model, tokenizer, longest, chunk_sizes)
    print(f"  LONG {longest['task']:12s} {longest['prompt_tokens_after']:5d}tok  status={res['status']}  "
          f"score={res['score']}  peak_gpu={res['peak_gpu_gb']}GB", flush=True)
    ok = ok and (res["status"] == "ok")
    print(f"=== VALIDATION {'PASSED' if ok else 'FAILED'} ===\n", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--chunk-sizes", default="1024,512,256,128")
    args = ap.parse_args()
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]

    records = longbench.load_records(str(DATA))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={MODEL}  records={len(records)}  chunk_sizes={chunk_sizes}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    if not validate(model, tokenizer, records, chunk_sizes):
        print("Validation failed; aborting.", flush=True)
        return
    if args.validate_only:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_record_path = OUT_DIR / "per_record.csv"
    done = set()
    if per_record_path.exists():
        with per_record_path.open(encoding="utf-8") as f:
            done = {row["record_id"] for row in csv.DictReader(f)}
    fields = ["task", "record_id", "input_tokens", "answer", "reference", "score",
              "peak_gpu_gb", "cpu_ram_gb", "seconds", "chunk_size", "status", "oom_source"]
    new_file = not per_record_path.exists()
    with per_record_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        todo = [r for r in records if r["record_id"] not in done]
        for i, r in enumerate(todo, 1):
            res = run_record(model, tokenizer, r, chunk_sizes)
            w.writerow(res); f.flush()
            print(f"[{i}/{len(todo)}] {res['task']:18s} {res['status']:3s} score={res['score']} "
                  f"{res['input_tokens']}tok peak={res['peak_gpu_gb']}GB {res['seconds']}s", flush=True)

    summarize(per_record_path)


def summarize(per_record_path):
    from collections import defaultdict
    rows = list(csv.DictReader(per_record_path.open(encoding="utf-8")))
    ok_rows = [r for r in rows if r["status"] == "ok" and r["score"] not in ("", "None", None)]
    by_task = defaultdict(list)
    for r in ok_rows:
        by_task[r["task"]].append(float(r["score"]))
    task_summary = {t: round(100 * sum(v) / len(v), 2) for t, v in sorted(by_task.items())}
    overall = round(sum(task_summary.values()) / len(task_summary), 2) if task_summary else 0.0

    with (OUT_DIR / "task_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task", "mean_score", "n"])
        for t in sorted(by_task):
            w.writerow([t, task_summary[t], len(by_task[t])])
        w.writerow(["AVERAGE", overall, len(ok_rows)])

    (OUT_DIR / "overall.json").write_text(json.dumps({
        "model": MODEL, "method": "full_cache_cpu_offloaded", "compression": "none",
        "n_total": len(rows), "n_ok": len(ok_rows), "n_failed": len(rows) - len(ok_rows),
        "overall_macro_avg": overall, "task_scores": task_summary,
    }, indent=2), encoding="utf-8")

    failed = [r for r in rows if r["status"] != "ok"]
    with (OUT_DIR / "oom_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task", "record_id", "input_tokens", "status", "oom_source"])
        for r in failed:
            w.writerow([r["task"], r["record_id"], r["input_tokens"], r["status"], r["oom_source"]])

    peaks = [float(r["peak_gpu_gb"]) for r in ok_rows if r["peak_gpu_gb"] not in ("", "None", None)]
    secs = [float(r["seconds"]) for r in ok_rows if r["seconds"] not in ("", "None", None)]
    rams = [float(r["cpu_ram_gb"]) for r in rows if r["cpu_ram_gb"] not in ("", "None", None)]
    (OUT_DIR / "memory_runtime.json").write_text(json.dumps({
        "peak_gpu_gb_max": max(peaks) if peaks else None,
        "peak_gpu_gb_mean": round(sum(peaks) / len(peaks), 2) if peaks else None,
        "cpu_ram_gb_max": max(rams) if rams else None,
        "mean_seconds": round(sum(secs) / len(secs), 2) if secs else None,
        "total_minutes": round(sum(secs) / 60, 1) if secs else None,
    }, indent=2), encoding="utf-8")
    print(f"\noverall macro avg: {overall}  ({len(ok_rows)}/{len(rows)} ok, {len(failed)} failed)", flush=True)
    print(f"saved summaries to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
