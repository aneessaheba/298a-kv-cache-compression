"""Run a model, with or without a KV-cache press, on RULER (SAM-94).

Uses the RULER build that kvpress evaluates on (simonjegou/ruler on Hugging Face:
13 tasks x 500 examples at 4K, 8K or 16K tokens) and kvpress's scoring:
string_match_part for the qa_* tasks, string_match_all for the rest.
Predictions go to results/ruler/<model>_<length>_<press>_<ratio>.jsonl; rerunning
the same command skips examples that are already there.

    # SAM-94: RestoreKV vs KVzip at a 5% budget, 4K context, 50 examples per task
    python harness/run_ruler.py --press kvzip --compression-ratio 0.95 --per-task 50
    python harness/run_ruler.py --press restorekv --compression-ratio 0.95 --per-task 50
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import pipeline

from run_longbench import kv_cache_gb, make_press, pick_device

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "ruler"
CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def ruler_score(task, prediction, answers):
    """Fraction of reference strings found in the prediction (any one for qa_* tasks), in [0, 1]."""
    prediction = CONTROL_CHARS.sub("", prediction.strip()).strip().lower()
    hits = [answer.lower() in prediction for answer in answers]
    if task.startswith("qa"):
        return float(any(hits))
    return sum(hits) / len(hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", default="4096", choices=["4096", "8192", "16384"])
    parser.add_argument("--press", default="none", help="none, snapkv, kvzip, restorekv, restorekv_4bit or restorekv_2bit")
    parser.add_argument("--compression-ratio", type=float, default=0.0, help="fraction of the KV cache removed")
    parser.add_argument("--per-task", type=int, default=None, help="examples per task (default: all 500)")
    args = parser.parse_args()

    data = load_dataset("simonjegou/ruler", args.length, split="test")
    counts = Counter()
    examples = []
    for idx, example in enumerate(data):
        if args.per_task is not None and counts[example["task"]] >= args.per_task:
            continue
        counts[example["task"]] += 1
        examples.append({"id": idx, **example})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ratio = args.compression_ratio if args.press != "none" else 0.0
    out_file = OUT_DIR / f"{args.model.split('/')[-1]}_{args.length}_{args.press}_{ratio}.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(line)["id"] for line in out_file.open()}
    todo = [e for e in examples if e["id"] not in done]

    device = pick_device()
    print(f"device={device}  model={args.model}  length={args.length}  press={args.press}  ratio={ratio}  "
          f"examples={len(examples)} ({len(done)} already done)")
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
    press = make_press(args.press, ratio)
    if args.press.startswith("restorekv"):
        # Load the LoRA adapter now so it lands on the model's device (see members/anees/smoke_test.py)
        press.post_init_from_model(pipe.model)
        pipe.model.to(device)

    with out_file.open("a") as out:
        for i, example in enumerate(todo, 1):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            prediction = pipe(example["context"], question=example["question"],
                              answer_prefix=example["answer_prefix"], press=press,
                              max_new_tokens=example["max_new_tokens"])["answer"]
            seconds = time.perf_counter() - start
            row = {
                "id": example["id"],
                "task": example["task"],
                "press": args.press,
                "compression_ratio": ratio,
                "prediction": prediction,
                "answers": example["answer"],
                "score": ruler_score(example["task"], prediction, example["answer"]),
                "seconds": round(seconds, 2),
                "kv_cache_gb": kv_cache_gb(pipe, example["context"],
                                           example["question"] + example["answer_prefix"], ratio),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {example['task']:18s} score={row['score']:.2f}  {row['seconds']:6.1f}s")

    wanted = {e["id"] for e in examples}
    by_task = defaultdict(list)
    for line in out_file.open():
        row = json.loads(line)
        if row["id"] in wanted:
            by_task[row["task"]].append(row["score"])
    summary = {task: round(100 * sum(s) / len(s), 2) for task, s in sorted(by_task.items())}
    summary["average"] = round(sum(summary.values()) / len(summary), 2) if summary else 0.0
    summary_file = out_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps({"args": vars(args), "device": device,
                                        "n_examples": sum(len(s) for s in by_task.values()),
                                        "scores": summary}, indent=2))
    for task, value in summary.items():
        print(f"{task:18s} {value:6.2f}")
    print(f"saved {out_file} and {summary_file}")


if __name__ == "__main__":
    main()
