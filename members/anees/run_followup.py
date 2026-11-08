"""Follow-up questions on one compressed cache (N3, SAM-98).

Some LongBench documents come with two or more questions (narrativeqa, qasper,
multifieldqa_en and qmsum: 125 documents in the full processed set). For each,
the document is compressed once and the first two questions are answered from
the same cache, in two settings:

- `doc`: only the document is compressed, as in the main LongBench runs.
- `q1`: the first question is appended to the document before compressing, so a
  question-aware method such as SnapKV picks tokens for it. The second question
  then has to make do with what was kept for the first.

RestoreKV and KVzip score tokens without the question, so they should hold up on
the second question, while SnapKV may not.

    python members/anees/run_followup.py --press restorekv --compression-ratio 0.9

Results go to results/restorekv/followup/<model>_<press>_<ratio>.jsonl, one line
per document and setting; rerunning skips documents already done.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import pipeline

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "harness"))

import longbench  # noqa: E402
from run_longbench import make_press, pick_device  # noqa: E402

DEFAULT_DATA = REPO / "data" / "processed_longbench" / "processed_longbench.jsonl"
OUT_DIR = REPO / "results" / "restorekv" / "followup"
TASKS = {"narrativeqa", "qasper", "multifieldqa_en", "qmsum"}
FOLLOW_UP = "\n\nNow answer a different question about the same text.\n\n"


def question_pairs(path, limit=None, max_prompt_tokens=None):
    """The first two questions of every document that has at least two, in file order."""
    by_doc = defaultdict(list)
    for record in longbench.load_records(path, tasks=TASKS, max_prompt_tokens=max_prompt_tokens):
        key = (record["task"], hashlib.md5(record["context"].encode()).hexdigest())
        by_doc[key].append(record)
    pairs = [records[:2] for records in by_doc.values() if len(records) >= 2]
    return pairs[:limit] if limit else pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--press", default="none", help="none, snapkv, kvzip or restorekv")
    parser.add_argument("--compression-ratio", type=float, default=0.9)
    parser.add_argument("--limit", type=int, help="first N documents only, for quick checks")
    parser.add_argument("--max-prompt-tokens", type=int, help="skip longer records, for quick runs on a laptop")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    pairs = question_pairs(args.data, args.limit, args.max_prompt_tokens)
    ratio = args.compression_ratio if args.press != "none" else 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model.split('/')[-1]}_{args.press}_{ratio}.jsonl"
    done = set()
    if out_file.exists():
        done = {(row["doc_id"], row["setting"]) for row in map(json.loads, out_file.open())}

    device = pick_device()
    print(f"device={device}  press={args.press}  ratio={ratio}  documents={len(pairs)} ({len(done) // 2} done)")
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
    press = make_press(args.press, ratio)
    if args.press.startswith("restorekv"):
        press.post_init_from_model(pipe.model)  # load the adapter on the model's device
        pipe.model.to(device)

    with out_file.open("a") as out:
        for i, (first, second) in enumerate(pairs, 1):
            document, q1 = longbench.split_prompt(first)
            _, q2 = longbench.split_prompt(second)
            task = first["task"]
            for setting in ("doc", "q1"):
                if (first["record_id"], setting) in done:
                    continue
                if setting == "doc":
                    context, questions = document, [q1, q2]
                else:
                    context, questions = document + q1, ["", FOLLOW_UP + q2]
                start = time.perf_counter()
                answers = pipe(context, questions=questions, press=press,
                               max_new_tokens=longbench.MAX_NEW_TOKENS[task])["answers"]
                scores = [longbench.score(task, a, r["answers"], r["all_classes"]) for a, r in zip(answers, (first, second))]
                row = {
                    "doc_id": first["record_id"],
                    "second_id": second["record_id"],
                    "task": task,
                    "setting": setting,
                    "press": args.press,
                    "compression_ratio": ratio,
                    "prompt_tokens": first["prompt_tokens_after"],
                    "answers": answers,
                    "scores": scores,
                    "seconds": round(time.perf_counter() - start, 2),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"[{i}/{len(pairs)}] {task:16s} {setting:3s} q1={scores[0]:.3f} q2={scores[1]:.3f} {row['seconds']:6.1f}s")
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
