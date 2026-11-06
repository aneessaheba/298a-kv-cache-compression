"""Rebuild the processed LongBench dev set used by the harness (mirrors Data_pipeline notebook 02).

The 800-record dev set (processed_longbench_dev.jsonl) is distributed as a zip and
is not in git. This script reproduces it from scratch so a teammate can run the
harness without the zip:

  1. Download zai-org/LongBench `data.zip` from the Hugging Face Hub and read the
     16 English task JSONLs.
  2. Apply the official THUDM/LongBench prompt templates (dataset2prompt.json).
  3. Tokenize with the exact target tokenizer (meta-llama/Llama-3.1-8B-Instruct)
     and middle-truncate to 32K tokens (keep head + tail), matching the manifest.
  4. Keep only the 800 record_ids listed in dev_ids.json (the committed source of
     truth), emitting the same per-record schema the harness reads.

    # needs `huggingface-cli login` (Llama tokenizer + LongBench are gated/large)
    C:\\Python310\\python.exe members/samruddhi/build_dev_data.py

Writes data/processed_longbench/processed_longbench_dev.jsonl at the repo root.
Use --tokenizer to substitute an open tokenizer for a dry run (lengths will then
differ slightly from the committed data, which is noted in the output records).
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
PIPE_DIR = REPO / "Data_pipeline" / "processed_longbench"
OUT_DIR = REPO / "data" / "processed_longbench"
RAW_DIR = REPO / "data" / "longbench_raw"

HF_REPO = "zai-org/LongBench"
PROMPT_URL = "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CONTEXT_CAP_TOKENS = 32_000
PIPELINE_VERSION = "longbench_preprocessing_v1"

# task -> (category, metric); mirrors the manifest task_registry / notebook TASKS
TASK_REGISTRY = {
    "narrativeqa": ("Single-Doc QA", "F1"),
    "qasper": ("Single-Doc QA", "F1"),
    "multifieldqa_en": ("Single-Doc QA", "F1"),
    "hotpotqa": ("Multi-Doc QA", "F1"),
    "2wikimqa": ("Multi-Doc QA", "F1"),
    "musique": ("Multi-Doc QA", "F1"),
    "gov_report": ("Summarization", "ROUGE-L"),
    "qmsum": ("Summarization", "ROUGE-L"),
    "multi_news": ("Summarization", "ROUGE-L"),
    "trec": ("Few-shot", "Classification score"),
    "triviaqa": ("Few-shot", "F1"),
    "samsum": ("Few-shot", "ROUGE-L"),
    "passage_count": ("Synthetic", "Accuracy"),
    "passage_retrieval_en": ("Synthetic", "Accuracy"),
    "lcc": ("Code", "Edit similarity"),
    "repobench-p": ("Code", "Edit similarity"),
}
CODE_TASKS = {"lcc", "repobench-p"}


def clean_text(s):
    if s is None:
        return ""
    return str(s).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_answers(answers):
    if answers is None:
        return []
    if isinstance(answers, str):
        answers = [answers]
    return [clean_text(a) for a in answers if clean_text(a)]


def ensure_raw_files(token):
    """Download and unzip LongBench data.zip to RAW_DIR if not already present."""
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if any(RAW_DIR.glob("*.jsonl")):
        return
    print(f"downloading {HF_REPO} data.zip ...")
    zpath = hf_hub_download(repo_id=HF_REPO, filename="data.zip", repo_type="dataset", token=token)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW_DIR)
    print(f"unzipped raw tasks to {RAW_DIR}")


def load_task(task):
    """Read one task's JSONL; records get their task name from the filename."""
    matches = list(RAW_DIR.rglob(f"{task}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"{task}.jsonl not found under {RAW_DIR}")
    rows = []
    with open(matches[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fetch_prompts():
    resp = requests.get(PROMPT_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def middle_truncate_ids(token_ids, max_tokens):
    if len(token_ids) <= max_tokens:
        return token_ids, False
    head = max_tokens // 2
    tail = max_tokens - head
    return token_ids[:head] + token_ids[-tail:], True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=MODEL_NAME, help="tokenizer id (default: the gated Llama tokenizer)")
    p.add_argument("--token", default=None, help="HF token (else uses the logged-in token)")
    p.add_argument("--limit-per-task", type=int, default=None, help="debug: cap raw rows per task before filtering")
    args = p.parse_args()

    dev_ids = set(json.loads((PIPE_DIR / "dev_ids.json").read_text())["record_ids"])
    print(f"target dev records: {len(dev_ids)}")

    ensure_raw_files(args.token)
    prompts = fetch_prompts()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, token=args.token)
    exact_tokenizer = args.tokenizer == MODEL_NAME

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "processed_longbench_dev.jsonl"
    seen = set()
    n_written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for task, (category, metric) in TASK_REGISTRY.items():
            rows = load_task(task)
            if args.limit_per_task:
                rows = rows[: args.limit_per_task]
            template = prompts.get(task) or prompts.get(task.replace("-", "_")) or "{context}\n\n{input}"
            for rec in rows:
                record_id = clean_text(rec.get("_id"))
                if not record_id or record_id not in dev_ids or record_id in seen:
                    continue
                context = clean_text(rec.get("context"))
                input_text = clean_text(rec.get("input"))
                answers = normalize_answers(rec.get("answers"))
                if not context or not answers:
                    continue
                language = clean_text(rec.get("language")).lower() or ("code" if task in CODE_TASKS else "en")
                all_classes = rec.get("all_classes")
                dataset = clean_text(rec.get("dataset")) or task

                prompt = template.format(context=context, input=input_text)
                src_words = len(context.split()) + len(input_text.split())

                ids_before = tok(prompt, add_special_tokens=False).input_ids
                trunc_ids, truncated = middle_truncate_ids(ids_before, CONTEXT_CAP_TOKENS)
                if truncated:
                    prompt = tok.decode(trunc_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    # Keep context aligned with the (truncated) prompt tail logic in the harness:
                    # the harness only needs the last 200 chars of `context` to appear in `prompt`,
                    # which middle truncation preserves because the tail is kept.

                record = {
                    "record_id": record_id,
                    "task": task,
                    "category": category,
                    "metric": metric,
                    "split": "dev",
                    "language": language,
                    "dataset": dataset,
                    "input": input_text,
                    "context": context,
                    "answers": answers,
                    "all_classes": all_classes,
                    "prompt": prompt,
                    "source_length_words": src_words,
                    "prompt_tokens_before": len(ids_before),
                    "prompt_tokens_after": len(trunc_ids),
                    "truncated": truncated,
                    "truncation_method": "middle_keep_head_tail",
                    "pipeline_version": PIPELINE_VERSION,
                    "tokenizer_used": args.tokenizer,
                    "exact_tokenizer": exact_tokenizer,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(record_id)
                n_written += 1

    print(f"wrote {n_written} records to {out_path}")
    missing = dev_ids - seen
    if missing:
        print(f"WARNING: {len(missing)} dev ids not found in the raw data "
              f"(first few: {list(missing)[:3]})")
    if not exact_tokenizer:
        print("NOTE: a non-Llama tokenizer was used, so prompt_tokens_* and truncation "
              "may differ slightly from the committed data.")


if __name__ == "__main__":
    main()
