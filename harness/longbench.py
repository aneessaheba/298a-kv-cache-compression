"""Load the processed LongBench records and score predictions with the official metrics.

The records come from Data_pipeline/02_LongBench_Data_Preprocessing.ipynb
(processed_longbench_dev.jsonl: 800 records, 50 per task, Llama-3.1-8B-Instruct
tokenizer, 32K cap). Metrics follow THUDM/LongBench metrics.py and eval.py.
"""

import json
import re
import string
from collections import Counter, defaultdict

from rapidfuzz import fuzz
from rouge import Rouge

# Generation length per task, from LongBench config/dataset2maxlen.json
MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
    "2wikimqa": 32, "musique": 32, "gov_report": 512, "qmsum": 512,
    "multi_news": 512, "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64,
}

# Few-shot tasks whose answer is the first line of the output (LongBench eval.py)
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum"}

# Few-shot and code-completion tasks that LongBench pred.py runs without the chat template
NO_CHAT_TEMPLATE_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


def load_records(path, tasks=None, per_task=None, max_prompt_tokens=None):
    """Read records from a processed LongBench jsonl, optionally keeping the first `per_task` of each task."""
    counts = Counter()
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            task = record["task"]
            if tasks and task not in tasks:
                continue
            if max_prompt_tokens and record["prompt_tokens_after"] > max_prompt_tokens:
                continue
            if per_task is not None and counts[task] >= per_task:
                continue
            counts[task] += 1
            records.append(record)
    return records


def split_prompt(record):
    """Split the full prompt into the part a press may compress and the part it must keep.

    Everything up to the end of the document (`context`) is compressible; the
    rest of the template (instruction, question, answer cue) is passed as the
    question so that every method sees it uncompressed. The prompt may be
    truncated in the middle, but the tail of the context always survives.
    """
    prompt = record["prompt"]
    tail = record["context"][-200:].strip()
    cut = prompt.rfind(tail) + len(tail) if tail else -1
    if cut < len(tail):
        raise ValueError(f"context tail not found in prompt for {record['record_id']}")
    return prompt[:cut], prompt[cut:]


def split_answer_cue(question):
    """Split the question so that kvpress's no-template join (question + "\\n" + answer_prefix) rebuilds it exactly.

    Without a chat template kvpress always inserts a newline before the answer
    prefix. Left in place, "...Type:" becomes "...Type:\\n" and the model starts
    a new few-shot example instead of answering, so the last line (the answer
    cue) is passed as the answer prefix instead.
    """
    if question.endswith("\n"):
        return question[:-1], ""
    head, _, cue = question.rpartition("\n")
    return head, cue


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1_score(prediction, ground_truth, **kwargs):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    num_same = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction, ground_truth, **kwargs):
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:  # empty or punctuation-only prediction
        return 0.0


def classification_score(prediction, ground_truth, all_classes, **kwargs):
    matches = [c for c in all_classes if c in prediction]
    # Drop classes that only matched as a substring of the true label. Removing
    # while iterating is kept as in LongBench so scores match the official ones.
    for m in matches:
        if m in ground_truth and m != ground_truth:
            matches.remove(m)
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    target = re.findall(r"Paragraph (\d+)", ground_truth)[0]
    numbers = re.findall(r"\d+", prediction)
    return numbers.count(target) / len(numbers) if numbers else 0.0


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    return numbers.count(str(ground_truth)) / len(numbers) if numbers else 0.0


def code_sim_score(prediction, ground_truth, **kwargs):
    # Score the first line that is not a code fence or comment
    first = next((line for line in prediction.lstrip("\n").split("\n")
                  if "`" not in line and "#" not in line and "//" not in line), "")
    return fuzz.ratio(first, ground_truth) / 100


METRICS = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "gov_report": rouge_score, "qmsum": rouge_score, "multi_news": rouge_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score, "passage_count": count_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


def score(task, prediction, answers, all_classes=None):
    """Best score over the reference answers, in [0, 1]."""
    if task in FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    metric = METRICS[task]
    return max(metric(prediction, gt, all_classes=all_classes) for gt in answers)


def summarize(rows):
    """Per-task mean score (0-100) plus the macro average over tasks."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row["score"])
    summary = {task: round(100 * sum(s) / len(s), 2) for task, s in by_task.items()}
    summary["average"] = round(sum(summary.values()) / len(summary), 2) if summary else 0.0
    return summary
