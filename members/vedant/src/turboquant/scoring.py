"""Official LongBench v1 English metrics with the declared Team 6 whitespace fix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .protocol import sha256

SCORING_POLICY = "longbench-v1-team6-leading-whitespace-v1"
UPSTREAM_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    from .vendor.longbench import metrics

    metric_names = {
        **dict.fromkeys(("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"), "qa_f1_score"),
        **dict.fromkeys(("gov_report", "qmsum", "multi_news", "samsum"), "rouge_score"),
        "trec": "classification_score", "passage_count": "count_score",
        "passage_retrieval_en": "retrieval_score", "lcc": "code_sim_score", "repobench-p": "code_sim_score",
    }
    task = row["task"]
    if task not in metric_names:
        raise ValueError(f"unsupported team task: {task}")
    prediction = row["prediction"]
    if not isinstance(prediction, str):
        raise ValueError("prediction must be text")
    # Upstream uses lstrip('\n'); spaces before newlines otherwise produce a blank first line.
    normalized = prediction.lstrip().split("\n")[0] if task in FIRST_LINE_TASKS else prediction
    answers = row.get("answers")
    if not isinstance(answers, list) or not answers:
        raise ValueError("answers must be a nonempty list")
    if task == "trec" and not row.get("all_classes"):
        raise ValueError("trec requires all_classes; regenerate or join source metadata by record_id")
    fn = getattr(metrics, metric_names[task])
    value = max(fn(normalized, answer, all_classes=row.get("all_classes", [])) for answer in answers)
    return {**row, "score": 100.0 * float(value), "score_scale": "0-100",
            "scoring_policy": SCORING_POLICY, "scoring_upstream_revision": UPSTREAM_REVISION}


def score_file(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("scored output already exists; choose a new path")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("predictions must be nonempty and contain unique record IDs")
    scored = [score_row(row) for row in rows]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        grouped[row["task"]].append(row["score"])
    per_task = {task: sum(values) / len(values) for task, values in sorted(grouped.items())}
    summary = {"records": len(scored), "tasks": len(per_task), "score_scale": "0-100",
               "scoring_policy": SCORING_POLICY, "upstream_revision": UPSTREAM_REVISION,
               "prediction_sha256": sha256(source), "per_task": per_task,
               "longbench_macro_score": sum(per_task.values()) / len(per_task)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(output.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(score_file(args.input, args.output), indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
