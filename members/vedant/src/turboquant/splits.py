"""Create deterministic, task-stratified calibration and context subsets."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .protocol import validate_dataset


def make_splits(
    rows: list[dict[str, Any]],
    *,
    calibration_per_task: int = 2,
    context_per_task: int = 2,
    seed: int = 20_260,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    calibration = []
    context = []
    evaluation = []
    for task in sorted(grouped):
        task_rows = sorted(grouped[task], key=lambda row: row["record_id"])
        task_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(task))
        random.Random(task_seed).shuffle(task_rows)
        required = calibration_per_task + context_per_task
        if len(task_rows) < required:
            raise ValueError(f"task {task} has {len(task_rows)} rows but needs at least {required}")
        calibration.extend(task_rows[:calibration_per_task])
        context.extend(task_rows[calibration_per_task:required])
        evaluation.extend(task_rows[calibration_per_task:])
    return {
        "calibration": sorted(calibration, key=lambda row: (row["task"], row["record_id"])),
        "context": sorted(context, key=lambda row: (row["task"], row["record_id"])),
        "evaluation": sorted(evaluation, key=lambda row: (row["task"], row["record_id"])),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/splits"))
    parser.add_argument("--calibration-per-task", type=int, default=2)
    parser.add_argument("--context-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260)
    parser.add_argument("--dataset-scope", choices=["team", "subset"], default="team")
    args = parser.parse_args()
    rows = validate_dataset(args.input, scope=args.dataset_scope)
    splits = make_splits(
        rows,
        calibration_per_task=args.calibration_per_task,
        context_per_task=args.context_per_task,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "calibration": "adaptive_calibration.jsonl",
        "context": "context_scaling.jsonl",
    }
    for key, name in names.items():
        _write_jsonl(args.output_dir / name, splits[key])
    calibration_ids = [row["record_id"] for row in splits["calibration"]]
    manifest = {
        "seed": args.seed,
        "calibration_per_task": args.calibration_per_task,
        "context_per_task": args.context_per_task,
        "tasks": len({row["task"] for row in rows}),
        "source_records": len(rows),
        "calibration_records": len(splits["calibration"]),
        "context_records": len(splits["context"]),
        "evaluation_records": len(splits["evaluation"]),
        "calibration_and_evaluation_disjoint": not bool(
            set(calibration_ids) & {row["record_id"] for row in splits["evaluation"]}
        ),
        "record_ids": calibration_ids,
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
