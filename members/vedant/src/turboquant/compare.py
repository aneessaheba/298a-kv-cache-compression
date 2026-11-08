"""Build the paired TurboQuant-vs-KIVI R5 table from scored harness JSONL files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .protocol import DATA_DIR, validate_dataset


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_macro_score_drop(
    indexed: dict[str, dict[str, dict[str, Any]]],
    run_name: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for record_id, full_row in indexed["full"].items():
        row = indexed[run_name][record_id]
        by_task[row["task"]].append(float(full_row["score"]) - float(row["score"]))
    observed = sum(sum(values) / len(values) for values in by_task.values()) / len(by_task)
    if samples <= 0:
        return observed, float("nan"), float("nan")
    name_seed = int.from_bytes(hashlib.sha256(run_name.encode()).digest()[:8], "big")
    rng = random.Random(seed ^ name_seed)
    draws = []
    for _ in range(samples):
        task_means = []
        for values in by_task.values():
            task_means.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        draws.append(sum(task_means) / len(task_means))
    return observed, _percentile(draws, 0.025), _percentile(draws, 0.975)


def compare_runs(
    named_paths: dict[str, Path],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20_260,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    runs = {name: _read(path) for name, path in named_paths.items()}
    excluded = exclude_ids or set()
    if excluded:
        runs = {
            name: [row for row in rows if row["record_id"] not in excluded]
            for name, rows in runs.items()
        }
    if "full" not in runs:
        raise ValueError("a run named 'full' is required")
    for name, rows in runs.items():
        record_ids = [row["record_id"] for row in rows]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"{name} contains duplicate record IDs")
    indexed = {
        name: {row["record_id"]: row for row in rows}
        for name, rows in runs.items()
    }
    full_ids = set(indexed["full"])
    if not full_ids:
        raise ValueError("the full run contains no records")
    for name, rows in indexed.items():
        run_ids = set(rows)
        missing = sorted(full_ids - run_ids)
        extra = sorted(run_ids - full_ids)
        if missing or extra:
            raise ValueError(
                f"{name} does not match the full run: "
                f"{len(missing)} missing and {len(extra)} extra record IDs"
            )
        mismatched_tasks = [
            record_id
            for record_id in full_ids
            if rows[record_id]["task"] != indexed["full"][record_id]["task"]
        ]
        if mismatched_tasks:
            raise ValueError(f"{name} has {len(mismatched_tasks)} task-label mismatches")
    common_ids = full_ids
    for name, rows in indexed.items():
        missing_score = [rid for rid in common_ids if rows[rid].get("score") is None]
        if missing_score:
            raise ValueError(
                f"{name} has {len(missing_score)} unscored paired rows; run the shared official evaluator first"
            )
        for rid, row in rows.items():
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f"{name} contains a nonfinite or out-of-range score")
            full_row = indexed["full"][rid]
            for field in ("model", "model_revision", "seed", "dataset_sha256", "input_ids_sha256",
                          "scoring_policy", "score_scale", "dtype", "protocol_version"):
                if row.get(field) is not None and full_row.get(field) is not None and row[field] != full_row[field]:
                    raise ValueError(f"{name} has mismatched {field} for {rid}")

    per_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    paired_rows = []
    for record_id in sorted(common_ids):
        full_row = indexed["full"][record_id]
        for name, rows in indexed.items():
            row = rows[record_id]
            score = float(row["score"])
            full_score = float(full_row["score"])
            full_bpt = full_row.get("measured_kv_bytes_per_token")
            bpt = row.get("measured_kv_bytes_per_token")
            paired = {
                "run": name,
                "method": row.get("method", name),
                "bits": row.get("bits"),
                "task": row["task"],
                "record_id": record_id,
                "score": score,
                "full_score": full_score,
                "score_drop": full_score - score,
                "quality_retained": score / full_score if full_score else None,
                "kv_bytes_per_token": row.get("measured_kv_bytes_per_token"),
                "kv_storage_ratio_vs_full": float(bpt) / float(full_bpt) if bpt is not None and full_bpt else None,
            }
            paired_rows.append(paired)
            per_task[(name, row["task"])].append(paired)

    task_rows = []
    for (name, task), rows in sorted(per_task.items()):
        mean_score = sum(row["score"] for row in rows) / len(rows)
        mean_full = sum(row["full_score"] for row in rows) / len(rows)
        byte_values = [float(row["kv_bytes_per_token"]) for row in rows if row["kv_bytes_per_token"] is not None]
        ratios = [row["kv_storage_ratio_vs_full"] for row in rows if row["kv_storage_ratio_vs_full"] is not None]
        task_rows.append(
            {
                "run": name,
                "method": rows[0]["method"],
                "bits": rows[0]["bits"],
                "task": task,
                "paired_samples": len(rows),
                "mean_score": mean_score,
                "mean_full_score": mean_full,
                "quality_retained": mean_score / mean_full if mean_full else None,
                "mean_kv_bytes_per_token": sum(byte_values) / len(byte_values) if byte_values else None,
                "mean_kv_storage_ratio_vs_full": sum(ratios) / len(ratios) if ratios else None,
            }
        )

    overall_rows = []
    for name in runs:
        rows = [row for row in task_rows if row["run"] == name]
        byte_values = [
            row["mean_kv_bytes_per_token"]
            for row in rows
            if row["mean_kv_bytes_per_token"] is not None
        ]
        score_drop, ci_low, ci_high = _bootstrap_macro_score_drop(
            indexed, name, samples=bootstrap_samples, seed=bootstrap_seed
        )
        ratios = [row["mean_kv_storage_ratio_vs_full"] for row in rows if row["mean_kv_storage_ratio_vs_full"] is not None]
        overall_rows.append(
            {
                "run": name,
                "method": rows[0]["method"],
                "bits": rows[0]["bits"],
                "tasks": len(rows),
                "paired_samples": sum(row["paired_samples"] for row in rows),
                "longbench_average": sum(row["mean_score"] for row in rows) / len(rows),
                "full_average": sum(row["mean_full_score"] for row in rows) / len(rows),
                "quality_retained": (
                    sum(row["mean_score"] for row in rows)
                    / sum(row["mean_full_score"] for row in rows)
                ) if sum(row["mean_full_score"] for row in rows) else None,
                "macro_score_drop": score_drop,
                "macro_score_drop_ci95_low": ci_low,
                "macro_score_drop_ci95_high": ci_high,
                "mean_kv_bytes_per_token": (
                    sum(byte_values) / len(byte_values) if byte_values else None
                ),
                "mean_kv_storage_ratio_vs_full": sum(ratios) / len(ratios) if ratios else None,
            }
        )
    return {
        "paired_record_count": len(common_ids),
        "runs": {name: str(path) for name, path in named_paths.items()},
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "paired records resampled within task, then macro-averaged across tasks",
        },
        "excluded_record_count": len(excluded),
        "memory_comparison": "observed storage values; equal budgets are not implied by nominal bits",
        "overall": overall_rows,
        "per_task": task_rows,
        "paired": paired_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=path.jsonl; include full=... and one or more TurboQuant/KIVI runs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/r5"))
    parser.add_argument("--evaluation-set", choices=["team", "adaptive", "custom"], default="team")
    parser.add_argument("--dataset", type=Path, help="frozen 800-record team dataset, required except for custom studies")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260)
    parser.add_argument(
        "--exclude-ids",
        type=Path,
        help="JSON list or an object containing record_ids, used to exclude calibration records",
    )
    args = parser.parse_args()
    named_paths = {}
    for value in args.run:
        if "=" not in value:
            raise SystemExit("each --run must be NAME=path.jsonl")
        name, path = value.split("=", 1)
        named_paths[name] = Path(path)
    exclude_ids: set[str] = set()
    if args.evaluation_set != "custom" and args.dataset is None:
        parser.error("--dataset is required to verify team/adaptive evaluation IDs")
    if args.evaluation_set == "team" and args.exclude_ids:
        parser.error("team comparison uses all 800 records; use --evaluation-set adaptive for calibration exclusion")
    if args.evaluation_set == "adaptive" and args.exclude_ids is None:
        args.exclude_ids = DATA_DIR / "adaptive_excluded_ids.json"
    if args.exclude_ids is not None:
        payload = json.loads(args.exclude_ids.read_text(encoding="utf-8"))
        values = payload.get("record_ids", []) if isinstance(payload, dict) else payload
        exclude_ids = {str(value) for value in values}
    if args.evaluation_set == "adaptive":
        frozen_ids = set(json.loads((DATA_DIR / "adaptive_excluded_ids.json").read_text())["record_ids"])
        if exclude_ids != frozen_ids:
            parser.error("adaptive comparison must exclude exactly the frozen 32 calibration IDs")
    if args.evaluation_set != "custom":
        from .scoring import SCORING_POLICY
        expected_rows = validate_dataset(args.dataset)
        expected = {row["record_id"]: row["task"] for row in expected_rows if row["record_id"] not in exclude_ids}
        for name, path in named_paths.items():
            rows = [row for row in _read(path) if row["record_id"] not in exclude_ids]
            if {row["record_id"]: row["task"] for row in rows} != expected:
                parser.error(f"{name} does not contain the exact {len(expected)} expected team record IDs/tasks")
            if any(row.get("score_scale") != "0-100" or row.get("scoring_policy") != SCORING_POLICY for row in rows):
                parser.error(f"{name} requires declared scoring_policy and 0-100 score_scale; score with turboquant-score")
    report = compare_runs(
        named_paths,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        exclude_ids=exclude_ids,
    )
    report["evaluation_set"] = args.evaluation_set
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r5_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_csv(args.output_dir / "r5_overall.csv", report["overall"])
    _write_csv(args.output_dir / "r5_per_task.csv", report["per_task"])
    _write_csv(args.output_dir / "r5_paired_samples.csv", report["paired"])
    print(json.dumps({"paired_record_count": report["paired_record_count"], "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
