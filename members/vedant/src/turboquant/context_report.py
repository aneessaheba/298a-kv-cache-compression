"""Summarize scored context-length scaling runs and generate report figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_context_runs(runs: dict[tuple[str, int], Path]) -> list[dict[str, Any]]:
    loaded = {key: _read(path) for key, path in runs.items()}
    expected_ids: set[str] | None = None
    summaries = []
    for (name, context_length), rows in sorted(loaded.items(), key=lambda item: (item[0][0], item[0][1])):
        ids = {row["record_id"] for row in rows}
        if len(ids) != len(rows):
            raise ValueError(f"{name}@{context_length} contains duplicate record IDs")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"{name}@{context_length} does not contain the same paired records")
        if any(row.get("score") is None for row in rows):
            raise ValueError(f"{name}@{context_length} contains unscored records")
        per_task: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            per_task[row["task"]].append(float(row["score"]))
        macro_score = sum(sum(values) / len(values) for values in per_task.values()) / len(per_task)
        summaries.append(
            {
                "run": name,
                "context_cap": context_length,
                "records": len(rows),
                "tasks": len(per_task),
                "longbench_macro_score": macro_score,
                "mean_input_tokens": sum(float(row["input_tokens"]) for row in rows) / len(rows),
                "mean_cache_mib": sum(float(row["measured_kv_bytes"]) for row in rows)
                / len(rows)
                / (1024 * 1024),
                "mean_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows) / len(rows),
            }
        )
    return summaries


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_scaling.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output_dir / "context_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name in sorted({row["run"] for row in rows}):
        selected = [row for row in rows if row["run"] == name]
        axes[0].plot(
            [row["context_cap"] for row in selected],
            [row["longbench_macro_score"] for row in selected],
            marker="o",
            label=name,
        )
        axes[1].plot(
            [row["context_cap"] for row in selected],
            [row["mean_cache_mib"] for row in selected],
            marker="o",
            label=name,
        )
    for axis in axes:
        axis.set_xlabel("context cap (tokens)")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("LongBench macro score")
    axes[0].set_title("Quality across context lengths")
    axes[1].set_ylabel("mean measured KV cache (MiB)")
    axes[1].set_title("Cache growth across context lengths")
    figure.tight_layout()
    figure.savefig(output_dir / "context_scaling.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="NAME,CONTEXT_LENGTH=path.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/context_scaling/report"))
    args = parser.parse_args()
    runs = {}
    for specification in args.run:
        if "=" not in specification or "," not in specification.split("=", 1)[0]:
            raise SystemExit("each --run must be NAME,CONTEXT_LENGTH=path.jsonl")
        label, path = specification.split("=", 1)
        name, length = label.rsplit(",", 1)
        runs[(name, int(length))] = Path(path)
    rows = summarize_context_runs(runs)
    _write_outputs(rows, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
