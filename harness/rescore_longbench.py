"""Recompute the scores in LongBench result files after a scoring change, without rerunning the model.

Rewrites each .jsonl and its .summary.json in place from the saved predictions.

    python harness/rescore_longbench.py results/restorekv/longbench/*.jsonl
"""

import argparse
import json
from pathlib import Path

import longbench

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    records = {r["record_id"]: r for r in longbench.load_records(args.data)}
    for path in args.files:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        for row in rows:
            record = records[row["record_id"]]
            row["score"] = longbench.score(row["task"], row["prediction"], record["answers"], record["all_classes"])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        summary = longbench.summarize(rows)
        summary_file = path.with_suffix(".summary.json")
        meta = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        meta.update(n_records=len(rows), scores=summary)
        summary_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{path.name}: average {summary['average']}, triviaqa {summary.get('triviaqa')}")


if __name__ == "__main__":
    main()
