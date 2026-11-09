"""Dataset identity and safe resumability for the Team 6 experiment protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
PROTOCOL_VERSION = "team6-longbench-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(path: Path, *, scope: str = "team") -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input dataset is empty")
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("input dataset contains duplicate record IDs")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("each input record requires a nonempty preprocessed prompt")
        if not row.get("answers") or not isinstance(row["answers"], list):
            raise ValueError("each input record requires a nonempty answers list")
        if row["task"] == "trec" and not row.get("all_classes"):
            raise ValueError("trec requires all_classes for official scoring")
    if scope == "team":
        manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text())
        expected = manifest["files"]["processed_longbench_dev.jsonl"]["sha256"]
        if sha256(path) != expected:
            raise ValueError("dataset identity mismatch: use the frozen Team 6 800-record dev file; "
                             "use --dataset-scope subset only for explicitly labelled smoke/context runs")
        if len(rows) != 800 or sorted(Counter(row["task"] for row in rows).values()) != [50] * 16:
            raise ValueError("team scope requires 16 tasks with exactly 50 unique records each")
    elif scope != "subset":
        raise ValueError(f"unknown dataset scope: {scope}")
    return rows


def fingerprint(specification: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def prepare_output(output: Path, specification: dict[str, Any], valid_ids: set[str]) -> str:
    """Reject changed settings, orphaned outputs and incomplete/corrupt JSONL tails."""
    manifest_path = output.with_suffix(output.suffix + ".run.json")
    run_id = fingerprint(specification)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved != {"run_fingerprint": run_id, "specification": specification}:
            raise ValueError("resume settings differ; choose a new output path")
    elif output.exists() and output.stat().st_size:
        raise ValueError("existing output lacks a run manifest; choose a new output path")
    if output.exists():
        seen = set()
        with output.open() as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("resume output has a blank or incomplete row")
                row = json.loads(line)
                rid = row["record_id"]
                if rid not in valid_ids or rid in seen:
                    raise ValueError("resume output contains unexpected or duplicate record IDs")
                if row.get("run_fingerprint") != run_id:
                    raise ValueError("resume row provenance differs; choose a new output path")
                seen.add(rid)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        with manifest_path.open("x") as handle:
            json.dump({"run_fingerprint": run_id, "specification": specification}, handle, indent=2)
            handle.write("\n")
    return run_id
