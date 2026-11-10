#!/usr/bin/env python3
"""Verify the frozen Team 6 LongBench preprocessing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "dataset_manifest.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    rows = []
    for name, expected in manifest["files"].items():
        path = args.data_directory / name
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual_bytes = path.stat().st_size
        actual_hash = sha256(path)
        passed = actual_bytes == expected["bytes"] and actual_hash == expected["sha256"]
        rows.append(
            {
                "file": name,
                "bytes": actual_bytes,
                "sha256": actual_hash,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"identity mismatch: {path}")
        if "records" in expected:
            with path.open("rb") as handle:
                records = sum(1 for line in handle if line.strip())
            if records != expected["records"]:
                failures.append(
                    f"record-count mismatch: {path} has {records}, expected {expected['records']}"
                )
    print(json.dumps({"files": rows, "passed": not failures, "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
