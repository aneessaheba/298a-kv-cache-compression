import json

import pytest

from turboquant.compare import compare_runs


def test_compare_requires_paired_scored_rows(tmp_path) -> None:
    full = tmp_path / "full.jsonl"
    compressed = tmp_path / "compressed.jsonl"
    rows = [
        {"record_id": "a", "task": "qasper", "method": "full", "bits": 16, "score": 80.0, "measured_kv_bytes_per_token": 100},
        {"record_id": "b", "task": "qasper", "method": "full", "bits": 16, "score": 100.0, "measured_kv_bytes_per_token": 100},
    ]
    full.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rows = [
        {"record_id": "a", "task": "qasper", "method": "turboquant", "bits": 3, "score": 72.0, "measured_kv_bytes_per_token": 20},
        {"record_id": "b", "task": "qasper", "method": "turboquant", "bits": 3, "score": 90.0, "measured_kv_bytes_per_token": 20},
    ]
    compressed.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = compare_runs({"full": full, "tq3": compressed})
    assert report["paired_record_count"] == 2
    tq = next(row for row in report["overall"] if row["run"] == "tq3")
    assert tq["longbench_average"] == 81.0
    assert tq["quality_retained"] == 0.9
    assert tq["macro_score_drop"] == 9.0
    assert tq["macro_score_drop_ci95_low"] <= tq["macro_score_drop_ci95_high"]


def test_compare_rejects_unmatched_records(tmp_path) -> None:
    full = tmp_path / "full.jsonl"
    compressed = tmp_path / "compressed.jsonl"
    full.write_text(
        json.dumps({"record_id": "a", "task": "qasper", "score": 1.0}) + "\n"
        + json.dumps({"record_id": "b", "task": "qasper", "score": 1.0}) + "\n"
    )
    compressed.write_text(
        json.dumps({"record_id": "a", "task": "qasper", "score": 1.0}) + "\n"
    )
    with pytest.raises(ValueError, match="1 missing"):
        compare_runs({"full": full, "tq": compressed})


def test_compare_accepts_external_run_without_byte_measurement(tmp_path) -> None:
    full = tmp_path / "full.jsonl"
    kivi = tmp_path / "kivi.jsonl"
    full.write_text(
        json.dumps({"record_id": "a", "task": "qasper", "score": 1.0}) + "\n"
    )
    kivi.write_text(
        json.dumps({"record_id": "a", "task": "qasper", "score": 0.9}) + "\n"
    )
    report = compare_runs({"full": full, "kivi": kivi})
    kivi_row = next(row for row in report["overall"] if row["run"] == "kivi")
    assert kivi_row["mean_kv_bytes_per_token"] is None


@pytest.mark.parametrize("field,values", [("input_ids_sha256", ("a", "b")),
                                       ("scoring_policy", ("old", "fixed"))])
def test_compare_rejects_prompt_or_scorer_mismatch(tmp_path, field, values):
    paths = {}
    for name, value in zip(("full", "tq"), values):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"record_id": "a", "task": "qasper", "score": 10, field: value}) + "\n")
        paths[name] = path
    with pytest.raises(ValueError, match=field):
        compare_runs(paths)
