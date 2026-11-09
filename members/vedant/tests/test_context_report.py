import json

import pytest

from turboquant.context_report import summarize_context_runs


def test_context_report_uses_task_macro_average_and_paired_records(tmp_path) -> None:
    paths = {}
    for name, length, scores in (
        ("full", 4096, [1.0, 3.0, 8.0]),
        ("adaptive", 4096, [0.8, 2.8, 7.0]),
    ):
        path = tmp_path / f"{name}_{length}.jsonl"
        rows = [
            {
                "record_id": f"r{index}",
                "task": "a" if index < 2 else "b",
                "score": score,
                "input_tokens": 100,
                "measured_kv_bytes": 1024 * 1024,
                "elapsed_seconds": 2.0,
            }
            for index, score in enumerate(scores)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[(name, length)] = path
    report = summarize_context_runs(paths)
    full = next(row for row in report if row["run"] == "full")
    assert full["longbench_macro_score"] == pytest.approx(5.0)
    assert full["mean_cache_mib"] == pytest.approx(1.0)


def test_oom_keeps_failed_quality_without_claiming_zero_memory(tmp_path):
    p=tmp_path/'run.jsonl'
    p.write_text(''.join(json.dumps(row)+'\n' for row in (
        {'record_id':'a','task':'a','score':0,'input_tokens':20,'oom':True,
         'measured_kv_bytes':None,'elapsed_seconds':2},
        {'record_id':'b','task':'a','score':10,'input_tokens':20,'oom':False,
         'measured_kv_bytes':1024**2,'elapsed_seconds':2})))
    row=summarize_context_runs({('tq',4096):p})[0]
    assert row['longbench_macro_score']==5
    assert row['oom_records']==1 and row['memory_observations']==1
    assert row['mean_cache_mib']==1
