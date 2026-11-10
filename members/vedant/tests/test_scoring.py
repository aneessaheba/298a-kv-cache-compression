import json

import pytest

from turboquant.scoring import SCORING_POLICY, score_file, score_row


@pytest.mark.parametrize("task,prediction,answer,classes", [
    ("triviaqa", " \n\n Paris\nextra", "Paris", []),
    ("trec", " \nHUM:ind\nextra", "HUM:ind", ["HUM:ind", "LOC:city"]),
    ("samsum", " \nAlice arrived.\nextra", "Alice arrived.", []),
    ("qasper", "The red fox", "red fox", []),
    ("passage_count", "3", "3", []),
    ("passage_retrieval_en", "Paragraph 4", "Paragraph 4", []),
    ("lcc", "return 1", "return 1", []),
])
def test_official_metrics_and_whitespace_fix(task, prediction, answer, classes):
    row = score_row({"task": task, "prediction": prediction, "answers": [answer], "all_classes": classes})
    assert row["score"] == pytest.approx(100, abs=1e-5)
    assert row["prediction"] == prediction
    assert row["scoring_policy"] == SCORING_POLICY


def test_scoring_rejects_missing_class_labels():
    with pytest.raises(ValueError, match="all_classes"):
        score_row({"task": "trec", "prediction": "HUM:ind", "answers": ["HUM:ind"]})


def test_summary_averages_tasks_and_preserves_metadata(tmp_path):
    source = tmp_path / "raw.jsonl"
    rows = [{"record_id": "a", "task": "qasper", "prediction": "yes", "answers": ["yes"], "run_fingerprint": "run"},
            {"record_id": "b", "task": "qasper", "prediction": "yes", "answers": ["yes"]},
            {"record_id": "c", "task": "triviaqa", "prediction": "no", "answers": ["yes"]}]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "scored.jsonl"
    summary = score_file(source, output)
    assert summary["longbench_macro_score"] == 50
    assert json.loads(output.read_text().splitlines()[0])["run_fingerprint"] == "run"
    with pytest.raises(ValueError, match="already exists"):
        score_file(source, output)
