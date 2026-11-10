from turboquant.splits import make_splits


def test_stratified_splits_are_deterministic_and_disjoint() -> None:
    rows = [
        {"record_id": f"{task}-{index}", "task": task}
        for task in ("a", "b", "c")
        for index in range(8)
    ]
    first = make_splits(rows, calibration_per_task=2, context_per_task=2, seed=7)
    second = make_splits(rows, calibration_per_task=2, context_per_task=2, seed=7)
    assert first == second
    calibration = {row["record_id"] for row in first["calibration"]}
    evaluation = {row["record_id"] for row in first["evaluation"]}
    assert len(calibration) == 6
    assert len(first["context"]) == 6
    assert not calibration & evaluation
    assert len(evaluation) == 18
