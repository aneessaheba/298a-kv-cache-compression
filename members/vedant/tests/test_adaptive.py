import pytest

from turboquant.adaptive import LayerCandidate, allocate_layers


def candidate(layer: int, bits: int, objective: float, storage: float) -> LayerCandidate:
    return LayerCandidate(
        layer=layer,
        bits=bits,
        qjl_keys=True,
        objective=objective,
        bytes_per_vector=storage,
        key_error=objective,
        value_error=0.0,
    )


def test_allocator_improves_objective_with_reference_storage_budget() -> None:
    candidates = {
        0: [candidate(0, 2, 0.6, 8), candidate(0, 3, 0.5, 10), candidate(0, 4, 0.1, 12)],
        1: [candidate(1, 2, 0.2, 8), candidate(1, 3, 0.5, 10), candidate(1, 4, 0.05, 12)],
    }
    report = allocate_layers(candidates, reference_bits=3, reference_qjl=True)
    assert [row["bits"] for row in report["layers"]] == [4, 2]
    assert report["summary"]["selected_measured_bytes_per_vector"] == 20
    assert report["summary"]["selected_objective"] == pytest.approx(0.3)
    assert report["summary"]["objective_reduction"] == pytest.approx(0.7)
