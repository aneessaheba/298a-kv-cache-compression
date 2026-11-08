import pytest

from turboquant.adaptive import LayerCandidate, allocate_layers
from turboquant.adaptive import _aggregate_reports


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


def test_adaptive_objective_is_not_dominated_by_absolute_key_units():
    row={'method':'turboquant_product','bits':3,'inner_product_error_mse':1e9,
         'key_normalized_mse':0.03,'value_normalized_mse':0.02,'packed_bytes':100,'vectors':10}
    candidates=_aggregate_reports({0:[{'rows':[row]}]},key_weight=1,value_weight=1)
    assert candidates[0][0].objective == pytest.approx(0.05)
    row['inner_product_error_mse'] *= 1000
    assert _aggregate_reports({0:[{'rows':[row]}]},key_weight=1,value_weight=1)[0][0].objective == pytest.approx(0.05)
