"""Checks for the quantization and byte accounting, no model needed.

    python members/anees/test_quantized_restorekv.py
"""

import torch

from quantized_restorekv_press import (
    eviction_ratio,
    key_metadata_bytes,
    quantize_keys,
    quantize_values,
    slot_bytes,
)

LAYERS, HEADS, DIM = 32, 8, 128  # Llama-3.1-8B


def cache_bytes(evict, bits, tokens):
    kept = (1 - evict) * LAYERS * HEADS * tokens
    return kept * slot_bytes(DIM, bits) + key_metadata_bytes(LAYERS, HEADS, DIM)


def test_slot_sizes():
    assert slot_bytes(DIM, 16) - 4 == 512  # bf16 key + value in one head
    assert slot_bytes(DIM, 4) == 132
    assert slot_bytes(DIM, 2) == 68
    assert key_metadata_bytes(LAYERS, HEADS, DIM) == 131072  # the size of one bf16 token


def test_budget_is_met_exactly():
    for tokens in (4000, 32000):
        full = LAYERS * HEADS * tokens * 512
        for budget in (0.05, 0.10):
            for bits in (4, 2):
                evict = eviction_ratio(budget, bits, tokens, LAYERS, HEADS, DIM)
                assert abs(cache_bytes(evict, bits, tokens) / full - budget) < 1e-9


def test_more_tokens_fit_at_fewer_bits():
    keep = [1 - eviction_ratio(0.05, bits, 32000, LAYERS, HEADS, DIM) for bits in (4, 2)]
    assert 0.19 < keep[0] < 0.20 and 0.37 < keep[1] < 0.38


def test_large_budget_keeps_everything():
    assert eviction_ratio(0.25, 2, 32000, LAYERS, HEADS, DIM) == 0.0


def test_keys_use_kept_range_and_leave_evicted_alone():
    torch.manual_seed(0)
    keys = torch.randn(1, 2, 50, 16, dtype=torch.float64)
    kept = torch.rand(1, 2, 50) < 0.5
    keys[0, 0, ~kept[0, 0], 0] = 1000.0  # an evicted outlier must not widen the range
    for bits in (4, 2):
        out = quantize_keys(keys, kept, bits)
        assert torch.equal(out[~kept], keys[~kept].float().to(keys.dtype))
        for head in range(2):
            for channel in range(16):
                part, original = out[0, head, kept[0, head], channel], keys[0, head, kept[0, head], channel]
                assert part.unique().numel() <= 2**bits
                step = (original.max() - original.min()) / (2**bits - 1)
                assert (part - original).abs().max() <= step / 2 + 1e-5


def test_values_are_rounded_per_token():
    torch.manual_seed(0)
    values = torch.randn(1, 2, 50, 16, dtype=torch.float64)
    values[0, 0, 7] *= 100  # a large token must not affect its neighbours
    for bits in (4, 2):
        out = quantize_values(values, bits)
        for head in range(2):
            for token in range(50):
                row, original = out[0, head, token], values[0, head, token]
                assert row.unique().numel() <= 2**bits
                step = (original.max() - original.min()) / (2**bits - 1)
                assert (row - original).abs().max() <= step / 2 + 1e-3


def test_constant_and_empty_inputs_stay_finite():
    assert torch.isfinite(quantize_values(torch.ones(1, 1, 3, 4), 2)).all()
    assert torch.isfinite(quantize_keys(torch.ones(1, 1, 3, 4), torch.zeros(1, 1, 3, dtype=torch.bool), 2)).all()


def test_bf16_round_trip():
    keys = torch.randn(1, 2, 10, 8).to(torch.bfloat16)
    assert quantize_keys(keys, torch.ones(1, 2, 10, dtype=torch.bool), 4).dtype == torch.bfloat16
    assert quantize_values(keys, 4).dtype == torch.bfloat16


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed {test.__name__}")
    print(f"{len(tests)} tests passed")
