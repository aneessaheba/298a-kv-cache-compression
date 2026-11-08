"""Checks for the quantization and byte accounting, no model needed.

    python members/anees/test_quantized_restorekv.py
"""

import torch

from quantized_restorekv_press import (
    cache_bytes,
    eviction_ratio,
    key_metadata_bytes,
    quantize_keys,
    quantize_values,
    slot_bytes,
)

LAYERS, HEADS, DIM = 32, 8, 128  # Llama-3.1-8B
BF16_SLOT = 2 * DIM * 2  # bf16 key + value in one head


def test_slot_sizes():
    assert BF16_SLOT == 512
    assert slot_bytes(DIM, 4) == 160  # 128 data + 16 value metadata + 16 key metadata
    assert slot_bytes(DIM, 2) == 96
    assert key_metadata_bytes(LAYERS, HEADS, DIM) == 131072  # the size of one bf16 token


def test_budget_is_met():
    for tokens in (4000, 32000):
        full = LAYERS * HEADS * tokens * BF16_SLOT
        for budget in (0.05, 0.10):
            for bits in (4, 2):
                evict = eviction_ratio(budget, bits, tokens, LAYERS, HEADS, DIM)
                bound = (1 - evict) * LAYERS * HEADS * tokens * slot_bytes(DIM, bits) + key_metadata_bytes(LAYERS, HEADS, DIM)
                assert abs(bound / full - budget) < 1e-9


def test_exact_bytes_stay_under_the_bound():
    torch.manual_seed(0)
    layers, heads, tokens = 3, 4, 500
    masks = [torch.rand(1, heads, tokens) < 0.3 for _ in range(layers)]
    slots = sum(int(m.sum()) for m in masks)
    for bits in (4, 2):
        exact = cache_bytes(masks, DIM, bits)
        bound = slots * slot_bytes(DIM, bits) + key_metadata_bytes(layers, heads, DIM)
        assert slots * slot_bytes(DIM, bits) <= exact <= bound


def test_more_tokens_fit_at_fewer_bits():
    keep = [1 - eviction_ratio(0.05, bits, 32000, LAYERS, HEADS, DIM) for bits in (4, 2)]
    assert 0.15 < keep[0] < 0.16 and 0.26 < keep[1] < 0.267


def test_large_budget_keeps_everything():
    assert eviction_ratio(0.25, 2, 32000, LAYERS, HEADS, DIM) == 0.0


def test_keys_are_grouped_over_kept_tokens():
    torch.manual_seed(0)
    group = 8
    keys = torch.randn(1, 2, 100, 16, dtype=torch.float64)
    kept = torch.rand(1, 2, 100) < 0.5
    keys[0, 0, ~kept[0, 0], 0] = 1000.0  # an evicted outlier must not widen any range
    first = kept[0, 1].nonzero().flatten()[0]
    keys[0, 1, first, 1] = 500.0  # a kept outlier must only widen its own group
    for bits in (4, 2):
        out = quantize_keys(keys, kept, bits, group)
        assert torch.equal(out[~kept], keys[~kept].float().to(keys.dtype))
        for head in range(2):
            positions = kept[0, head].nonzero().flatten()
            for start in range(0, len(positions), group):
                rows = positions[start:start + group]
                for channel in range(16):
                    part, original = out[0, head, rows, channel], keys[0, head, rows, channel]
                    assert part.unique().numel() <= 2**bits
                    step = (original.max() - original.min()) / (2**bits - 1)
                    assert (part - original).abs().max() <= step / 2 + 1e-5
        later = kept[0, 1].nonzero().flatten()[group:]
        assert (out[0, 1, later, 1] - keys[0, 1, later, 1]).abs().max() < 1.0


def test_values_are_grouped_per_token():
    torch.manual_seed(0)
    group = 4
    values = torch.randn(1, 2, 50, 16, dtype=torch.float64)
    values[0, 0, 7] *= 100  # a large token must not affect its neighbours
    values[0, 1, 3, 0] = 300.0  # nor a large channel the other channel groups
    for bits in (4, 2):
        out = quantize_values(values, bits, group)
        for head in range(2):
            for token in range(50):
                for start in range(0, 16, group):
                    row = out[0, head, token, start:start + group]
                    original = values[0, head, token, start:start + group]
                    assert row.unique().numel() <= 2**bits
                    step = (original.max() - original.min()) / (2**bits - 1)
                    assert (row - original).abs().max() <= step / 2 + 1e-3
        assert (out[0, 1, 3, group:] - values[0, 1, 3, group:]).abs().max() < 1.0


def test_constant_and_empty_inputs_stay_finite():
    assert torch.isfinite(quantize_values(torch.ones(1, 1, 3, 32), 2)).all()
    assert torch.isfinite(quantize_keys(torch.ones(1, 1, 3, 4), torch.zeros(1, 1, 3, dtype=torch.bool), 2)).all()
    assert torch.isfinite(quantize_keys(torch.ones(1, 1, 3, 4), torch.ones(1, 1, 3, dtype=torch.bool), 2)).all()


def test_bf16_round_trip():
    keys = torch.randn(1, 2, 70, 32).to(torch.bfloat16)
    assert quantize_keys(keys, torch.ones(1, 2, 70, dtype=torch.bool), 4).dtype == torch.bfloat16
    assert quantize_values(keys, 4).dtype == torch.bfloat16


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed {test.__name__}")
    print(f"{len(tests)} tests passed")
