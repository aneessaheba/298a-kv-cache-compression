import torch

from turboquant.kv_cache import TurboQuantCacheLayer


def test_cache_appends_tokens_and_counts_physical_bytes() -> None:
    generator = torch.Generator().manual_seed(4)
    layer = TurboQuantCacheLayer(16, 3, qjl_keys=True, seed=9)
    first = torch.randn(1, 2, 5, 16, generator=generator)
    second = torch.randn(1, 2, 1, 16, generator=generator)
    keys, values = layer.update(first, first)
    assert keys.shape == values.shape == first.shape
    keys, values = layer.update(second, second)
    assert keys.shape == values.shape == (1, 2, 6, 16)
    stats = layer.storage_stats()
    assert stats.tokens == 6
    assert stats.compressed_bytes < stats.full_precision_bytes
    assert 0.0 < stats.ratio < 1.0


def test_non_qjl_cache_runs_at_two_bits() -> None:
    x = torch.randn(1, 1, 3, 16)
    layer = TurboQuantCacheLayer(16, 2, qjl_keys=False)
    keys, values = layer.update(x, x)
    assert keys.shape == values.shape == x.shape
