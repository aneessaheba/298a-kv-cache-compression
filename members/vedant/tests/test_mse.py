import torch

from turboquant import TurboQuantMSE


def _unit_vectors(count: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    return torch.nn.functional.normalize(x, dim=-1)


def test_roundtrip_preserves_shape_and_zero() -> None:
    quantizer = TurboQuantMSE(128, 3, seed=11)
    x = torch.cat([_unit_vectors(7, 128, 1), torch.zeros(1, 128)], dim=0)
    reconstructed = quantizer.roundtrip(x)
    assert reconstructed.shape == x.shape
    torch.testing.assert_close(reconstructed[-1], torch.zeros(128, dtype=x.dtype))


def test_mse_matches_paper_small_bit_values() -> None:
    x = _unit_vectors(2048, 128, 21)
    paper = {1: 0.36, 2: 0.117, 3: 0.030, 4: 0.009}
    for bits, reference in paper.items():
        reconstructed = TurboQuantMSE(128, bits, seed=30 + bits).roundtrip(x)
        observed = (x - reconstructed).square().sum(-1).mean().item()
        assert abs(observed - reference) / reference < 0.18
