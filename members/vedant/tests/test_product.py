import math

import torch

from turboquant import TurboQuantProd


def _unit_vectors(count: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    return torch.nn.functional.normalize(x, dim=-1)


def test_product_inner_product_error_is_statistically_unbiased() -> None:
    samples = 4096
    x = _unit_vectors(samples, 128, 100)
    y = _unit_vectors(samples, 128, 101)
    reconstructed = TurboQuantProd(128, 2, seed=102).roundtrip(x)
    errors = (y * reconstructed).sum(-1) - (y * x).sum(-1)
    standard_error = errors.std(unbiased=True).item() / math.sqrt(samples)
    assert abs(errors.mean().item()) < 4.0 * standard_error


def test_more_bits_reduce_product_distortion() -> None:
    x = _unit_vectors(1024, 128, 200)
    y = _unit_vectors(1024, 128, 201)
    distortions = []
    for bits in (2, 3, 4):
        reconstructed = TurboQuantProd(128, bits, seed=202 + bits).roundtrip(x)
        errors = (y * reconstructed).sum(-1) - (y * x).sum(-1)
        distortions.append(errors.square().mean().item())
    assert distortions[0] > distortions[1] > distortions[2]
