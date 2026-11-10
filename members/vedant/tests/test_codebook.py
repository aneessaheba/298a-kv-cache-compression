import math

import torch

from turboquant import build_sphere_codebook


def test_one_bit_codebook_matches_high_dimensional_limit() -> None:
    codebook = build_sphere_codebook(128, 1)
    expected = math.sqrt(2.0 / (math.pi * 128))
    torch.testing.assert_close(
        codebook.centroids,
        torch.tensor([-expected, expected], dtype=torch.float64),
        atol=3e-4,
        rtol=0.0,
    )


def test_codebook_is_symmetric_and_ordered() -> None:
    codebook = build_sphere_codebook(128, 4)
    assert torch.all(codebook.centroids[1:] > codebook.centroids[:-1])
    torch.testing.assert_close(
        codebook.centroids,
        -torch.flip(codebook.centroids, dims=(0,)),
        atol=1e-11,
        rtol=0.0,
    )
