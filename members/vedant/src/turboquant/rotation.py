"""Random rotations used by TurboQuant."""

from __future__ import annotations

import torch


def random_orthogonal_matrix(
    dimension: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a deterministic Haar-random orthogonal matrix.

    A Gaussian matrix followed by QR decomposition yields a Haar draw after
    correcting the arbitrary signs on the diagonal of R.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(
        dimension,
        dimension,
        generator=generator,
        dtype=dtype,
        device="cpu",
    )
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.to(device=device)
