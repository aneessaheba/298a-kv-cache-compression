"""Lloyd-Max codebooks for rotated unit-sphere coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class SphereCodebook:
    """Scalar centroids and decision boundaries for one dimension/bit width."""

    dimension: int
    bits: int
    centroids: torch.Tensor
    boundaries: torch.Tensor
    estimated_scalar_mse: float

    def to(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "SphereCodebook":
        return SphereCodebook(
            dimension=self.dimension,
            bits=self.bits,
            centroids=self.centroids.to(device=device, dtype=dtype),
            boundaries=self.boundaries.to(device=device, dtype=dtype),
            estimated_scalar_mse=self.estimated_scalar_mse,
        )


def _weighted_quantile_indices(weights: torch.Tensor, levels: int) -> torch.Tensor:
    cdf = torch.cumsum(weights, dim=0)
    cdf = cdf / cdf[-1]
    probabilities = (torch.arange(levels, dtype=weights.dtype) + 0.5) / levels
    return torch.searchsorted(cdf, probabilities).clamp_max(weights.numel() - 1)


@lru_cache(maxsize=32)
def build_sphere_codebook(
    dimension: int,
    bits: int,
    *,
    grid_size: int = 200_001,
    max_iter: int = 200,
    tolerance: float = 1e-13,
) -> SphereCodebook:
    """Numerically solve the paper's 1-D continuous k-means problem.

    The coordinate density for a uniformly random point on S^(d-1) is
    proportional to `(1 - x^2)^((d - 3) / 2)` on [-1, 1]. A dense midpoint
    grid makes the Lloyd-Max update deterministic and avoids a SciPy runtime
    dependency.
    """

    if dimension < 3:
        raise ValueError("dimension must be at least 3")
    if bits < 1:
        raise ValueError("bits must be at least 1")
    if grid_size < 10_001 or grid_size % 2 == 0:
        raise ValueError("grid_size must be odd and at least 10001")

    levels = 1 << bits
    x = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
    base = torch.clamp(1.0 - x.square(), min=0.0)
    weights = torch.pow(base, (dimension - 3.0) / 2.0)
    weights[[0, -1]] = 0.0
    weights /= weights.sum()

    centroids = x[_weighted_quantile_indices(weights, levels)].clone()
    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        assignments = torch.bucketize(x, boundaries)
        masses = torch.zeros(levels, dtype=torch.float64)
        weighted_x = torch.zeros(levels, dtype=torch.float64)
        masses.scatter_add_(0, assignments, weights)
        weighted_x.scatter_add_(0, assignments, weights * x)
        updated = weighted_x / masses.clamp_min(torch.finfo(torch.float64).tiny)
        # The sphere-coordinate density is exactly even.  The dense numerical
        # grid can still put a boundary point in only one of two mirrored
        # buckets, so project every update back onto the symmetric solution.
        updated = (updated - torch.flip(updated, dims=(0,))) / 2.0
        if torch.max(torch.abs(updated - centroids)).item() < tolerance:
            centroids = updated
            break
        centroids = updated

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    assignments = torch.bucketize(x, boundaries)
    scalar_mse = torch.sum(weights * (x - centroids[assignments]).square()).item()
    return SphereCodebook(
        dimension=dimension,
        bits=bits,
        centroids=centroids,
        boundaries=boundaries,
        estimated_scalar_mse=scalar_mse,
    )
