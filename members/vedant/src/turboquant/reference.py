"""Independent NumPy float64 scalar-quantizer reference for G3 fixtures."""

from __future__ import annotations

import numpy as np


def scalar_quantize_float64(
    values: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize with a NumPy-only nearest-centroid implementation."""

    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 1 or c.size < 2 or np.any(c[1:] <= c[:-1]):
        raise ValueError("centroids must be a strictly increasing 1-D array")
    boundaries = (c[:-1] + c[1:]) / np.float64(2.0)
    indices = np.searchsorted(boundaries, x, side="left")
    return indices.astype(np.int64), c[indices]


def scalar_mse_float64(
    centroids: np.ndarray,
    dimension: int,
    *,
    grid_size: int = 1_000_001,
) -> float:
    """Numerically integrate the matched unit-sphere scalar MSE in NumPy."""

    if dimension < 3:
        raise ValueError("dimension must be at least 3")
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    base = np.maximum(1.0 - x * x, 0.0)
    weights = np.power(base, (dimension - 3.0) / 2.0)
    weights[[0, -1]] = 0.0
    weights /= weights.sum(dtype=np.float64)
    _, reconstructed = scalar_quantize_float64(x, centroids)
    return float(np.sum(weights * np.square(x - reconstructed), dtype=np.float64))
