"""Algorithm 1: MSE-optimized TurboQuant."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .codebook import SphereCodebook, build_sphere_codebook
from .rotation import random_orthogonal_matrix


@dataclass(frozen=True)
class MSECodes:
    indices: torch.Tensor
    norms: torch.Tensor


class TurboQuantMSE:
    """Reference implementation of the paper's MSE quantizer.

    The final axis is interpreted as the vector dimension. Inputs may have any
    leading batch shape. Norms are stored separately, as prescribed by the
    paper for non-unit vectors.
    """

    def __init__(
        self,
        dimension: int,
        bits: int,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        codebook: SphereCodebook | None = None,
        rotation: torch.Tensor | None = None,
    ) -> None:
        if bits < 1:
            raise ValueError("bits must be at least 1")
        self.dimension = dimension
        self.bits = bits
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        if rotation is None:
            self.rotation = random_orthogonal_matrix(
                dimension, seed=seed, device=self.device, dtype=dtype
            )
        else:
            if rotation.shape != (dimension, dimension):
                raise ValueError("rotation must have shape (dimension, dimension)")
            self.rotation = rotation.to(device=self.device, dtype=dtype)
        base_codebook = codebook or build_sphere_codebook(dimension, bits)
        if base_codebook.dimension != dimension or base_codebook.bits != bits:
            raise ValueError("codebook dimension and bits must match the quantizer")
        self.codebook = base_codebook.to(device=self.device, dtype=dtype)

    def _validate(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"expected final dimension {self.dimension}, got {vectors.shape[-1]}"
            )
        return vectors.to(device=self.device, dtype=self.dtype)

    def quantize(self, vectors: torch.Tensor) -> MSECodes:
        vectors = self._validate(vectors)
        norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        safe_norms = norms.clamp_min(torch.finfo(self.dtype).tiny)
        unit = vectors / safe_norms
        unit = torch.where(norms > 0, unit, torch.zeros_like(unit))
        rotated = unit @ self.rotation.T
        indices = torch.bucketize(rotated.contiguous(), self.codebook.boundaries)
        return MSECodes(indices=indices.to(torch.int16), norms=norms)

    def dequantize(self, codes: MSECodes) -> torch.Tensor:
        if codes.indices.shape[-1] != self.dimension:
            raise ValueError("code tensor has the wrong final dimension")
        indices = codes.indices.to(device=self.device, dtype=torch.long)
        reconstructed_rotated = self.codebook.centroids[indices]
        unit = reconstructed_rotated @ self.rotation
        return unit * codes.norms.to(device=self.device, dtype=self.dtype)

    def roundtrip(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(vectors))
