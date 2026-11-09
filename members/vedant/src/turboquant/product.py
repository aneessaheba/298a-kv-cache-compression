"""Algorithm 2: inner-product optimized TurboQuant."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .codebook import SphereCodebook
from .mse import MSECodes, TurboQuantMSE


@dataclass(frozen=True)
class ProductCodes:
    mse: MSECodes
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor


class TurboQuantProd:
    """Reference `(b-1)`-bit MSE plus 1-bit residual QJL quantizer."""

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
        projection: torch.Tensor | None = None,
    ) -> None:
        if bits < 2:
            raise ValueError("this reference implementation supports total bits >= 2")
        self.dimension = dimension
        self.bits = bits
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.mse = TurboQuantMSE(
            dimension,
            bits - 1,
            seed=seed,
            device=self.device,
            dtype=dtype,
            codebook=codebook,
            rotation=rotation,
        )
        if projection is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1)
            self.projection = torch.randn(
                dimension,
                dimension,
                generator=generator,
                dtype=dtype,
                device="cpu",
            ).to(device=self.device)
        else:
            if projection.shape != (dimension, dimension):
                raise ValueError("projection must have shape (dimension, dimension)")
            self.projection = projection.to(device=self.device, dtype=dtype)

    def quantize(self, vectors: torch.Tensor) -> ProductCodes:
        vectors = vectors.to(device=self.device, dtype=self.dtype)
        if vectors.shape[-1] != self.dimension:
            raise ValueError("input tensor has the wrong final dimension")
        norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        safe_norms = norms.clamp_min(torch.finfo(self.dtype).tiny)
        unit = vectors / safe_norms
        unit = torch.where(norms > 0, unit, torch.zeros_like(unit))

        unit_codes = self.mse.quantize(unit)
        base = self.mse.dequantize(unit_codes)
        residual = unit - base
        residual_norms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        projected = residual @ self.projection.T
        signs = torch.where(projected >= 0, 1, -1).to(torch.int8)
        signs = torch.where(residual_norms > 0, signs, torch.ones_like(signs))
        scaled_mse_codes = MSECodes(indices=unit_codes.indices, norms=norms)
        return ProductCodes(
            mse=scaled_mse_codes,
            qjl_signs=signs,
            residual_norms=residual_norms,
        )

    def dequantize(self, codes: ProductCodes) -> torch.Tensor:
        base = self.mse.dequantize(codes.mse)
        signs = codes.qjl_signs.to(device=self.device, dtype=self.dtype)
        qjl_unit = (
            math.sqrt(math.pi / 2.0)
            / self.dimension
            * codes.residual_norms.to(device=self.device, dtype=self.dtype)
            * (signs @ self.projection)
        )
        norms = codes.mse.norms.to(device=self.device, dtype=self.dtype)
        return base + norms * qjl_unit

    def roundtrip(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(vectors))
