"""Physical bit packing and exact per-cache storage accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from .mse import MSECodes
from .product import ProductCodes


@dataclass(frozen=True)
class PackedBits:
    data: torch.Tensor
    bits: int
    shape: tuple[int, ...]

    @property
    def count(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()


@dataclass(frozen=True)
class PackedMSECodes:
    indices: PackedBits
    norms: torch.Tensor

    @property
    def nbytes(self) -> int:
        return self.indices.nbytes + self.norms.numel() * self.norms.element_size()


@dataclass(frozen=True)
class PackedProductCodes:
    mse: PackedMSECodes
    qjl_signs: PackedBits
    residual_norms: torch.Tensor

    @property
    def nbytes(self) -> int:
        return (
            self.mse.nbytes
            + self.qjl_signs.nbytes
            + self.residual_norms.numel() * self.residual_norms.element_size()
        )


def pack_bits(values: torch.Tensor, bits: int) -> PackedBits:
    """Pack unsigned integer values into a byte tensor without padding per vector."""

    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    values_i64 = values.detach().to(dtype=torch.int64).reshape(-1)
    if values_i64.numel() and (
        values_i64.min().item() < 0 or values_i64.max().item() >= (1 << bits)
    ):
        raise ValueError(f"values must fit in {bits} unsigned bits")

    positions = torch.arange(values_i64.numel(), device=values_i64.device) * bits
    byte_index = torch.div(positions, 8, rounding_mode="floor")
    offsets = positions.remainder(8)
    output_size = (values_i64.numel() * bits + 7) // 8
    packed = torch.zeros(output_size, dtype=torch.int64, device=values_i64.device)
    if values_i64.numel():
        packed.scatter_add_(0, byte_index, (values_i64 << offsets) & 0xFF)
        crosses = offsets + bits > 8
        if crosses.any():
            packed.scatter_add_(
                0,
                byte_index[crosses] + 1,
                values_i64[crosses] >> (8 - offsets[crosses]),
            )
    return PackedBits(packed.to(torch.uint8), bits, tuple(values.shape))


def unpack_bits(packed: PackedBits, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Invert :func:`pack_bits` and return an int64 tensor."""

    target = torch.device(device) if device is not None else packed.data.device
    data = packed.data.to(device=target, dtype=torch.int64)
    positions = torch.arange(packed.count, device=target) * packed.bits
    byte_index = torch.div(positions, 8, rounding_mode="floor")
    offsets = positions.remainder(8)
    if packed.count == 0:
        return torch.empty(packed.shape, dtype=torch.int64, device=target)
    values = data[byte_index] >> offsets
    crosses = offsets + packed.bits > 8
    if crosses.any():
        values[crosses] |= data[byte_index[crosses] + 1] << (8 - offsets[crosses])
    values &= (1 << packed.bits) - 1
    return values.reshape(packed.shape)


def pack_mse_codes(
    codes: MSECodes,
    bits: int,
    *,
    norm_dtype: torch.dtype = torch.float16,
) -> PackedMSECodes:
    return PackedMSECodes(
        indices=pack_bits(codes.indices, bits),
        norms=codes.norms.detach().to(dtype=norm_dtype),
    )


def unpack_mse_codes(
    codes: PackedMSECodes,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MSECodes:
    return MSECodes(
        indices=unpack_bits(codes.indices, device=device).to(torch.int16),
        norms=codes.norms.to(device=device, dtype=dtype),
    )


def pack_product_codes(
    codes: ProductCodes,
    mse_bits: int,
    *,
    norm_dtype: torch.dtype = torch.float16,
) -> PackedProductCodes:
    binary_signs = (codes.qjl_signs > 0).to(torch.uint8)
    return PackedProductCodes(
        mse=pack_mse_codes(codes.mse, mse_bits, norm_dtype=norm_dtype),
        qjl_signs=pack_bits(binary_signs, 1),
        residual_norms=codes.residual_norms.detach().to(dtype=norm_dtype),
    )


def unpack_product_codes(
    codes: PackedProductCodes,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ProductCodes:
    signs = unpack_bits(codes.qjl_signs, device=device).to(torch.int8) * 2 - 1
    return ProductCodes(
        mse=unpack_mse_codes(codes.mse, device=device, dtype=dtype),
        qjl_signs=signs,
        residual_norms=codes.residual_norms.to(device=device, dtype=dtype),
    )
