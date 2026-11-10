"""Paper-faithful TurboQuant reference implementation."""

from .codebook import SphereCodebook, build_sphere_codebook
from .mse import MSECodes, TurboQuantMSE
from .packing import (
    PackedBits,
    PackedMSECodes,
    PackedProductCodes,
    pack_bits,
    pack_mse_codes,
    pack_product_codes,
    unpack_bits,
    unpack_mse_codes,
    unpack_product_codes,
)
from .product import ProductCodes, TurboQuantProd
from .rotation import random_orthogonal_matrix

__all__ = [
    "MSECodes",
    "PackedBits",
    "PackedMSECodes",
    "PackedProductCodes",
    "ProductCodes",
    "SphereCodebook",
    "TurboQuantMSE",
    "TurboQuantProd",
    "build_sphere_codebook",
    "pack_bits",
    "pack_mse_codes",
    "pack_product_codes",
    "random_orthogonal_matrix",
    "unpack_bits",
    "unpack_mse_codes",
    "unpack_product_codes",
]
