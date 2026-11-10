"""Outlier-aware independent subgroup quantizers with honest nominal bit rates.

The paper's printed 32x3+96x2 example is 2.25, not 2.5 bits. Our explicitly
defined 2.5/3.5 settings use half the channels at the next higher precision.
This implements the paper's two-quantizer strategy, not an exact reproduction
of its ambiguous channel-allocation recipe.
"""

from __future__ import annotations

import torch

from .kv_cache import TurboQuantCacheLayer


def head_token_bytes(head_dim: int, bits: float, qjl_keys: bool = True) -> float:
    groups = 2 if bits % 1 else 1
    if bits not in (2, 2.5, 3, 3.5, 4) or (groups == 2 and head_dim % 2):
        raise ValueError("supported rates: 2/2.5/3/3.5/4; fractional settings require even head_dim")
    # Two-byte vector norms; QJL keys additionally store residual norms.
    return 2 * head_dim * bits / 8 + groups * (6 if qjl_keys else 4)


def quantize_document(keys, values, bits: float, *, qjl_keys=True, seed=42, chunk_tokens=4096):
    """Return reconstructed document K/V and actual packed representation bytes.

    Select high-energy original channels independently per KV head, before any
    rotation. Rank using document-only mean-square magnitude; no future query
    or evaluation answer is used. Each subgroup gets an independent rotation.
    """
    dim = keys.shape[-1]
    head_token_bytes(dim, bits, qjl_keys)
    if not keys.shape[-2]:
        raise ValueError("document cannot be empty")
    if bits % 1:
        permutations = [x.float().square().mean(dim=-2).argsort(dim=-1, descending=True)
                        for x in (keys, values)]
        split = dim // 2
        groups = [(int(bits) + 1, 0, split), (int(bits), split, dim)]
        # Store two complete permutations in uint16-sized slots (K and V).
        if dim > 32767:
            raise ValueError("channel-index representation requires head_dim <= 32767")
        stored_permutations = [p.to(torch.int16) for p in permutations]
        selector_bytes = sum(p.numel() * p.element_size() for p in stored_permutations)
    else:
        permutations = None
        selector_bytes = 0
        groups = [(int(bits), 0, dim)]
    decoded_keys, decoded_values = torch.empty_like(keys), torch.empty_like(values)
    packed_bytes = selector_bytes
    for group_idx, (group_bits, left, right) in enumerate(groups):
        layer = TurboQuantCacheLayer(right-left, group_bits, qjl_keys=qjl_keys,
                                    seed=seed + group_idx * 1_000_003)
        for start in range(0, keys.shape[-2], chunk_tokens):
            end = min(start + chunk_tokens, keys.shape[-2])
            if permutations is None:
                k, v = keys[:, :, start:end], values[:, :, start:end]
            else:
                ki = permutations[0][..., left:right].unsqueeze(-2).expand(-1, -1, end-start, -1)
                vi = permutations[1][..., left:right].unsqueeze(-2).expand(-1, -1, end-start, -1)
                k, v = keys[:, :, start:end].gather(-1, ki), values[:, :, start:end].gather(-1, vi)
            # Reset each chunk's buffers but reuse the same initialized quantizers.
            layer.reset()
            layer.update(k, v)
            packed_bytes += layer.storage_stats().compressed_bytes
            dk, dv = layer._decoded_keys, layer._decoded_values
            if permutations is None:
                decoded_keys[:, :, start:end] = dk
                decoded_values[:, :, start:end] = dv
            else:
                decoded_keys[:, :, start:end].scatter_(-1, ki, dk)
                decoded_values[:, :, start:end].scatter_(-1, vi, dv)
    return decoded_keys, decoded_values, {"packed_bytes": packed_bytes,
                                        "selector_bytes": selector_bytes,
                                        "nominal_bits": float(bits),
                                        "outlier_channels": dim // 2 if bits % 1 else 0}
