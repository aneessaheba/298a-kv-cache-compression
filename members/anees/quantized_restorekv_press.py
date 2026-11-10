"""RestoreKV plus KIVI-style quantization of the kept cache, at equal memory (SAM-96).

The question: for the same number of cache bytes, is it better to keep few tokens
at full precision (RestoreKV) or more tokens at 4 or 2 bits?

`compression_ratio` is the share of the bf16 document cache's bytes to remove, as in
the rest of the harness, so 0.95 means a 5% memory budget. The press works out how
many KV pairs fit in that budget at the chosen bit-width, evicts the rest with
RestoreKV, then quantizes what is left, including the restore tokens. As in KIVI,
values are grouped in groups of `group_size`:

- Keys: one scale and zero point per channel for every `group_size` kept tokens of a head.
- Values: one scale and zero point per token for every `group_size` channels.

Scales and zero points are stored in bf16 and counted in the budget. As in kvpress,
evicted pairs are masked and tokens are rounded in place rather than stored in packed
form, so accuracy is real but memory is computed. Question tokens added after the
document are not compressed.
"""

import math
from dataclasses import dataclass, field

import torch
from kvpress import RestoreKVPress

METADATA_BYTES = 4  # scale + zero point, both bf16
GROUP_SIZE = 32  # KIVI's default


def slot_bytes(head_dim, bits, group_size=GROUP_SIZE):
    """Bytes for one kept token in one head: quantized key and value, the value's
    scales and zero points, and the token's share of the key's."""
    return 2 * head_dim * bits / 8 + 2 * head_dim / group_size * METADATA_BYTES


def key_metadata_bytes(num_layers, num_heads, head_dim):
    """Upper bound on the key metadata not covered by `slot_bytes`: one partly
    filled token group per head."""
    return num_layers * num_heads * head_dim * METADATA_BYTES


def cache_bytes(kept, head_dim, bits, group_size=GROUP_SIZE):
    """Exact bytes of a quantized cache, given the kept mask (batch, head, token) of each layer."""
    total = 0
    for mask in kept:
        per_head = mask.sum(dim=-1)
        groups = int(torch.div(per_head + group_size - 1, group_size, rounding_mode="floor").sum())
        slots = int(per_head.sum())
        total += slots * 2 * head_dim * bits / 8                       # keys and values
        total += slots * head_dim / group_size * METADATA_BYTES         # value groups
        total += groups * head_dim * METADATA_BYTES                     # key groups
    return total


def eviction_ratio(memory_budget, bits, num_tokens, num_layers, num_heads, head_dim, group_size=GROUP_SIZE):
    """Fraction of KV pairs to evict so that the quantized cache fits in `memory_budget`.

    `memory_budget` is a fraction of the document's bf16 cache size. Returns 0 if
    everything fits, in which case the cache ends up smaller than the budget.
    """
    slots = num_layers * num_heads * num_tokens
    budget_bytes = memory_budget * slots * 2 * head_dim * 2
    spare = budget_bytes - key_metadata_bytes(num_layers, num_heads, head_dim)
    kept_slots = spare / slot_bytes(head_dim, bits, group_size)
    return min(max(1 - kept_slots / slots, 0.0), 0.99)


def _round_trip(x, low, high, bits):
    scale = (high - low) / (2**bits - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round((x - low) / scale) * scale + low


def quantize_keys(keys, kept, bits, group_size=GROUP_SIZE):
    """Round keys (batch, head, token, channel) per channel, over each run of
    `group_size` kept tokens of a head. Evicted tokens are left as they are."""
    x = keys.float()
    batch, heads, tokens, dim = x.shape
    n_groups = math.ceil(tokens / group_size)
    rank = kept.long().cumsum(dim=-1) - 1
    group = torch.where(kept, torch.div(rank, group_size, rounding_mode="floor"), n_groups)
    index = group.unsqueeze(-1).expand(-1, -1, -1, dim)
    shape = (batch, heads, n_groups + 1, dim)  # the extra group collects evicted tokens
    low = x.new_full(shape, float("inf")).scatter_reduce(2, index, x, "amin")
    high = x.new_full(shape, float("-inf")).scatter_reduce(2, index, x, "amax")
    rounded = _round_trip(x, low.gather(2, index), high.gather(2, index), bits)
    return torch.where(kept.unsqueeze(-1), rounded, x).to(keys.dtype)


def quantize_values(values, bits, group_size=GROUP_SIZE):
    """Round values (batch, head, token, channel) per token, over each group of `group_size` channels."""
    x = values.float()
    *lead, dim = x.shape
    if dim % group_size:
        raise ValueError(f"head_dim {dim} is not a multiple of group_size {group_size}")
    grouped = x.reshape(*lead, dim // group_size, group_size)
    low, high = grouped.amin(dim=-1, keepdim=True), grouped.amax(dim=-1, keepdim=True)
    return _round_trip(grouped, low, high, bits).reshape(x.shape).to(values.dtype)


@dataclass
class QuantizedRestoreKVPress(RestoreKVPress):
    bits: int = 4
    group_size: int = GROUP_SIZE
    stats: dict = field(init=False, default_factory=dict, repr=False)

    def compress_post(self, model):
        config = model.config
        module = model.model.layers[0].self_attn
        shape = (config.num_hidden_layers, config.num_key_value_heads, module.head_dim)
        requested = self.compression_ratio
        evict = eviction_ratio(1 - requested, self.bits, self.context_length, *shape, self.group_size)
        self.compression_ratio = evict
        try:
            super().compress_post(model)
        finally:
            self.compression_ratio = requested
        self._quantize_kept(model, evict, shape)

    def _quantize_kept(self, model, evict, shape):
        masks = []
        for layer in model.model.layers:
            module = layer.self_attn
            cache_layer = self._cache.layers[module.layer_idx]
            kept = torch.ones(cache_layer.keys.shape[:3], dtype=torch.bool, device=cache_layer.keys.device)
            batch, head, token = module.masked_key_indices
            kept[batch.to(kept.device), head.to(kept.device), token.to(kept.device)] = False
            masks.append(kept)
            cache_layer.keys = quantize_keys(cache_layer.keys, kept, self.bits, self.group_size)
            cache_layer.values = quantize_values(cache_layer.values, self.bits, self.group_size)
        num_layers, num_heads, head_dim = shape
        self.stats = {
            "bits": self.bits,
            "group_size": self.group_size,
            "eviction_ratio": evict,
            "kept_slots": sum(int(m.sum()) for m in masks),
            "bytes": cache_bytes(masks, head_dim, self.bits, self.group_size),
            "budget_bytes": (1 - self.compression_ratio) * num_layers * num_heads * self.context_length * 4 * head_dim,
        }
