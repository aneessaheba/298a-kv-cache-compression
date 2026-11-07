"""RestoreKV plus KIVI-style quantization of the kept cache, at equal memory (SAM-96).

The question: for the same number of cache bytes, is it better to keep few tokens
at full precision (RestoreKV) or more tokens at 4 or 2 bits?

`compression_ratio` is the share of the bf16 document cache's bytes to remove, as in
the rest of the harness, so 0.95 means a 5% memory budget. The press works out how
many KV pairs fit in that budget at the chosen bit-width, evicts the rest with
RestoreKV, then quantizes what is left, including the restore tokens:

- Keys: one scale and zero point per channel, over the kept tokens of each head.
- Values: one scale and zero point per token, over its channels.

Scales and zero points are stored in bf16 and counted in the budget. As in kvpress,
evicted pairs are masked and tokens are rounded in place rather than stored in packed
form, so accuracy is real but memory is computed. Question tokens added after the
document are not compressed.
"""

from dataclasses import dataclass, field

import torch
from kvpress import RestoreKVPress

METADATA_BYTES = 4  # scale + zero point, both bf16


def slot_bytes(head_dim, bits):
    """Bytes for one kept token in one head: quantized key and value, plus the value's metadata."""
    return 2 * head_dim * bits / 8 + METADATA_BYTES


def key_metadata_bytes(num_layers, num_heads, head_dim):
    """Fixed cost of the per-channel key scales and zero points."""
    return num_layers * num_heads * head_dim * METADATA_BYTES


def eviction_ratio(memory_budget, bits, num_tokens, num_layers, num_heads, head_dim):
    """Fraction of KV pairs to evict so that the quantized cache fits in `memory_budget`.

    `memory_budget` is a fraction of the document's bf16 cache size. Returns 0 if
    everything fits, in which case the cache ends up smaller than the budget.
    """
    slots = num_layers * num_heads * num_tokens
    budget_bytes = memory_budget * slots * 2 * head_dim * 2
    kept_slots = (budget_bytes - key_metadata_bytes(num_layers, num_heads, head_dim)) / slot_bytes(head_dim, bits)
    return min(max(1 - kept_slots / slots, 0.0), 0.99)


def _round_trip(x, low, high, bits):
    scale = (high - low) / (2**bits - 1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round((x - low) / scale) * scale + low


def quantize_keys(keys, kept, bits):
    """Round keys (batch, head, token, channel) using each channel's range over the kept tokens."""
    x = keys.float()
    mask = kept.unsqueeze(-1)
    low = x.masked_fill(~mask, float("inf")).amin(dim=2, keepdim=True)
    high = x.masked_fill(~mask, float("-inf")).amax(dim=2, keepdim=True)
    empty = ~kept.any(dim=2, keepdim=True).unsqueeze(-1)  # a head with nothing kept
    low, high = low.masked_fill(empty, 0.0), high.masked_fill(empty, 0.0)
    rounded = _round_trip(x, low, high, bits)
    return torch.where(mask, rounded, x).to(keys.dtype)


def quantize_values(values, bits):
    """Round values (batch, head, token, channel) using each token's own range."""
    x = values.float()
    rounded = _round_trip(x, x.amin(dim=3, keepdim=True), x.amax(dim=3, keepdim=True), bits)
    return rounded.to(values.dtype)


@dataclass
class QuantizedRestoreKVPress(RestoreKVPress):
    bits: int = 4
    stats: dict = field(init=False, default_factory=dict, repr=False)

    def compress_post(self, model):
        config = model.config
        module = model.model.layers[0].self_attn
        shape = (config.num_hidden_layers, config.num_key_value_heads, module.head_dim)
        requested = self.compression_ratio
        evict = eviction_ratio(1 - requested, self.bits, self.context_length, *shape)
        self.compression_ratio = evict
        try:
            super().compress_post(model)
        finally:
            self.compression_ratio = requested
        self._quantize_kept(model, evict, shape)

    def _quantize_kept(self, model, evict, shape):
        kept_slots = 0
        for layer in model.model.layers:
            module = layer.self_attn
            cache_layer = self._cache.layers[module.layer_idx]
            kept = torch.ones(cache_layer.keys.shape[:3], dtype=torch.bool, device=cache_layer.keys.device)
            batch, head, token = module.masked_key_indices
            kept[batch.to(kept.device), head.to(kept.device), token.to(kept.device)] = False
            kept_slots += int(kept.sum())
            cache_layer.keys = quantize_keys(cache_layer.keys, kept, self.bits)
            cache_layer.values = quantize_values(cache_layer.values, self.bits)
        num_layers, num_heads, head_dim = shape
        self.stats = {
            "bits": self.bits,
            "eviction_ratio": evict,
            "kept_slots": kept_slots,
            "bytes": kept_slots * slot_bytes(head_dim, self.bits) + key_metadata_bytes(*shape),
            "budget_bytes": (1 - self.compression_ratio) * num_layers * num_heads * self.context_length * 4 * head_dim,
        }
