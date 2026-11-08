"""Packed TurboQuant KV cache and Hugging Face cache adapter.

The adapter stores only packed codes between calls. It reconstructs the full
key/value tensors for the stock attention implementation, making it suitable
for accuracy experiments and exact persistent-cache byte accounting. A fused
attention kernel would be required to turn the storage saving into the same
peak-memory and latency saving during attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .mse import TurboQuantMSE
from .packing import (
    PackedMSECodes,
    PackedProductCodes,
    pack_mse_codes,
    pack_product_codes,
    unpack_mse_codes,
    unpack_product_codes,
)
from .product import TurboQuantProd

try:
    from transformers import Cache
    from transformers.cache_utils import CacheLayerMixin
except ImportError:  # pragma: no cover - optional dependency
    Cache = object  # type: ignore[assignment,misc]
    CacheLayerMixin = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class CacheStorageStats:
    tokens: int
    key_bytes: int
    value_bytes: int
    full_precision_bytes: int

    @property
    def compressed_bytes(self) -> int:
        return self.key_bytes + self.value_bytes

    @property
    def ratio(self) -> float:
        return self.compressed_bytes / self.full_precision_bytes if self.full_precision_bytes else 0.0

    @property
    def bytes_per_token(self) -> float:
        return self.compressed_bytes / self.tokens if self.tokens else 0.0


@dataclass(frozen=True)
class LayerQuantConfig:
    """Quantization choice for one transformer layer."""

    bits: int
    qjl_keys: bool

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 4:
            raise ValueError("adaptive layer bits must be 2, 3, or 4")


class TurboQuantCacheLayer(CacheLayerMixin):  # type: ignore[misc]
    """One dynamic `[batch, kv_heads, tokens, head_dim]` cache layer."""

    is_sliding = False
    is_croppable = False
    supports_early_init = True

    def __init__(
        self,
        head_dim: int,
        bits: int,
        *,
        qjl_keys: bool = True,
        seed: int = 0,
        norm_dtype: torch.dtype = torch.float16,
    ) -> None:
        if qjl_keys and bits < 2:
            raise ValueError("QJL keys require at least 2 total bits")
        if not 1 <= bits <= 8:
            raise ValueError("bits must be between 1 and 8")
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.qjl_keys = qjl_keys
        self.seed = seed
        self.norm_dtype = norm_dtype
        self.key_chunks: list[PackedMSECodes | PackedProductCodes] = []
        self.value_chunks: list[PackedMSECodes] = []
        self.chunk_token_counts: list[int] = []
        self._key_quantizer: TurboQuantMSE | TurboQuantProd | None = None
        self._value_quantizer: TurboQuantMSE | None = None
        self._batch_size = 0
        self._num_heads = 0
        self._source_element_size = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("cache tensors must have shape [batch, heads, tokens, head_dim]")
        if key_states.shape != value_states.shape:
            raise ValueError("key and value cache tensors must have identical shapes")
        if key_states.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim {self.head_dim}, got {key_states.shape[-1]}")
        self.dtype, self.device = key_states.dtype, key_states.device
        quant_dtype = torch.float64 if key_states.dtype == torch.float64 else torch.float32
        quantizer_type = TurboQuantProd if self.qjl_keys else TurboQuantMSE
        self._key_quantizer = quantizer_type(
            self.head_dim,
            self.bits,
            seed=self.seed,
            device=self.device,
            dtype=quant_dtype,
        )
        self._value_quantizer = TurboQuantMSE(
            self.head_dim,
            self.bits,
            seed=self.seed + 10_000,
            device=self.device,
            dtype=quant_dtype,
        )
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._source_element_size = key_states.element_size()
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if key_states.shape[:2] != (self._batch_size, self._num_heads):
            raise ValueError("batch size and KV-head count must remain constant")
        assert self._key_quantizer is not None and self._value_quantizer is not None

        key_codes = self._key_quantizer.quantize(key_states)
        if self.qjl_keys:
            assert isinstance(self._key_quantizer, TurboQuantProd)
            packed_keys: PackedMSECodes | PackedProductCodes = pack_product_codes(
                key_codes, self.bits - 1, norm_dtype=self.norm_dtype
            )
        else:
            packed_keys = pack_mse_codes(key_codes, self.bits, norm_dtype=self.norm_dtype)
        packed_values = pack_mse_codes(
            self._value_quantizer.quantize(value_states),
            self.bits,
            norm_dtype=self.norm_dtype,
        )
        self.key_chunks.append(packed_keys)
        self.value_chunks.append(packed_values)
        self.chunk_token_counts.append(key_states.shape[-2])
        return self.dequantize()

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized or not self.key_chunks:
            empty = torch.empty(
                (self._batch_size, self._num_heads, 0, self.head_dim),
                device=getattr(self, "device", "cpu"),
                dtype=getattr(self, "dtype", torch.float32),
            )
            return empty, empty.clone()
        assert self._key_quantizer is not None and self._value_quantizer is not None
        keys = []
        values = []
        for key_codes, value_codes in zip(self.key_chunks, self.value_chunks):
            if self.qjl_keys:
                assert isinstance(key_codes, PackedProductCodes)
                decoded_key_codes = unpack_product_codes(
                    key_codes, device=self.device, dtype=self._key_quantizer.dtype
                )
            else:
                assert isinstance(key_codes, PackedMSECodes)
                decoded_key_codes = unpack_mse_codes(
                    key_codes, device=self.device, dtype=self._key_quantizer.dtype
                )
            keys.append(self._key_quantizer.dequantize(decoded_key_codes).to(self.dtype))
            value_unpacked = unpack_mse_codes(
                value_codes, device=self.device, dtype=self._value_quantizer.dtype
            )
            values.append(self._value_quantizer.dequantize(value_unpacked).to(self.dtype))
        return torch.cat(keys, dim=-2), torch.cat(values, dim=-2)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return sum(self.chunk_token_counts)

    def get_max_length(self) -> int:
        return -1

    def storage_stats(self) -> CacheStorageStats:
        key_bytes = sum(chunk.nbytes for chunk in self.key_chunks)
        value_bytes = sum(chunk.nbytes for chunk in self.value_chunks)
        tokens = self.get_seq_length()
        full_bytes = (
            2
            * self._batch_size
            * self._num_heads
            * tokens
            * self.head_dim
            * self._source_element_size
        )
        return CacheStorageStats(tokens, key_bytes, value_bytes, full_bytes)

    def reset(self) -> None:
        self.key_chunks.clear()
        self.value_chunks.clear()
        self.chunk_token_counts.clear()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() > 0:
            raise NotImplementedError("beam search is outside the fixed greedy-decoding protocol")

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove != 0:
            raise NotImplementedError("cache cropping is not implemented for packed chunks")

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats != 1 and self.get_seq_length() > 0:
            raise NotImplementedError("beam expansion is outside the fixed greedy-decoding protocol")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() > 0:
            raise NotImplementedError("batch selection is not implemented for packed chunks")


class TransformersTurboQuantCache(Cache):  # type: ignore[misc]
    """Hugging Face `Cache` container for fixed-model accuracy experiments."""

    def __init__(
        self,
        config: Any,
        bits: int | None = None,
        *,
        qjl_keys: bool = True,
        seed: int = 0,
        norm_dtype: torch.dtype = torch.float16,
        layer_configs: list[LayerQuantConfig | dict[str, Any]] | None = None,
    ) -> None:
        if Cache is object:
            raise ImportError("install the 'hf' extra to use TransformersTurboQuantCache")
        text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        num_layers = text_config.num_hidden_layers
        if layer_configs is None:
            if bits is None:
                raise ValueError("bits are required when layer_configs are not supplied")
            resolved_configs = [LayerQuantConfig(bits, qjl_keys) for _ in range(num_layers)]
        else:
            if len(layer_configs) != num_layers:
                raise ValueError(
                    f"expected {num_layers} adaptive layer configurations, got {len(layer_configs)}"
                )
            resolved_configs = []
            for row in layer_configs:
                if isinstance(row, LayerQuantConfig):
                    resolved_configs.append(row)
                    continue
                if "qjl_keys" not in row or not isinstance(row["qjl_keys"], bool):
                    raise ValueError("every adaptive layer must contain a boolean qjl_keys field")
                resolved_configs.append(
                    LayerQuantConfig(bits=int(row["bits"]), qjl_keys=row["qjl_keys"])
                )
        layers = [
            TurboQuantCacheLayer(
                head_dim,
                layer_config.bits,
                qjl_keys=layer_config.qjl_keys,
                seed=seed + layer_idx * 100_003,
                norm_dtype=norm_dtype,
            )
            for layer_idx, layer_config in enumerate(resolved_configs)
        ]
        super().__init__(layers=layers)
        self.layer_configs = resolved_configs
        self.bits = bits if layer_configs is None else sum(row.bits for row in resolved_configs) / num_layers
        self.qjl_keys = qjl_keys if layer_configs is None else None

    def storage_stats(self) -> CacheStorageStats:
        rows = [layer.storage_stats() for layer in self.layers]
        tokens = max((row.tokens for row in rows), default=0)
        return CacheStorageStats(
            tokens=tokens,
            key_bytes=sum(row.key_bytes for row in rows),
            value_bytes=sum(row.value_bytes for row in rows),
            full_precision_bytes=sum(row.full_precision_bytes for row in rows),
        )

    def quantizer_state_bytes(self) -> int:
        """Count persistent rotation, projection and codebook tensors separately from KV payload."""
        seen: set[int] = set()

        def count(value: Any) -> int:
            if isinstance(value, torch.Tensor):
                key = value.data_ptr()
                if key in seen:
                    return 0
                seen.add(key)
                return value.numel() * value.element_size()
            if isinstance(value, (TurboQuantMSE, TurboQuantProd)) or type(value).__name__ == "SphereCodebook":
                return sum(count(item) for item in vars(value).values())
            return 0

        return sum(count(layer._key_quantizer) + count(layer._value_quantizer) for layer in self.layers)
