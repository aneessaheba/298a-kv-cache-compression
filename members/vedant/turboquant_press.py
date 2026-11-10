"""TurboQuant as a kvpress press, so it runs in the shared harness (harness/run_longbench.py).

After each layer's forward pass over the document, the document's keys and values are
packed with Vedant's `TurboQuantCacheLayer` (src/turboquant/kv_cache.py) and replaced by
their decoded values:
- keys: TurboQuant-Prod ((bits-1)-bit MSE codes plus a 1-bit QJL residual), or
  TurboQuant-MSE at `bits` when `qjl_keys=False`;
- values: TurboQuant-MSE at `bits`;
- vector norms (and QJL residual norms) in fp16.

This follows the shared protocol used for every other method:
- the document's own attention during prefill runs at full precision, and only what is
  stored is quantized (as in the TurboQuant and KIVI papers);
- the question and the generated answer stay in bf16, uncompressed;
- the cache holds the decoded bf16 tensors, so memory is computed, not measured, as for
  KVzip and RestoreKV. `context_bytes_per_token` gives the packed size, which the harness
  uses for `kv_cache_gb`; `stats["packed_bytes"]` is the exact count from the last record.

Seeds per layer follow `TransformersTurboQuantCache`, so both paths quantize identically.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from kvpress.presses.base_press import BasePress

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from turboquant.mixed import head_token_bytes, quantize_document  # noqa: E402

# Tokens packed at a time. Every vector is quantized on its own, so chunking gives identical
# results; it only caps the temporary int64 tensors used for bit packing on 32k-token prompts.
CHUNK_TOKENS = 4096


@dataclass
class TurboQuantPress(BasePress):
    bits: float = 2.5
    qjl_keys: bool = True
    seed: int = 42
    layer_configs: list[dict] | None = None
    stats: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.bits not in (2, 2.5, 3, 3.5, 4):
            raise ValueError("bits must be 2, 2.5, 3, 3.5 or 4")
        if self.qjl_keys and self.bits < 2:
            raise ValueError("QJL keys need at least 2 bits")
        if self.layer_configs is not None:
            if not self.layer_configs or [r['layer'] for r in self.layer_configs] != list(range(len(self.layer_configs))):
                raise ValueError("adaptive configurations must cover consecutive layers starting at 0")
            for row in self.layer_configs:
                if row['bits'] not in (2,3,4) or not isinstance(row['qjl_keys'],bool):
                    raise ValueError("adaptive layers require 2/3/4 bits and boolean qjl_keys")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        bits, qjl = self.bits, self.qjl_keys
        if self.layer_configs is not None:
            if len(self.layer_configs) != module.config.num_hidden_layers:
                raise ValueError("adaptive allocation must cover every model layer")
            setting = self.layer_configs[module.layer_idx]
            bits, qjl = setting['bits'], setting['qjl_keys']
        decoded_keys, decoded_values, stats = quantize_document(
            keys, values, bits, qjl_keys=qjl,
            seed=self.seed + module.layer_idx * 100_003, chunk_tokens=CHUNK_TOKENS)
        if module.layer_idx == 0 or not self.stats:  # layer 0 starts a new document
            self.stats = {"bits": self.bits, "qjl_keys": self.qjl_keys, "packed_bytes": 0,
                          "selector_bytes": 0, "context_tokens": keys.shape[-2], "layers": 0,
                          "layer_settings": [],
                          "memory_definition": "computed_packed_document_codes_norms_selectors; BF16_decode_cache_resident"}
        self.stats["packed_bytes"] += stats["packed_bytes"]
        self.stats["selector_bytes"] += stats["selector_bytes"]
        self.stats["layers"] += 1
        self.stats['layer_settings'].append({'layer':module.layer_idx, 'bits':bits, 'qjl_keys':qjl})
        return decoded_keys, decoded_values

    def context_bytes_per_token(self, config) -> float:
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        if self.layer_configs is not None:
            if len(self.layer_configs) != config.num_hidden_layers:
                raise ValueError("adaptive allocation must cover every model layer")
            return config.num_key_value_heads * sum(head_token_bytes(head_dim,r['bits'],r['qjl_keys']) for r in self.layer_configs)
        per_head = head_token_bytes(head_dim, self.bits, self.qjl_keys)
        return config.num_hidden_layers * config.num_key_value_heads * per_head
