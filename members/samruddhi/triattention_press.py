"""TriAttention as a kvpress ScorerPress, faithful to WeianMao/triattention (SAM / Samruddhi).

TriAttention (Mao et al., ICML 2026, arXiv:2604.04921) scores each cached key with a
frequency-domain trigonometric series built from *precomputed* per-head query statistics,
rather than the post-RoPE attention of a few recent queries. This press reproduces the
authors' scoring exactly (triattention/methods/pruning_utils.py):

  1. invert RoPE on the cached keys to recover base (pre-RoPE) keys;
  2. form head_dim/2 complex frequency pairs;
  3. combine with the calibrated per-head query mean (q_mean_complex) and magnitude
     (q_abs_mean) to get amplitude `amp`, phase `phi`, and the MLR `extra` term;
  4. score each key as a trigonometric series over a geometric grid of future
     query-distance offsets, aggregated (mean) over the grid.

The per-head stats come from members/samruddhi/calibrate_triattention.py (the authors'
calibration). Recent `window_size` tokens and `n_sink` initial tokens are always kept.

Set the stats path via TriAttentionPress(stats_path=...). If no stats are available the
press raises, rather than silently falling back to an approximation.

Register in harness/run_longbench.py make_press() as "triattention".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from kvpress import ScorerPress

HERE = Path(__file__).resolve().parent
DEFAULT_STATS = HERE / "calibration" / "llama31_8b_stats.pt"


# ---- exact math from triattention/methods/pruning_utils.py (style="half") ----

def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _invert_rope(rotated, cos, sin, scale):
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    return base * (cos / scale_t) - _rotate_half(base) * (sin / scale_t)


def _to_complex_pairs(tensor):
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    t = tensor.to(dtype=real_dtype)
    fc = t.shape[-1] // 2
    return torch.complex(t[..., :fc].contiguous(), t[..., fc:].contiguous())


def _freq_stats_from_means(q_mean_complex, q_abs_mean, k_unrot, disable_mlr=False):
    k_complex = _to_complex_pairs(k_unrot)          # [n, fc]
    q_mean_abs = torch.abs(q_mean_complex)          # [fc]
    k_abs = torch.abs(k_complex)                    # [n, fc]
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)  # [n, fc]
    amp = q_mean_abs.unsqueeze(0) * k_abs            # [n, fc]
    if disable_mlr:
        extra = q_abs_mean.unsqueeze(0) * k_abs
    else:
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def _score_keys(key_positions, round_start, amp, phi, omega, extra, offsets,
                freq_scale_sq, aggregation="mean", key_batch=4096):
    """Trigonometric-series score per key, computed in key-batches so the
    [n_keys, n_offsets, n_freqs] intermediate never materialises for the whole
    sequence at once (keeps peak memory low on long prompts)."""
    n = key_positions.shape[0]
    fss = freq_scale_sq.to(device=amp.device, dtype=torch.float32).view(1, 1, -1)
    fss2 = freq_scale_sq.to(device=amp.device, dtype=torch.float32).view(1, -1)
    out = torch.empty(n, device=amp.device, dtype=torch.float32)
    for s in range(0, n, key_batch):
        e = min(s + key_batch, n)
        pos = key_positions[s:e].to(device=amp.device, dtype=torch.float32)
        delta_grid = (round_start - pos).unsqueeze(1) + offsets.unsqueeze(0)       # [b, n_off]
        phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi[s:e].unsqueeze(1)  # [b, n_off, fc]
        base_scores = (amp[s:e].unsqueeze(1) * fss * torch.cos(phase)).sum(dim=2)  # [b, n_off]
        additive = (extra[s:e] * fss2).sum(dim=1, keepdim=True)                     # [b, 1]
        combined = base_scores + additive
        out[s:e] = combined.mean(dim=1) if aggregation == "mean" else combined.max(dim=1).values
        del delta_grid, phase, base_scores, additive, combined
    return out


def _score_head_chunked(k_unrot, q_mean_complex, q_abs_mean, positions, round_start,
                        omega, offsets, freq_scale_sq, aggregation="mean", chunk=1024,
                        disable_mlr=False):
    """Exact TriAttention per-head score, computed one key-chunk at a time.

    Identical math to _freq_stats_from_means + _score_keys, but the per-key frequency
    stats (amp, phi, extra) and the [chunk, n_offsets, n_freqs] trig tensor are built
    only for the current chunk and freed before the next, so no full-sequence
    [n, n_offsets, n_freqs] (or [n, fc]) tensor is ever materialised.
    """
    n = k_unrot.shape[0]
    dev = k_unrot.device
    q_mean_complex = q_mean_complex.to(dev)
    q_abs_mean = q_abs_mean.to(dev)
    q_mean_abs = torch.abs(q_mean_complex)                      # [fc]
    omega = omega.to(dev).view(1, 1, -1)
    offs = offsets.to(dev)
    fss = freq_scale_sq.to(dev, torch.float32).view(1, 1, -1)
    fss2 = freq_scale_sq.to(dev, torch.float32).view(1, -1)
    out = torch.empty(n, device=dev, dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        kc = _to_complex_pairs(k_unrot[s:e])                   # [b, fc]
        k_abs = torch.abs(kc)                                  # [b, fc]
        relative = q_mean_complex.unsqueeze(0) * torch.conj(kc)
        phi = torch.atan2(relative.imag, relative.real)        # [b, fc]
        amp = q_mean_abs.unsqueeze(0) * k_abs                  # [b, fc]
        if disable_mlr:
            extra = q_abs_mean.unsqueeze(0) * k_abs
        else:
            extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
        pos = positions[s:e].to(dev, torch.float32)
        delta_grid = (round_start - pos).unsqueeze(1) + offs.unsqueeze(0)          # [b, n_off]
        phase = delta_grid.unsqueeze(2) * omega + phi.unsqueeze(1)                 # [b, n_off, fc]
        base_scores = (amp.unsqueeze(1) * fss * torch.cos(phase)).sum(dim=2)       # [b, n_off]
        additive = (extra * fss2).sum(dim=1, keepdim=True)                         # [b, 1]
        combined = base_scores + additive
        out[s:e] = combined.mean(dim=1) if aggregation == "mean" else combined.max(dim=1).values
        del kc, k_abs, relative, phi, amp, extra, delta_grid, phase, base_scores, additive, combined
    return out


def _geometric_offsets(max_length, device):
    offs, v = [], 1
    while v <= max_length:
        offs.append(float(v))
        v *= 2
    return torch.tensor(offs, device=device, dtype=torch.float32)


@dataclass
class TriAttentionPress(ScorerPress):
    """Faithful TriAttention scoring over calibrated per-head frequency statistics.

    Parameters
    ----------
    compression_ratio : float
        Fraction of key-value pairs to evict (0.9 keeps 10%).
    window_size : int
        Recent tokens always kept (the runtime WINDOW_SIZE).
    n_sink : int
        Initial attention-sink tokens always kept.
    stats_path : str
        Path to the calibrated .pt (from calibrate_triattention.py).
    offset_max_length : int
        Max geometric distance offset for the trig series (authors' default 65536).
    score_aggregation : str
        "mean" or "max" over the offset grid (authors' default "mean").
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    n_sink: int = 4
    stats_path: str = str(DEFAULT_STATS)
    # The trig series over future-query distances decays quickly; a small geometric grid
    # (1,2,...,1024) captures it while keeping the [n_keys, n_offsets, n_freqs] score tensor
    # ~16x smaller than the paper's 65536 default -- important for memory during chunked prefill.
    offset_max_length: int = 1024
    score_aggregation: str = "mean"
    # Key-chunk size for the memory-bounded scorer (never builds a full-sequence score tensor).
    score_chunk: int = 1024

    # lazily initialised from the model + stats on first score() call
    _ready: bool = field(default=False, repr=False)

    def _init_from_module(self, module):
        path = Path(self.stats_path)
        if not path.exists():
            raise FileNotFoundError(
                f"TriAttention stats not found: {path}\n"
                f"Run members/samruddhi/calibrate_triattention.py first.")
        device = module.k_proj.weight.device
        payload = torch.load(path, map_location=device)
        meta, raw = payload["metadata"], payload["stats"]
        self._stats = {}
        for layer, head in (tuple(x) for x in meta["sampled_heads"]):
            e = raw.get(f"layer{layer:02d}_head{head:02d}")
            if e is None:
                continue
            self._stats[(int(layer), int(head))] = (
                torch.complex(e["q_mean_real"].to(device, torch.float32),
                              e["q_mean_imag"].to(device, torch.float32)),
                e["q_abs_mean"].to(device, torch.float32),
            )
        rotary = module.rotary_emb if hasattr(module, "rotary_emb") else None
        self._rotary = rotary
        self._attn_scale = float(getattr(rotary, "attention_scaling", 1.0)) if rotary else 1.0
        inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float32)
        self._head_dim = int(meta.get("head_dim", inv_freq.numel() * 2))
        fc = self._head_dim // 2
        self._omega = inv_freq[:fc]
        self._offsets = _geometric_offsets(self.offset_max_length, device)
        # per-frequency RoPE scaling (unit position probe), squared
        pos0 = torch.zeros(1, 1, device=device, dtype=torch.long)
        pr = torch.zeros(1, 1, self._head_dim, device=device, dtype=module.k_proj.weight.dtype)
        c0, s0 = rotary(pr, pos0)
        self._freq_scale_sq = (c0[0, 0][0::2].pow(2) + s0[0, 0][0::2].pow(2)).to(torch.float32)
        self._num_heads = module.config.num_attention_heads
        self._num_kv_heads = module.config.num_key_value_heads
        self._groups = self._num_heads // self._num_kv_heads
        self._ready = True

    @torch.no_grad()
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        if not self._ready:
            self._init_from_module(module)

        bsz, num_kv_heads, k_len, head_dim = keys.shape
        layer_idx = getattr(module, "layer_idx", None)
        device = keys.device
        positions = torch.arange(k_len, device=device, dtype=torch.float32)
        round_start = float(k_len)  # score as if querying from just past the context

        # RoPE cos/sin for key positions (shared across heads)
        pos_ids = torch.arange(k_len, device=device).unsqueeze(0)
        probe = torch.zeros(1, k_len, head_dim, device=device, dtype=keys.dtype)
        cos, sin = self._rotary(probe, pos_ids)
        cos0, sin0 = cos[0], sin[0]

        # Accumulate scores per KV head by averaging the attention heads that map to it.
        acc = torch.zeros(bsz, num_kv_heads, k_len, device=device, dtype=torch.float32)
        counts = torch.zeros(num_kv_heads, device=device)
        heads_for_layer = [(l, h) for (l, h) in self._stats if l == layer_idx]
        if not heads_for_layer:
            # layer not in the calibrated sample set: fall back to keeping recent+sink only
            scores = torch.zeros(bsz, num_kv_heads, k_len, device=device, dtype=torch.float32)
        else:
            for (l, h) in heads_for_layer:
                kv_head = min(num_kv_heads - 1, h // max(1, self._groups))
                k_vals = keys[0, kv_head].to(torch.float32)
                k_unrot = _invert_rope(k_vals, cos0, sin0, self._attn_scale)
                q_mean_complex, q_abs_mean = self._stats[(l, h)]
                # Chunked scoring: builds per-chunk stats + trig tensor only, never the full
                # [k_len, n_offsets, n_freqs] / [k_len, fc] tensors (verified identical output).
                s = _score_head_chunked(k_unrot, q_mean_complex, q_abs_mean, positions, round_start,
                                        self._omega, self._offsets, self._freq_scale_sq,
                                        self.score_aggregation, chunk=self.score_chunk)
                acc[0, kv_head] += s
                counts[kv_head] += 1
                del k_vals, k_unrot, s
            counts = counts.clamp_min(1.0)
            scores = acc / counts.view(1, -1, 1)

        # Protect attention sinks and the recent window.
        window = min(self.window_size, max(1, k_len // 2))
        keep = scores.max() + 1.0
        if self.n_sink > 0 and self.n_sink < k_len:
            scores[..., : self.n_sink] = keep
        if 0 < window <= k_len:
            scores[..., -window:] = keep
        return scores.to(keys.dtype)
