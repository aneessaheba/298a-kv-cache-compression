"""
RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache
(Zhang, Guo, Benini, Li - arXiv 2605.08317, May 2026)

Team-6 re-implementation as a kvpress press (no official code is public).

What it does (once, at the end of prefill, per layer, per KV head):
  Stage 1  score units
           - V cache, per token  : w_t = attention mass from the last `window_size` queries,
                                   summed over the GQA group, smoothed with AvgPool1d(k=5)
           - K cache, per channel: w_c = ||Q[:,c]|| * ||K[:,c]|| / sqrt(d)
  Stage 2  V allocation over tokens:  choose b_t in {0,2,4,8,16} minimising sum w_t*eps_t(b_t)
           under the V bit budget (Lagrangian + bisection on lambda). b_t = 0 -> token evicted.
  Stage 3  K allocation over channels, only over tokens kept in stage 2, same solver.
           b_c = 0 -> channel removed (zeroed).

Budget: the team's `compression_ratio` r means "remove r of the FP16 cache", so every head gets
          (1 - r) * T * d * 16 bits for V and the same for K   (r_K = 0.5 split, as in the paper).
This means RDKV hits 0.75 / 0.8 / 0.9 / 0.95 exactly in payload bits; the small quantiser
metadata (scales/zero-points, keep bitmaps) is reported on top in `kv_cache_gb`.

Simulation note (same as every other kvpress method in our comparison):
  - quantised units are quantise->dequantise'd in place (exact numerics of the quantiser),
  - evicted tokens are masked with kvpress' `masked_key_indices` (same mechanism AdaKVPress uses),
  - removed K channels are zeroed.
  So GPU memory does not physically drop; we report the computed size, like the rest of the team.

Deviations from the paper (write these in the report):
  1. eps_u(b) is the *exact* per-unit RMS quantisation error computed online, instead of the
     paper's offline calibrated table (32 LongBench prefixes). Same objective, no calibration set.
  2. The last `window_size` context tokens are always kept at FP16 (SnapKV convention), and their
     cost is charged to the V budget.
  3. No TriZone packing / Triton kernel (speed is not part of our comparison).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

try:  # kvpress >= 0.2
    from kvpress.presses.base_press import BasePress
except ImportError:  # pragma: no cover
    from kvpress import BasePress  # type: ignore

try:
    from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
except ImportError:  # pragma: no cover
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def repeat_kv(h, n_rep):
        b, kvh, s, d = h.shape
        if n_rep == 1:
            return h
        return h[:, :, None].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


BITSET = (0, 2, 4, 8, 16)  # 0 = evict, 16 = keep FP16


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def fake_quant(x: torch.Tensor, bits: int, dim: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Asymmetric uniform min-max quantise -> dequantise along `dim`.
    `mask` (broadcastable to x, True = participates) restricts the min/max statistics."""
    if bits >= 16:
        return x
    if mask is None:
        xmin = x.amin(dim=dim, keepdim=True)
        xmax = x.amax(dim=dim, keepdim=True)
    else:
        xmin = x.masked_fill(~mask, float("inf")).amin(dim=dim, keepdim=True)
        xmax = x.masked_fill(~mask, float("-inf")).amax(dim=dim, keepdim=True)
        xmin = torch.where(torch.isfinite(xmin), xmin, torch.zeros_like(xmin))
        xmax = torch.where(torch.isfinite(xmax), xmax, torch.zeros_like(xmax))
    levels = 2**bits - 1
    scale = (xmax - xmin).clamp_min(1e-8) / levels
    q = ((x - xmin) / scale).round().clamp_(0, levels)
    return q * scale + xmin


@torch.no_grad()
def allocate_bits(
    D: torch.Tensor,  # [N, U, nb] weighted distortion w_u * eps_u(b)
    bitset: torch.Tensor,  # [nb] ascending, starts with 0
    target_avg: torch.Tensor,  # [N] average bits per unit allowed
    unit_mask: torch.Tensor | None = None,  # [N, U] True = unit exists
    max_iter: int = 64,
    tol: float = 1e-2,
) -> torch.Tensor:
    """Discrete reverse water-filling (Lagrangian relaxation + bisection on lambda), solved for
    all N heads in parallel. Returns chosen bit index [N, U] (index into bitset).
    The returned allocation always satisfies avg_bits <= target_avg."""
    N, U, nb = D.shape
    if unit_mask is None:
        unit_mask = torch.ones(N, U, dtype=torch.bool, device=D.device)
    n_units = unit_mask.sum(-1).clamp_min(1).float()
    target = target_avg.clamp(min=0.0, max=float(bitset[-1]))

    # normalise distortions per head for numerical range
    scale = D.masked_fill(~unit_mask[..., None], 0).amax(dim=(1, 2)).clamp_min(1e-20)
    Dn = D / scale[:, None, None]
    b = bitset.to(D.dtype)

    def solve(log2_lam):
        lam = torch.exp2(log2_lam)[:, None, None]
        idx = torch.argmin(Dn + lam * b, dim=-1)  # ties -> fewer bits
        idx = torch.where(unit_mask, idx, torch.zeros_like(idx))
        avg = b[idx].masked_fill(~unit_mask, 0).sum(-1) / n_units
        return idx, avg

    lo = torch.full((N,), -40.0, device=D.device)  # small lambda -> many bits
    hi = torch.full((N,), 40.0, device=D.device)  # large lambda -> 0 bits (always feasible)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        _, avg = solve(mid)
        over = avg > target
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
        _, avg_hi = solve(hi)
        if bool(((target - avg_hi).abs() <= tol * target.clamp_min(1e-6)).all()):
            break
    idx, _ = solve(hi)
    # heads whose budget allows full precision everywhere
    full = target >= float(bitset[-1]) - 1e-6
    idx[full] = nb - 1
    return idx


def window_queries(module: nn.Module, hidden_states: torch.Tensor, window: int, kwargs: dict) -> torch.Tensor:
    """Post-RoPE queries of the last `window` positions: [B, Hq, W, d]."""
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    q = module.q_proj(hidden_states[:, -window:])
    q = q.view(bsz, window, num_heads, head_dim).transpose(1, 2)
    cos, sin = kwargs["position_embeddings"]
    cos, sin = cos[:, -window:].unsqueeze(1), sin[:, -window:].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin)


# --------------------------------------------------------------------------------------------
# press
# --------------------------------------------------------------------------------------------
@dataclass
class RDKVPress(BasePress):
    compression_ratio: float = 0.0
    window_size: int = 32  # S_w in the paper
    kernel_size: int = 5  # AvgPool1d kernel
    k_share: float = 0.5  # r_K, fraction of the head budget given to K
    bitset: tuple = BITSET
    # filled during the forward pass (one prefill = one record)
    stats: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        assert 0.0 <= self.compression_ratio < 1.0, "compression_ratio must be in [0, 1)"
        self._reset_stats()

    # ---- bookkeeping -------------------------------------------------------------------
    def _reset_stats(self):
        self.stats = {"payload_bits": 0.0, "meta_bits": 0.0, "full_bits": 0.0, "kept_tokens": 0.0,
                      "total_tokens": 0.0, "layers": 0, "v_bit_hist": [0] * len(self.bitset),
                      "k_bit_hist": [0] * len(self.bitset)}

    def kv_cache_gb(self, include_meta: bool = True) -> float:
        bits = self.stats["payload_bits"] + (self.stats["meta_bits"] if include_meta else 0.0)
        return bits / 8 / 1e9

    def full_cache_gb(self) -> float:
        return self.stats["full_bits"] / 8 / 1e9

    def summary(self) -> dict:
        s = self.stats
        full = max(s["full_bits"], 1.0)
        return {
            "kv_cache_gb": self.kv_cache_gb(True),
            "kv_cache_gb_payload": self.kv_cache_gb(False),
            "kv_cache_gb_full": self.full_cache_gb(),
            "kept_fraction_bits": (s["payload_bits"] + s["meta_bits"]) / full,
            "kept_fraction_tokens": s["kept_tokens"] / max(s["total_tokens"], 1.0),
            "v_bit_hist": dict(zip(self.bitset, s["v_bit_hist"])),
            "k_bit_hist": dict(zip(self.bitset, s["k_bit_hist"])),
        }

    # ---- main --------------------------------------------------------------------------
    @torch.no_grad()
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if getattr(module, "layer_idx", 0) == 0:
            self._reset_stats()
        if self.compression_ratio == 0:
            return keys, values
        assert module.config._attn_implementation != "eager", "RDKV masking needs sdpa/flash attention"

        B, H, T, d = keys.shape
        W = self.window_size
        if T <= 2 * W:  # too short to compress meaningfully
            return keys, values

        dev = keys.device
        f = 1.0 - self.compression_ratio
        bitset = torch.tensor(self.bitset, device=dev, dtype=torch.float32)
        nb = len(self.bitset)
        n_rep = module.config.num_attention_heads // module.config.num_key_value_heads

        # Memory-lean: only K is held in fp32 for the whole call; quantised candidates are built one bit-width
        # at a time and freed straight away (long prompts on 15 GB T4s were running out of memory).
        K = keys.float()

        # ---------------- Stage 1: scores ----------------
        q = window_queries(module, hidden_states, W, kwargs).float()  # [B, Hq, W, d]
        qg = q.reshape(B, H, n_rep * W, d)  # query heads grouped under their KV head (no repeat_kv copy)
        del q
        attn = torch.matmul(qg, K.transpose(2, 3)).div_(math.sqrt(d)).view(B, H, n_rep, W, T)
        causal = torch.full((W, T), float("-inf"), device=dev).triu_(T - W + 1)
        attn = torch.softmax(attn + causal, dim=-1)
        Tc = T - W  # candidate tokens (window always kept)
        w_tok = attn[..., :Tc].sum(dim=(2, 3))  # [B, H, Tc]
        del attn, causal
        w_tok = F.avg_pool1d(w_tok, self.kernel_size, 1, self.kernel_size // 2, count_include_pad=False)

        # ---------------- Stage 2: V tokens ----------------
        Vc = values[:, :, :Tc].float()  # [B,H,Tc,d]
        eps_v = []
        for b in self.bitset:
            if b == 0:
                eps_v.append(Vc.pow(2).mean(-1).sqrt())
            elif b >= 16:
                eps_v.append(torch.zeros_like(w_tok))
            else:
                vq = fake_quant(Vc, b, dim=-1)
                eps_v.append((Vc - vq).pow_(2).mean(-1).sqrt_())
                del vq
        D_v = (w_tok[..., None] * torch.stack(eps_v, dim=-1)).reshape(B * H, Tc, nb)
        del eps_v
        # V budget per head = (1 - k_share) * 2 * f * T * d * 16 bits; the window costs W * d * 16
        v_bits_head = (1 - self.k_share) * 2 * f * T * 16  # per coordinate-row => /d already
        target_v = torch.full((B * H,), (v_bits_head - W * 16) / Tc, device=dev)
        idx_v = allocate_bits(D_v, bitset, target_v).view(B, H, Tc)
        del D_v
        # per-token quantisation is row-wise, so rows can be replaced one bit-width at a time
        for i, b in enumerate(self.bitset):
            if 0 < b < 16:
                sel = (idx_v == i)[..., None]
                if bool(sel.any()):
                    Vc = torch.where(sel, fake_quant(Vc, b, dim=-1), Vc)
        values_out = values.clone()
        values_out[:, :, :Tc] = Vc.to(values.dtype)
        del Vc
        kept_c = idx_v > 0  # [B,H,Tc]
        kept = torch.cat([kept_c, torch.ones(B, H, W, dtype=torch.bool, device=dev)], dim=-1)  # [B,H,T]
        n_kept = kept.sum(-1).float()  # [B,H]

        # ---------------- Stage 3: K channels (over kept tokens) ----------------
        q_norm = qg.pow(2).sum(dim=2).sqrt()  # [B,H,d]
        del qg
        km = kept[..., None]  # [B,H,T,1]
        k_norm = (K * km).pow_(2).sum(2).sqrt_()  # [B,H,d]
        w_ch = q_norm * k_norm / math.sqrt(d)
        denom = n_kept[..., None].clamp_min(1)
        eps_k = []
        for b in self.bitset:
            if b == 0:
                eps_k.append(k_norm / denom.sqrt())
            elif b >= 16:
                eps_k.append(torch.zeros_like(w_ch))
            else:
                kq = fake_quant(K, b, dim=2, mask=km)
                eps_k.append(((K - kq) * km).pow_(2).sum(2).div_(denom).sqrt_())
                del kq
        D_k = (w_ch[..., None] * torch.stack(eps_k, dim=-1)).reshape(B * H, d, nb)
        del eps_k
        k_bits_head = self.k_share * 2 * f * T * d * 16
        target_k = (k_bits_head / (n_kept * d)).reshape(-1)
        idx_k = allocate_bits(D_k, bitset, target_k).view(B, H, d)
        # per-channel quantisation is column-wise, so columns can be replaced one bit-width at a time
        for i, b in enumerate(self.bitset):
            sel = (idx_k == i)[:, :, None, :]  # [B,H,1,d]
            if 0 < b < 16 and bool(sel.any()):
                K = torch.where(sel, fake_quant(K, b, dim=2, mask=km), K)
        K = K.masked_fill_((idx_k == 0)[:, :, None, :], 0.0)  # removed channels
        keys_out = K.to(keys.dtype)
        del K

        # evicted tokens -> kvpress attention mask (same mechanism as AdaKVPress)
        bi, hi, si = torch.nonzero(~kept, as_tuple=True)
        module.masked_key_indices = (bi, hi, si) if bi.numel() > 0 else None

        # ---------------- stats (computed KV size) ----------------
        bv = bitset[idx_v]  # [B,H,Tc]
        bk = bitset[idx_k]  # [B,H,d]
        payload = (bv.sum() * d + B * H * W * d * 16) + (bk * n_kept[..., None]).sum()
        n_q_tok = ((bv > 0) & (bv < 16)).sum()
        n_q_ch = ((bk > 0) & (bk < 16)).sum()
        meta = 32 * n_q_tok + 32 * n_q_ch + B * H * (T + d)  # fp16 scale+zero, keep bitmaps
        s = self.stats
        s["payload_bits"] += float(payload)
        s["meta_bits"] += float(meta)
        s["full_bits"] += float(B * H * T * d * 16 * 2)
        s["kept_tokens"] += float(n_kept.sum())
        s["total_tokens"] += float(B * H * T)
        s["layers"] += 1
        for i in range(nb):
            s["v_bit_hist"][i] += int((idx_v == i).sum())
            s["k_bit_hist"][i] += int((idx_k == i).sum())
        return keys_out, values_out
