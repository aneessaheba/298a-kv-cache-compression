"""TriAttention as a kvpress ScorerPress for the shared LongBench/RULER harness (SAM / Samruddhi).

TriAttention (Mao et al., ICML 2026, arXiv:2604.04921; repo WeianMao/triattention)
compresses the KV cache by scoring each cached key with a *pre-RoPE* importance
signal instead of the post-RoPE attention of a few recent queries. The paper's
observation is that, in long-reasoning models, pre-RoPE query and key vectors
concentrate around fixed centers, so a key's future importance can be estimated
from its pre-RoPE geometry (its alignment with the query center and its norm)
rather than from position-rotated "representative" queries, which rotate away
from each other and give unstable top-key selection.

The official repository implements this inside vLLM / SGLang / TensorRT-LLM with
per-model *precomputed* Q/K frequency statistics (centers and norms). That infra
is not available in this harness (we run the Hugging Face `kv-press` pipeline on
a single prompt at a time), so this is a faithful *reimplementation of the scoring
rule* that stays inside the kvpress `ScorerPress` contract:

    score(key_i) = cos(angle(k_i, q_center)) * ||k_i||     (pre-RoPE)

where `q_center` is the mean pre-RoPE query direction over a recent observation
window (an online proxy for the paper's calibrated query center, computed per
head). Recent `window_size` tokens are always kept, as in SnapKV / the paper's
runtime WINDOW_SIZE. The top (1 - compression_ratio) fraction of tokens by score
is retained; kvpress handles the actual pruning in ScorerPress.compress().

Register it in harness/run_longbench.py `make_press()` as "triattention" and both
runners (LongBench, RULER) can score it against the shared baselines.

Documented limitations of this reimplementation live in members/samruddhi/FLAWS.md.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class TriAttentionPress(ScorerPress):
    """Pre-RoPE, center-aligned KV scoring (TriAttention), as a kvpress ScorerPress.

    Parameters
    ----------
    compression_ratio : float, default 0.0
        Fraction of key-value pairs to evict. 0.9 keeps 10% of the cache.
    window_size : int, default 64
        Recent tokens always kept (never scored for eviction). Matches the
        paper's runtime WINDOW_SIZE and SnapKV's observation window.
    norm_power : float, default 1.0
        Weight of the key-norm term. 0.0 scores by center-alignment only
        (pure direction), 1.0 multiplies by the pre-RoPE key norm. Lets us
        probe the paper's claim that both direction and magnitude matter.
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    norm_power: float = 1.0
    n_sink: int = 4  # initial attention-sink tokens always kept (StreamingLLM finding)

    def _prerope_key_states(self, module, hidden_states):
        """Pre-RoPE keys, shape (bsz, num_kv_heads, k_len, head_dim).

        kvpress hands `score()` the post-RoPE cached keys; TriAttention scores in
        the *un-rotated* key space, so we recompute the pre-RoPE keys from k_proj
        on the same hidden states. At prefill q_len == k_len, so positions align.
        """
        bsz, q_len, _ = hidden_states.shape
        head_dim = module.head_dim
        num_kv_heads = module.config.num_key_value_heads
        key_states = module.k_proj(hidden_states)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        # Qwen3/Gemma3 apply a learned k_norm before RoPE.
        if hasattr(module, "k_norm") and module.k_norm is not None:
            try:
                key_states = module.k_norm(key_states)
            except Exception:
                pass
        return key_states

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_kv_heads, k_len, head_dim = keys.shape

        # --- Pre-RoPE query center, estimated from the recent observation window ---
        # The paper calibrates per-model Q centers offline; with no precomputed
        # stats we use the mean direction of the recent-window pre-RoPE queries as
        # an online proxy. Queries have num_attention_heads; average over the
        # heads that share each KV head (GQA) to get one center per KV head.
        num_heads = module.config.num_attention_heads
        num_kv_groups = num_heads // num_kv_heads
        window = min(self.window_size, max(1, hidden_states.shape[1] // 2))
        q_prerope = get_prerope_query_states(module, hidden_states[:, -window:])  # (bsz, num_heads, w, hd)
        q_center = q_prerope.mean(dim=2)  # (bsz, num_heads, hd)
        # Group query heads down to KV heads (mean over each GQA group).
        q_center = q_center.view(bsz, num_kv_heads, num_kv_groups, head_dim).mean(dim=2)  # (bsz, kv, hd)
        q_center = F.normalize(q_center.float(), dim=-1)  # unit direction per KV head

        # --- Pre-RoPE keys (recomputed in the same un-rotated space) ---
        k_prerope = self._prerope_key_states(module, hidden_states).float()  # (bsz, kv, k_len, hd)

        # --- TriAttention score: alignment with the query center, scaled by key norm ---
        # Computed in float32: a bf16 norm/score ties many distinct positions and
        # hands the eviction choice to arbitrary top-k tie-breaking (see KnormPress).
        k_norm = k_prerope.norm(dim=-1)  # (bsz, kv, k_len)
        k_dir = k_prerope / (k_norm.unsqueeze(-1) + 1e-6)
        alignment = torch.einsum("bkhd,bkd->bkh", k_dir, q_center)  # cos angle, (bsz, kv, k_len)
        # Shift cos from [-1, 1] to [0, 2] so the norm term can't be cancelled by sign.
        scores = (alignment + 1.0) * (k_norm.clamp_min(1e-6) ** self.norm_power)

        # --- Protect attention sinks and the recent window ---
        # Early tokens act as attention sinks (StreamingLLM); evicting them makes the
        # model emit degenerate output. Recent tokens carry the query/answer cue.
        keep = scores.max() + 1.0
        if self.n_sink > 0 and self.n_sink < k_len:
            scores[..., : self.n_sink] = keep
        if 0 < window <= k_len:
            scores[..., -window:] = keep

        return scores.to(keys.dtype)
