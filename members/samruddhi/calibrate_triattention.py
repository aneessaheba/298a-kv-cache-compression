"""Calibrate TriAttention frequency statistics for a model (faithful to the authors' scripts/calibrate.py).

TriAttention scores keys using per-head Q/K *frequency-domain* statistics computed
once per model. This reproduces WeianMao/triattention scripts/calibrate.py:
run one forward pass on plain text, hook every attention layer to capture the
post-RoPE query states, invert RoPE to recover the base (pre-RoPE) queries, treat
each head_dim vector as head_dim/2 complex frequency pairs, and store per head:

  * q_mean_complex  — mean of the complex queries over positions (the query "center")
  * q_abs_mean      — mean magnitude per frequency

The output .pt matches the authors' schema (metadata + per-head stats) so the same
scoring math can consume it. This replaces the online center-estimate proxy used
earlier, so measured results reflect TriAttention as designed.

    C:\\Python310\\python.exe members/samruddhi/calibrate_triattention.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output members/samruddhi/calibration/llama31_8b_stats.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated, cos, sin, scale):
    """Recover base (pre-RoPE) vectors. Mirrors pruning_utils.invert_rope (style='half')."""
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    return base * cos_unit - rotate_half(base) * sin_unit


def to_complex_pairs(tensor):
    """head_dim -> head_dim/2 complex pairs (front half = real, back half = imag)."""
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    t = tensor.to(dtype=real_dtype)
    fc = t.shape[-1] // 2
    return torch.complex(t[..., :fc].contiguous(), t[..., fc:].contiguous())


DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 4
    + "In information theory, entropy quantifies the uncertainty of a random variable. "
    + "Transformers process sequences in parallel using self-attention, where every token "
    + "attends to every earlier token through learned query, key and value projections. "
    + "The rotary position embedding rotates query and key vectors by an angle proportional "
    + "to their position, so that the dot product depends only on relative distance. "
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--input", default=None,
                   help="plain-text calibration file (defaults to a built-in prose sample repeated to --max-length)")
    p.add_argument("--output", default=str(HERE / "calibration" / "llama31_8b_stats.pt"))
    p.add_argument("--max-length", type=int, default=4096,
                   help="calibration token length (authors use up to 32768; 4096 gives stable per-head means)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dtype = torch.bfloat16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} on {device}", file=sys.stderr)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 device_map=device, trust_remote_code=True)
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    backbone = getattr(model, "model", model)
    attn_layers = [layer.self_attn for layer in backbone.layers]
    rotary = backbone.rotary_emb if hasattr(backbone, "rotary_emb") else attn_layers[0].rotary_emb
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # --- calibration text ---
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = DEFAULT_TEXT
        while len(tok(text, add_special_tokens=False).input_ids) < args.max_length:
            text += DEFAULT_TEXT
    input_ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
    seq_len = input_ids.shape[1]
    print(f"calibration tokens: {seq_len}", file=sys.stderr)

    # shared cos/sin for inversion, computed once
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    probe = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(probe, position_ids)  # [1, seq_len, head_dim]
    cos_i = cos_table[0].unsqueeze(0)  # [1, seq_len, head_dim]
    sin_i = sin_table[0].unsqueeze(0)

    # Compute per-head complex stats INSIDE each hook and keep only the small result
    # (head_dim/2 complex numbers per head), so GPU memory stays flat across layers.
    stats, sampled = {}, []

    def make_hook(idx):
        def hook(module, a, kw):
            hs = a[0] if a else kw.get("hidden_states")
            if hs is None:
                return
            bsz, q_len, _ = hs.shape
            q = module.q_proj(hs).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            pos = torch.arange(q_len, device=hs.device).unsqueeze(0)
            pr = torch.zeros(1, q_len, head_dim, device=hs.device, dtype=hs.dtype)
            cos, sin = rotary(pr, pos)
            q_rot = ((q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))) * attn_scale
            # invert RoPE back to base queries, then reduce to per-head means immediately
            cos_h = cos_i[:, :q_len, :].unsqueeze(1)  # [1,1,q_len,head_dim]
            sin_h = sin_i[:, :q_len, :].unsqueeze(1)
            q_base = invert_rope(q_rot, cos_h, sin_h, attn_scale)
            for head_idx in range(num_heads):
                qc = to_complex_pairs(q_base[0, head_idx])  # [q_len, fc]
                qmean = qc.mean(dim=0)
                qabs = qc.abs().mean(dim=0)
                stats[f"layer{idx:02d}_head{head_idx:02d}"] = {
                    "q_mean_real": qmean.real.cpu(),
                    "q_mean_imag": qmean.imag.cpu(),
                    "q_abs_mean": qabs.cpu(),
                }
                sampled.append((idx, head_idx))
            del q, q_rot, q_base
        return hook

    handles = [attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
               for i, attn in enumerate(attn_layers)]
    print("forward pass...", file=sys.stderr)
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()

    payload = {
        "metadata": {
            "num_traces": 1, "head_dim": head_dim, "dtype": "bfloat16",
            "rope_style": "half",
            "sampled_heads": [[int(l), int(h)] for l, h in sampled],
        },
        "stats": stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out} ({len(sampled)} heads)", file=sys.stderr)


if __name__ == "__main__":
    main()
