"""Run:  python -m pytest -q members/manjot/test_rdkv.py      (CPU, ~30 s, no model download)"""
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdkv_press import RDKVPress, allocate_bits, fake_quant  # noqa: E402

BITS = torch.tensor([0, 2, 4, 8, 16.0])


def test_fake_quant_error_shrinks_with_bits():
    x = torch.randn(4, 64, 128)
    errs = [(x - fake_quant(x, b, dim=-1)).pow(2).mean().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2] > 0
    assert torch.equal(fake_quant(x, 16, dim=-1), x)


@pytest.mark.parametrize("f", [0.25, 0.2, 0.1, 0.05])
def test_allocator_hits_budget(f):
    torch.manual_seed(0)
    N, U = 4, 3000
    w = torch.rand(N, U).pow(4) + 1e-4
    sig = torch.rand(N, U) + 0.5
    eps = torch.stack([sig, sig / 4, sig / 16, sig / 256, sig * 0], -1)
    target = torch.full((N,), 16 * f)
    idx = allocate_bits(w[..., None] * eps, BITS, target)
    avg = BITS[idx].mean(-1)
    assert (avg <= target + 1e-6).all()
    assert ((target - avg) / target < 0.02).all()
    # mixed allocation: some evicted, some quantised
    assert (idx == 0).any() and ((idx > 0) & (idx < 4)).any()


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=4096,
                      attn_implementation="sdpa")
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.mark.parametrize("ratio", [0.75, 0.8, 0.9, 0.95])
def test_end_to_end_tiny_llama(tiny_llama, ratio):
    ids = torch.randint(0, 256, (1, 1024))
    press = RDKVPress(compression_ratio=ratio)
    with torch.no_grad(), press(tiny_llama):
        out = tiny_llama(ids, use_cache=True)
    s = press.summary()
    assert s["kv_cache_gb_payload"] > 0
    payload_frac = s["kv_cache_gb_payload"] / s["kv_cache_gb_full"]
    assert payload_frac <= (1 - ratio) + 1e-3, payload_frac
    assert payload_frac >= (1 - ratio) * 0.9, payload_frac  # budget actually used
    assert torch.isfinite(out.logits).all()
    # a decode step after compression must run (exercises kvpress masked-key patch)
    with torch.no_grad():
        nxt = tiny_llama(ids[:, -1:], past_key_values=out.past_key_values, use_cache=True)
    assert torch.isfinite(nxt.logits).all()
    print(ratio, s)
