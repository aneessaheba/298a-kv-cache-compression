"""Chunked prefill so long prompts fit in GPU memory (enables the full 800-record run).

The kvpress pipeline prefills the whole context in one forward pass:

    model.model(input_ids=context_ids, past_key_values=cache)

which materialises the full O(seq_len^2) attention and OOMs on long LongBench prompts
(the 32 GB card cannot hold a 32K-token prefill). Compression happens inside the press
hook during that pass, so it cannot lower the *peak* prefill memory.

This module prefills the context in fixed-size chunks instead. After each chunk it runs
the press's compress() on every layer's cache, so the stored cache stays near the budget
and the peak attention memory is bounded by chunk_size x current_cache_size rather than
full_len^2. The result cache is the compressed context, exactly what the pipeline's
decode step expects.

Used by harness/run_longbench.py when --chunk-prefill N is set.
"""

import torch
from transformers import DynamicCache

from kvpress.utils import extract_keys_and_values


@torch.no_grad()
def chunked_prefill(model, context_ids, press, chunk_size, budget, cache=None):
    """Prefill `context_ids` in chunks of `chunk_size`, keeping the cache under `budget`.

    After each chunk, if the cache exceeds `budget` tokens, each layer is compressed
    back down to `budget` (using a one-shot ratio derived from the current length), so
    eviction does not compound across chunks and the final kept size ~= budget. This
    reproduces the single-shot target ratio while bounding peak prefill memory.

    If `cache` is provided (e.g. a DynamicCache(offloading=True) for the uncompressed
    CPU-offloaded baseline) it is used as-is and no compression is applied when press is
    None. Returns the cache holding the (optionally compressed) context.
    """
    import copy

    device = model.device
    context_ids = context_ids.to(device)
    total = context_ids.shape[1]
    if cache is None:
        cache = DynamicCache()

    if press is not None:
        press.post_init_from_model(model)
    # Work on a clone so we can set an exact per-step ratio without mutating the caller's press.
    step_press = copy.copy(press) if press is not None else None

    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layers = language_model.layers

    # Make the press's score()/compress() able to find rotary_emb on each attention module
    # (the pipeline's press context normally sets this; we replicate it here).
    for layer in layers:
        layer.self_attn.rotary_emb = language_model.rotary_emb

    # Capture each layer's hidden_states input during the chunk forward, so we can feed
    # the correct hidden_states to compress() (the press scores from them).
    captured = {}

    def make_capture(idx):
        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            captured[idx] = hs
        return hook

    handles = [layer.self_attn.register_forward_pre_hook(make_capture(i), with_kwargs=True)
               for i, layer in enumerate(layers)]

    # High-water mark: let the cache grow a little past budget, then compress back to budget.
    # Capped so scoring never runs over a huge cache (keeps long docs fast). Scoring a cache of
    # ~`hwm` tokens is the per-compress cost, independent of total document length.
    hwm = (budget + max(chunk_size, 2048)) if budget else None

    try:
        processed = 0
        while processed < total:
            chunk = context_ids[:, processed: processed + chunk_size]
            cache_position = torch.arange(processed, processed + chunk.shape[1], device=device)
            model.model(
                input_ids=chunk,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            processed += chunk.shape[1]

            # Compress back to `budget` only once the cache crosses the high-water mark.
            cur_len = cache.get_seq_length()
            if step_press is not None and budget and cur_len >= hwm:
                step_press.compression_ratio = 1.0 - (budget / cur_len)
                for i, layer in enumerate(layers):
                    module = layer.self_attn
                    keys, values = extract_keys_and_values(cache, i)
                    hs = captured.get(i)
                    new_keys, new_values = step_press.compress(module, hs, keys, values, None, {})
                    cache.layers[i].keys = new_keys
                    cache.layers[i].values = new_values

        # Final compression to exactly `budget` at the end (in case the tail chunk left us above it).
        cur_len = cache.get_seq_length()
        if step_press is not None and budget and cur_len > budget:
            step_press.compression_ratio = 1.0 - (budget / cur_len)
            for i, layer in enumerate(layers):
                module = layer.self_attn
                keys, values = extract_keys_and_values(cache, i)
                hs = captured.get(i)
                new_keys, new_values = step_press.compress(module, hs, keys, values, None, {})
                cache.layers[i].keys = new_keys
                cache.layers[i].values = new_values
    finally:
        for h in handles:
            h.remove()

    return cache
