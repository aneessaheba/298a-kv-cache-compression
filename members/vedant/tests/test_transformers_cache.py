import torch
import pytest

from turboquant.kv_cache import LayerQuantConfig, TransformersTurboQuantCache


def test_tiny_llama_accepts_turboquant_cache() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    cache = TransformersTurboQuantCache(config, 3, qjl_keys=True, seed=5)
    first = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        past_key_values=cache,
        use_cache=True,
    )
    assert first.past_key_values is cache
    second = model(
        input_ids=torch.tensor([[5]]),
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.tensor([4]),
    )
    assert second.logits.shape == (1, 1, 64)
    assert cache.get_seq_length() == 5
    assert cache.storage_stats().compressed_bytes < cache.storage_stats().full_precision_bytes


def test_tiny_llama_accepts_mixed_layer_configuration() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    cache = TransformersTurboQuantCache(
        config,
        layer_configs=[LayerQuantConfig(2, True), LayerQuantConfig(4, False)],
        seed=11,
    )
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            past_key_values=cache,
            use_cache=True,
        )
    assert output.logits.shape == (1, 3, 64)
    assert [layer.bits for layer in cache.layers] == [2, 4]
    assert [layer.qjl_keys for layer in cache.layers] == [True, False]
    assert cache.storage_stats().compressed_bytes > 0


def test_mixed_layer_configuration_rejects_ambiguous_qjl_value() -> None:
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    with pytest.raises(ValueError, match="boolean qjl_keys"):
        TransformersTurboQuantCache(
            config,
            layer_configs=[{"bits": 3, "qjl_keys": "false"}],
        )
