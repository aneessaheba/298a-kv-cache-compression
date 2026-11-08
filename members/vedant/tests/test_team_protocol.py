import json
from pathlib import Path

import pytest

from turboquant.protocol import prepare_output, validate_dataset


def test_team_scope_rejects_wrong_dataset_and_duplicate_ids(tmp_path):
    row = {"record_id": "a", "task": "qasper", "prompt": "text", "answers": ["answer"]}
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_dataset(path)
    assert len(validate_dataset(path, scope="subset")) == 1
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(path, scope="subset")


def test_resume_rejects_changed_settings_or_orphaned_rows(tmp_path):
    path = tmp_path / "run.jsonl"
    spec = {"bits": 3, "input_sha256": "a"}
    run_id = prepare_output(path, spec, {"a"})
    path.write_text(json.dumps({"record_id": "a", "run_fingerprint": run_id}) + "\n")
    assert prepare_output(path, spec, {"a"}) == run_id
    with pytest.raises(ValueError, match="settings differ"):
        prepare_output(path, {**spec, "bits": 4}, {"a"})
    with pytest.raises(ValueError, match="unexpected"):
        prepare_output(path, spec, {"b"})
    orphan = tmp_path / "orphan.jsonl"
    orphan.write_text(path.read_text())
    with pytest.raises(ValueError, match="lacks a run manifest"):
        prepare_output(orphan, spec, {"a"})
    path.write_text(path.read_text() + "{truncated")
    with pytest.raises(ValueError):
        prepare_output(path, spec, {"a"})


def test_prompt_field_and_chat_exceptions_are_preserved():
    from turboquant.longbench import NO_CHAT_TEMPLATE_TASKS, _format_prompt
    import torch

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return type("Encoding", (), {"input_ids": torch.tensor([[1, 2]])})()

        def apply_chat_template(self, messages, **kwargs):
            return "CHAT:" + messages[0]["content"]

    tokenizer = Tokenizer()
    for task in NO_CHAT_TEMPLATE_TASKS:
        assert _format_prompt(tokenizer, {"task": task, "prompt": "frozen prompt"}, 32000) == ("frozen prompt", True)
    assert _format_prompt(tokenizer, {"task": "qasper", "prompt": "frozen prompt"}, 32000) == ("CHAT:frozen prompt", False)


def test_packed_quantizer_state_is_reported_separately():
    import torch
    from transformers import LlamaConfig
    from turboquant.kv_cache import TransformersTurboQuantCache

    config = LlamaConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16)
    cache = TransformersTurboQuantCache(config, 3)
    assert cache.quantizer_state_bytes() == 0
    x = torch.randn(1, 1, 4, 16)
    for layer in cache.layers:
        layer.update(x, x)
    assert cache.quantizer_state_bytes() > 0
    assert len(cache.layers) == 2
    assert cache.storage_stats().compressed_bytes == sum(layer.storage_stats().compressed_bytes for layer in cache.layers)
