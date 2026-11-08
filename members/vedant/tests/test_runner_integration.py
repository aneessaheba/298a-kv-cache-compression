"""Offline, actual tiny-Llama generation -> resume -> official scoring -> comparison."""

import json
import sys

import pytest


def test_offline_runner_score_resume_and_compare(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from turboquant.longbench import main as run
    from turboquant.scoring import score_file
    from turboquant.compare import compare_runs

    model_dir = tmp_path / "tiny_model"
    tokenizer_impl = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                         "context": 3, "question": 4, "answer": 5}, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl,
                                        pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(model_dir)
    config = LlamaConfig(vocab_size=6, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, max_position_embeddings=64, pad_token_id=0, eos_token_id=2)
    LlamaForCausalLM(config).save_pretrained(model_dir)
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"record_id": rid, "task": "qasper", "prompt": "context question",
                                         "answers": ["answer"], "all_classes": [], "metric": "f1"}) + "\n"
                            for rid in ("a", "b")))
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"qasper": 2}))
    base_args = ["runner", "--input", str(data), "--dataset-scope", "subset", "--model", str(model_dir),
                 "--device", "cpu", "--dtype", "float32", "--max-new-tokens-config", str(limits)]
    full = tmp_path / "full.jsonl"
    tq = tmp_path / "tq.jsonl"
    monkeypatch.setattr(sys, "argv", base_args + ["--method", "full", "--output", str(full), "--max-samples", "1"])
    run()
    assert len(full.read_text().splitlines()) == 1
    monkeypatch.setattr(sys, "argv", base_args + ["--method", "full", "--output", str(full)])
    run()
    assert len(full.read_text().splitlines()) == 2
    monkeypatch.setattr(sys, "argv", base_args + ["--method", "turboquant", "--bits", "3", "--output", str(tq)])
    run()
    first = json.loads(tq.read_text().splitlines()[0])
    assert "all_classes" in first and first["quantizer_state_bytes"] > 0
    assert first["persistent_cache_plus_quantizer_bytes"] > first["measured_kv_bytes"]
    assert first["run_fingerprint"] and first["input_ids_sha256"]
    assert json.loads(full.read_text().splitlines()[0])["bits"] == 32
    paths = {"full": tmp_path / "full_scored.jsonl", "tq": tmp_path / "tq_scored.jsonl"}
    score_file(full, paths["full"])
    score_file(tq, paths["tq"])
    assert compare_runs(paths, bootstrap_samples=10)["paired_record_count"] == 2
    monkeypatch.setattr(sys, "argv", base_args + ["--method", "turboquant", "--bits", "4", "--output", str(tq)])
    with pytest.raises(SystemExit):
        run()
