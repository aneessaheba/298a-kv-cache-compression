"""The kvpress TurboQuant press and the cache changes for the shared harness."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from turboquant.kv_cache import TurboQuantCacheLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("kvpress")
from turboquant_press import TurboQuantPress, head_token_bytes  # noqa: E402

LLAMA = SimpleNamespace(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)


def test_bytes_per_token_match_the_forecast_table() -> None:
    # STATUS.md: 17,920 B/token for QJL keys at 2 bits, 17,408 for MSE keys at 2 bits
    assert TurboQuantPress(bits=2).context_bytes_per_token(LLAMA) == 17_920
    assert TurboQuantPress(bits=2, qjl_keys=False).context_bytes_per_token(LLAMA) == 17_408
    assert TurboQuantPress(bits=4).context_bytes_per_token(LLAMA) == 34_304


@pytest.mark.parametrize("bits,qjl", [(2, True), (3, True), (4, False)])
def test_formula_matches_packed_bytes(bits: int, qjl: bool) -> None:
    keys = torch.randn(1, 2, 64, 128)
    layer = TurboQuantCacheLayer(128, bits, qjl_keys=qjl)
    layer.update(keys, keys.clone())
    per_head_token = layer.storage_stats().compressed_bytes / (2 * 64)
    assert per_head_token == head_token_bytes(128, bits, qjl)


def test_prefill_attends_at_full_precision_and_stores_quantized() -> None:
    x = torch.randn(1, 2, 8, 16)
    layer = TurboQuantCacheLayer(16, 3, seed=1)
    keys, values = layer.update(x, x)
    assert torch.equal(keys, x) and torch.equal(values, x)
    assert not torch.allclose(layer._decoded_keys, x)
    step = torch.randn(1, 2, 1, 16)
    keys, _ = layer.update(step, step)
    assert keys.shape == (1, 2, 9, 16)
    assert torch.equal(keys[:, :, :8], layer._decoded_keys[:, :, :8])
    assert torch.equal(keys[:, :, 8:], step)


def test_kept_decoded_copy_equals_full_dequantize() -> None:
    layer = TurboQuantCacheLayer(16, 3, seed=2)
    for length in (6, 1, 1, 3):
        x = torch.randn(1, 2, length, 16)
        layer.update(x, x * 0.5)
    keys, values = layer.dequantize()
    assert torch.equal(keys, layer._decoded_keys)
    assert torch.equal(values, layer._decoded_values)


def test_press_quantizes_like_the_cache_and_counts_bytes() -> None:
    press = TurboQuantPress(bits=3, seed=42)
    keys = torch.randn(1, 2, 10, 16, dtype=torch.bfloat16)
    values = torch.randn(1, 2, 10, 16, dtype=torch.bfloat16)
    module = SimpleNamespace(layer_idx=0)
    out_keys, out_values = press.compress(module, None, keys, values, None, {})
    assert out_keys.dtype == torch.bfloat16 and out_keys.shape == keys.shape
    reference = TurboQuantCacheLayer(16, 3, seed=42)
    reference.update(keys, values)
    assert torch.equal(out_keys, reference._decoded_keys)
    assert torch.equal(out_values, reference._decoded_values)
    assert press.stats["packed_bytes"] == 2 * 10 * head_token_bytes(16, 3)
    error = (out_keys.float() - keys.float()).norm() / keys.float().norm()
    assert 0 < error < 0.6


def test_chunked_document_matches_one_pass(monkeypatch) -> None:
    import turboquant_press

    keys = torch.randn(1, 2, 37, 16, dtype=torch.bfloat16)
    values = torch.randn(1, 2, 37, 16, dtype=torch.bfloat16)
    module = SimpleNamespace(layer_idx=3)
    whole = TurboQuantPress(bits=3).compress(module, None, keys, values, None, {})
    monkeypatch.setattr(turboquant_press, "CHUNK_TOKENS", 8)
    press = TurboQuantPress(bits=3)
    chunked = press.compress(SimpleNamespace(layer_idx=0), None, keys, values, None, {})
    chunked = press.compress(module, None, keys, values, None, {})
    assert torch.equal(whole[0], chunked[0]) and torch.equal(whole[1], chunked[1])
    assert press.stats["packed_bytes"] == 2 * 2 * 37 * head_token_bytes(16, 3)


def test_decode_never_reunpacks_the_prompt(monkeypatch):
    layer = TurboQuantCacheLayer(128, 3)
    calls = []
    original = layer._decode_chunk
    def counted(keys, values):
        calls.append(keys.mse.indices.shape[-2] if hasattr(keys, 'mse') else keys.indices.shape[-2])
        return original(keys, values)
    monkeypatch.setattr(layer, '_decode_chunk', counted)
    x = torch.randn(1, 1, 4096, 128)
    layer.update(x, x)
    for _ in range(3):
        step = torch.randn(1, 1, 1, 128)
        layer.update(step, step)
    assert calls == [4096, 1, 1, 1]


def test_prefill_logits_equal_uncompressed_model():
    from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
    from turboquant.kv_cache import TransformersTurboQuantCache
    torch.manual_seed(2)
    config = LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    model = LlamaForCausalLM(config).eval()
    ids = torch.randint(0,16,(1,32))
    with torch.inference_mode():
        full = model(input_ids=ids, past_key_values=DynamicCache(config=config)).logits
        quant = model(input_ids=ids, past_key_values=TransformersTurboQuantCache(config,2)).logits
    torch.testing.assert_close(full, quant, atol=0, rtol=0)


@pytest.mark.parametrize('bits', [2.5,3.5])
def test_mixed_actual_rate_and_selector_accounting(bits):
    from turboquant.mixed import quantize_document
    x = torch.randn(1,2,16,128,dtype=torch.bfloat16)
    x[...,0] *= 30
    k,v,stats = quantize_document(x,x,bits,qjl_keys=True,chunk_tokens=8)
    assert stats['nominal_bits'] == bits
    assert stats['outlier_channels'] == 64
    assert stats['selector_bytes'] == 2 * 1 * 2 * 128 * 2
    assert stats['packed_bytes'] == 2 * 16 * head_token_bytes(128,bits) + stats['selector_bytes']
    assert k.shape == x.shape and v.dtype == torch.bfloat16
    whole = quantize_document(x,x,bits,qjl_keys=True,chunk_tokens=32)
    assert torch.equal(k,whole[0]) and torch.equal(v,whole[1])


@pytest.mark.parametrize('bits,qjl', [(bits,qjl) for bits in (2,2.5,3,3.5,4) for qjl in (True,False)] + [(None,True)])
def test_shared_pipeline_preserves_question_and_uses_full_prefill(bits,qjl):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from kvpress import KVPressTextGenerationPipeline
    tok=Tokenizer(WordLevel({'[UNK]':0,'[BOS]':1,'[EOS]':2,'context':3,'question':4},unk_token='[UNK]'))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    config=LlamaConfig(vocab_size=5,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                       num_attention_heads=2,num_key_value_heads=1,head_dim=16,eos_token_id=2)
    model=LlamaForCausalLM(config).to(torch.bfloat16).eval()
    pipe=KVPressTextGenerationPipeline(model=model,tokenizer=tokenizer)
    press=(TurboQuantPress(bits=bits,qjl_keys=qjl) if bits is not None else
           TurboQuantPress(layer_configs=[{'layer':i,'bits':2+i,'qjl_keys':True} for i in range(2)]))
    observed=[]
    original=press.compress
    def capture(module, hidden, keys, values, attentions, kwargs):
        observed.append((module.layer_idx,keys.shape[-2]))
        return original(module,hidden,keys,values,attentions,kwargs)
    press.compress=capture
    with torch.inference_mode():
        result=pipe('context context',question='question',press=press,max_new_tokens=3)
    # Exactly one document compression per layer. Question/decode forwards cannot trigger it.
    assert observed == [(0,3),(1,3)]
    assert isinstance(result['answer'],str)
    assert press.stats['layers'] == 2


def test_saved_queries_are_actual_post_rope_queries():
    from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
    from turboquant.extract_kv import attach_query_hooks
    config=LlamaConfig(vocab_size=8,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,head_dim=16)
    model=LlamaForCausalLM(config).eval()
    tensors={}
    hooks=attach_query_hooks(model,tensors)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1,2,3]]),past_key_values=DynamicCache(config=config))
    for hook in hooks: hook.remove()
    assert tensors['layer_0.queries'].shape == (1,2,3,16)
    assert torch.isfinite(tensors['layer_0.queries']).all()


def test_adaptive_press_uses_calibrated_layer_settings():
    settings=[{'layer':0,'bits':2,'qjl_keys':True},{'layer':1,'bits':4,'qjl_keys':False}]
    press=TurboQuantPress(layer_configs=settings)
    config=SimpleNamespace(num_hidden_layers=2,num_key_value_heads=1,head_dim=16)
    x=torch.randn(1,1,8,16,dtype=torch.bfloat16)
    for i in range(2):
        press.compress(SimpleNamespace(layer_idx=i,config=config),None,x,x,None,{})
    assert press.stats['layer_settings'] == settings
    assert press.stats['packed_bytes'] == 8*press.context_bytes_per_token(config)
    with pytest.raises(ValueError,match='every model layer'):
        press.context_bytes_per_token(LLAMA)


def test_shared_factory_fractional_and_adaptive_settings(tmp_path):
    import importlib.util
    import json
    repo=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(repo/'harness'))
    spec=importlib.util.spec_from_file_location('shared_longbench_runner',repo/'harness/run_longbench.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.make_press('turboquant_2.5bit',0).bits == 2.5
    assert runner.make_press('turboquant_mse_3.5bit',0).qjl_keys is False
    # Count the protected question with the pipeline's actual chat suffix, not raw text length.
    pipe=SimpleNamespace(model=SimpleNamespace(config=LLAMA),
                         tokenizer=lambda *a,**k: SimpleNamespace(input_ids=[1]))
    pipe.tokenizer.model_max_length=32000
    pipe.preprocess=lambda *a,**k: {'questions_ids':[torch.zeros((1,7),dtype=torch.long)]}
    press=runner.make_press('turboquant_2.5bit',0)
    press.stats={'packed_bytes':1234567}
    assert runner.kv_cache_gb(pipe,'document','question',0,press) == round((1234567+7*131072)/1e9,6)
    allocation=tmp_path/'allocation.json'
    allocation.write_text(json.dumps({'layers':[{'layer':0,'bits':3,'qjl_keys':True}]}))
    assert runner.make_press('turboquant_adaptive',0,allocation).layer_configs[0]['bits'] == 3
    with pytest.raises(ValueError,match='requires'):
        runner.make_press('turboquant_adaptive',0)


def test_calibration_uses_the_same_packed_reconstruction_as_press():
    from turboquant.real_kv import evaluate_real_kv
    x=torch.randn(1,1,16,16,dtype=torch.bfloat16)
    report=evaluate_real_kv(x,x,queries=x,bit_widths=(3,),seed=42)
    for row in report['rows']:
        press=TurboQuantPress(bits=3,qjl_keys=row['method']=='turboquant_product',seed=42)
        k,v=press.compress(SimpleNamespace(layer_idx=0),None,x,x,None,{})
        expected=((x.double()-k.double()).square().sum(-1)/x.double().square().sum(-1)).mean().item()
        assert row['key_normalized_mse'] == pytest.approx(expected)
        assert row['packed_bytes'] == press.stats['packed_bytes']


def test_shared_cli_writes_results_oom_summary_and_checked_resume(tmp_path,monkeypatch):
    import importlib.util
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    from kvpress import KVPressTextGenerationPipeline
    from turboquant import protocol
    repo=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(repo/'harness'))
    spec=importlib.util.spec_from_file_location('shared_cli_audit',repo/'harness/run_longbench.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    tok=Tokenizer(WordLevel({'[UNK]':0,'[BOS]':1,'[EOS]':2,'context':3,'question':4},unk_token='[UNK]'))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    config=LlamaConfig(vocab_size=5,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                       num_attention_heads=2,num_key_value_heads=1,head_dim=16,eos_token_id=2)
    pipe=KVPressTextGenerationPipeline(model=LlamaForCausalLM(config).to(torch.bfloat16).eval(),tokenizer=tokenizer)
    def factory(name,**kwargs):
        assert name=='kv-press-text-generation' and kwargs['dtype']==torch.bfloat16
        return pipe
    monkeypatch.setattr(runner,'pipeline',factory)
    monkeypatch.setattr(runner,'pick_device',lambda:'cpu')
    calls=[]
    original_forward=pipe._forward
    def fail_once(*args,**kwargs):
        calls.append(1)
        if len(calls)==1: raise torch.cuda.OutOfMemoryError('simulated')
        return original_forward(*args,**kwargs)
    monkeypatch.setattr(pipe,'_forward',fail_once)
    records=[{'record_id':rid,'task':'triviaqa','prompt':'context\nquestion\nanswer:',
              'context':'context','answers':['question'],'all_classes':[],
              'prompt_tokens_after':6} for rid in ('a','b')]
    data=tmp_path/'synthetic.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in records))
    # This isolated CLI test uses synthetic IDs; the real 800-file hash is audited separately.
    validations=[]
    def validate(path):
        assert path==data
        validations.append(path)
        return records
    monkeypatch.setattr(protocol,'validate_dataset',validate)
    args=['runner','--model','offline-tiny','--data',str(data),'--press','turboquant_2.5bit',
          '--output-dir',str(tmp_path/'out')]
    monkeypatch.setattr(sys,'argv',args)
    runner.main()
    output=tmp_path/'out/offline-tiny_turboquant_2.5bit_0.0.jsonl'
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==2 and rows[0]['oom'] and rows[0]['score']==0
    assert rows[0]['kv_cache_gb'] is None
    assert rows[1]['dtype']=='bfloat16' and rows[1]['score_scale']=='0-1'
    assert rows[1]['turboquant_stats']['layers']==2 and rows[1]['kv_cache_gb']>0
    summary=json.loads(output.with_suffix('.summary.json').read_text())
    assert summary['n_records']==2 and summary['device']=='cpu'
    runner.main()
    assert len(calls)==2 and len(validations)==2
    assert len(output.read_text().splitlines())==2
    monkeypatch.setattr(sys,'argv',args+['--per-task','1'])
    with pytest.raises(ValueError,match='resume settings differ'):
        runner.main()
