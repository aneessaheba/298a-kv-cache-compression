# ResidualKV (Charvee)

[ResidualKV](https://arxiv.org/abs/2602.08005), called DeltaKV in the first arXiv version, keeps a few layers whole and, in the other layers, replaces dropped tokens with small compressed residuals against similar tokens that are kept. It uses the authors' code ([CURRENTF/DeltaKV](https://github.com/CURRENTF/DeltaKV), commit `11a837b`) and their released Llama-3.1-8B-Instruct compressor, `JitaiHao/Llama-3.1-8B-Instruct-Compressor`.

Settings follow Charvee's configs (`CharveeSaraiya/Deltakv`, `runs/configs/`). The runner `run_residualkv.py` was set up by Anees so that ResidualKV uses the same 800 dev records, prompts, scorer and results format as the other methods.

## How to run (lab RTX 5090, nothing to install)

```bash
cd /app/298a-kv-cache-compression
git clone https://github.com/CURRENTF/DeltaKV ../DeltaKV
git -C ../DeltaKV checkout 11a837b21acfa475957210c5b2d339b89c1b7e3b

# 1. Quick check, 2 records per task (about 10 minutes). Stop here if it fails.
python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2

# 2. Full runs, 0.75 and 0.9 first, then 0.8 and 0.95
python members/charvee/run_residualkv.py --compression-ratio 0.9
python members/charvee/run_residualkv.py --compression-ratio 0.75
python members/charvee/run_residualkv.py --compression-ratio 0.8
python members/charvee/run_residualkv.py --compression-ratio 0.95
```

Results go to `results/residualkv/longbench/`. If the machine stops, run the same command again and it picks up where it left off. Back up that folder to Drive before leaving the lab.

## Memory

ResidualKV keeps layers 0, 1, 2, 8 and 18 whole (4-bit), plus 8 sink and 128 recent tokens, and stores compressed residuals for dropped tokens. So `--compression-ratio 0.9` is **not** a 10% cache. Each record logs the bytes actually held in its cache (`kv_cache_gb`) and the bf16 size of a full cache for the same tokens (`full_kv_cache_gb`). Compare methods on `kv_cache_gb`.

## Differences from the shared harness

- ResidualKV compresses the whole prompt, keeping the last 128 tokens, which usually include the question. kvpress methods compress only the document.
- The attention kernel is PyTorch's `sdpa`, because the lab image has no flash-attn.

## Status

- [ ] Quick check on the lab GPU
- [ ] Full runs at 0.75 and 0.9
- [ ] Full runs at 0.8 and 0.95
- [ ] Results table and report section
