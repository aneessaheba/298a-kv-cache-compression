# Lab commands (see also members/charvee/README.md)

Mount this repo’s parent folder (`charvee_residualkv` layout: `298a-kv-cache-compression` + `DeltaKV`) at `/app`.

```bash
cd /app/298a-kv-cache-compression
pip install transformers==5.2.0 rouge rapidfuzz loguru
hf auth login
python members/charvee/run_residualkv.py --compression-ratio 0.9 --per-task 2
nohup bash -c 'for r in 0.9 0.75 0.8 0.95; do python members/charvee/run_residualkv.py --compression-ratio $r; done' > residualkv.log 2>&1 &
```
