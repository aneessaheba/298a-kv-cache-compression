#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
mkdir -p results/environment
# Freeze the lab's accelerator/runtime versions so extras cannot silently replace them.
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch, transformers, kvpress
assert torch.cuda.is_available(), 'Run setup in the lab CUDA image.'
versions = {name: version(name) for name in ('torch', 'transformers', 'kvpress', 'tokenizers', 'accelerate')}
Path('results/environment/runtime-constraints.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
print(versions)
x = torch.ones((16,16), device='cuda')
assert (x@x).sum().item() == 4096
PY
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -c results/environment/runtime-constraints.txt pytest rouge==1.0.1 rapidfuzz fuzzywuzzy==0.18.0 jieba==0.42.1 matplotlib
nvidia-smi > results/environment/nvidia_smi.txt
python -m pip freeze > results/environment/python_packages.txt
python -m pytest -q
printf '%s\n' 'Setup passed without replacing the pinned lab runtime. Use the shared harness for team runs.'
