# RTX 5090 runbook — current shared protocol

Use the university-approved CUDA Docker image and your permitted working folder. This checkout targets the team image Python 3.11.13 / Torch 2.7.1+cu128 / Transformers 5.2.0 / kvpress 0.5.5. Run inside the container, not the Windows host. The Drive guide's five uniform runs and fork URL predate this branch.

## 1. Code and data

```bash
cd /app
git clone --branch vedant/turboquant-reference-implementation https://github.com/aneessaheba/298a-kv-cache-compression.git
cd 298a-kv-cache-compression
# Extract the supplied data ZIP here, or put the exact processed file at:
# data/processed_longbench/processed_longbench_dev.jsonl
python -m zipfile -e ../longbench_dev_data.zip .
sha256sum data/processed_longbench/processed_longbench_dev.jsonl
```

Expected hash: `fc9d8649f535619c6b3dbe31ad158b112131d85230f92cd253c2e32bb1b7a47c`. Do not overwrite this branch with the older `turboquant_fixed.zip`.

## 2. Preserve the lab runtime

```bash
bash members/vedant/scripts/lab_setup.sh
source members/vedant/.venv/bin/activate
hf auth login
hf auth whoami
```

Enter your own read token only at the authentication prompt. The setup uses `--system-site-packages`, performs a CUDA matrix operation, freezes runtime constraints and installs this package without replacing Torch. If the environment lacks the expected team packages, use the team's approved image/setup; do not install a replacement Torch blindly.

## 3. Run priorities

```bash
nohup bash members/vedant/scripts/run_harness_longbench.sh > turboquant.log 2>&1 &
tail -f turboquant.log
```

The 32-record smoke has a separate output directory; any OOM stops the script before full runs. Main order: 2.5 and 3.5 bits with QJL; 2.5/3.5 without QJL; uniform 3/4 with and without QJL; uniform 2 with and without QJL. These are ten settings, not the older guide's five. Runtime per setting is unmeasured for this implementation; plan a checkpoint after smoke and the first full mixed run rather than promising the old one-hour estimate. Each prediction flushes immediately.

To stop after one setting or use a shorter session:

```bash
python harness/run_longbench.py --press turboquant_2.5bit
python harness/run_longbench.py --press turboquant_3.5bit
```

Re-running identical commands resumes verified IDs. A changed dataset, code, selection, allocation or runtime rejects resume. Do not use a smoke file as a full result. OOM rows are recorded as failures; rerunning skips them too. Investigate OOM and use a new output directory for a controlled rerun, retaining failed evidence.

## 4. Validate and compare

Each complete full run must contain 800 unique IDs, 50 per task; inspect `oom` counts. Scores in JSONL are 0–1. Summary scores are percentages, with the final average computed per task then across tasks. Baseline 47.72 is a reference value, not a hard upper bound on a compressed method.

```bash
python members/vedant/scripts/compare_team_runs.py \
 --data data/processed_longbench/processed_longbench_dev.jsonl \
 --run full=results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl \
 --run tq25=results/longbench/Llama-3.1-8B-Instruct_turboquant_2.5bit_0.0.jsonl \
 --output-dir results/turboquant_comparison
```

No independent KIVI artifacts are required. Add other complete shared runs when available. `kv_cache_gb` counts logical document representation plus uncompressed question/chat suffix; it does not prove allocated memory savings. Peak GPU allocation and run time remain separate fields.

## 5. Optional adaptive work

Use the frozen calibration split and the instructions in ADAPTIVE_TURBOQUANT.md. Then:

```bash
python harness/run_longbench.py --press turboquant_adaptive \
 --turboquant-layer-config members/vedant/results/calibration/allocation/adaptive_allocation.json
```

The chosen allocation must cover all 32 model layers. Compare all method files using `compare_team_runs.py --adaptive` so every run excludes the same 32 calibration IDs. Its 768-record scores must not be compared directly to the 800-record headline baseline.

## 6. Bring back and resume safely

```bash
{ date; nvidia-smi; git rev-parse HEAD; python -m pip freeze; } > lab_env.txt 2>&1
tar -czf /app/turboquant_results.tgz results/longbench results/turboquant_smoke \
 members/vedant/results/environment turboquant.log lab_env.txt
```

Include optional comparison/calibration directories if produced. Retain raw predictions, `.run.json` manifests, summaries, environment records and logs. Upload the archive yourself to the team's designated results folder. Verify the archive is accessible before deleting local lab files under university rules. Resume requires restoring the saved files into an identical checkout/runtime; deleting the only copy prevents resume. Log out of Hugging Face and follow lab cleanup rules.
