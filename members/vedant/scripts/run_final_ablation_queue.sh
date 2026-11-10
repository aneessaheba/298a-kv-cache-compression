#!/usr/bin/env bash
set -u
cd /app/298a-kv-cache-compression || exit 1
source members/vedant/.venv/bin/activate || exit 1
export PYTHONUNBUFFERED=1

own_lock=/tmp/vedant_tq_final_queue.lock
shared_lock=/tmp/vedant_tq_remaining.lock
if ! mkdir "$own_lock" 2>/dev/null; then
    echo "A final queue already exists. No second queue started."
    exit 1
fi
trap 'rmdir "$own_lock" 2>/dev/null' EXIT
until mkdir "$shared_lock" 2>/dev/null; do
    echo "Waiting for the previous queue..."
    sleep 30
done
trap 'rmdir "$shared_lock" "$own_lock" 2>/dev/null' EXIT

while pgrep -f '^[^ ]*python[^ ]*( -u)? harness/run_longbench[.]py' >/dev/null ||
      pgrep -f '^[^ ]*bash members/vedant/scripts/run_harness_longbench[.]sh' >/dev/null; do
    echo "Waiting for the current experiment..."
    sleep 30
done

finished() {
python - "$1" <<'PY'
import hashlib, json, sys
from pathlib import Path
from turboquant.protocol import fingerprint, sha256, validate_dataset
try:
    name = sys.argv[1]
    p = Path('results/turboquant_full_clean') / ('Llama-3.1-8B-Instruct_' + name + '_0.0.jsonl')
    rows = [json.loads(line) for line in p.open()]
    manifest = json.loads(Path(str(p) + '.run.json').read_text())
    spec = manifest['specification']
    data = validate_dataset(Path('data/processed_longbench/processed_longbench_dev.jsonl'))
    assert len(rows) == len({r['record_id'] for r in rows}) == 800
    assert {r['record_id']: r['task'] for r in rows} == {r['record_id']: r['task'] for r in data}
    assert spec['press'] == name and spec['dtype'] == 'bfloat16'
    assert spec['scorer_sha256'] == sha256(Path('harness/longbench.py'))
    assert spec['runner_sha256'] == sha256(Path('harness/run_longbench.py'))
    files = sorted([Path('members/vedant/turboquant_press.py')] + list(Path('members/vedant/src').rglob('*.py')))
    assert spec['implementation_sha256'] == hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()
    assert spec['dataset_sha256'] == sha256(Path('data/processed_longbench/processed_longbench_dev.jsonl'))
    assert fingerprint(spec) == manifest['run_fingerprint']
    assert all(r['press'] == name and not r.get('oom') and
               r['run_fingerprint'] == manifest['run_fingerprint'] and
               r['score_scale'] == '0-1' and 0 <= r['score'] <= 1 for r in rows)
    summary = json.loads(p.with_suffix('.summary.json').read_text())
    assert summary['n_records'] == 800
except Exception:
    sys.exit(1)
PY
}

for press in turboquant_4bit turboquant_mse_3bit turboquant_mse_4bit turboquant_mse_2.5bit turboquant_mse_3.5bit
do
    if finished "$press"; then
        echo "SKIP: $press already has 800 valid records."
        printf '%s SKIP\n' "$press" >> turboquant_final_queue_status.txt
        continue
    fi
    echo "========== START: $press =========="
    date -Is
    if python -u harness/run_longbench.py --press "$press" --output-dir results/turboquant_full_clean; then
        code=0
    else
        code=$?
    fi
    if [ "$code" -eq 0 ] && finished "$press"; then
        result=PASS
    else
        result=NEEDS_REVIEW
    fi
    printf '%s exit=%s %s\n' "$press" "$code" "$result" | tee -a turboquant_final_queue_status.txt
done
echo "QUEUE FINISHED. Check the status file and preserve all outputs."
date -Is
