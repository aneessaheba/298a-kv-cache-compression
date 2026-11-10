"""Paired comparison of a run against the shared full-cache baseline (same record_ids only).

    python members/manjot/compare.py results/longbench/Llama-3.1-8B-Instruct_rdkv_0.9.jsonl
    python members/manjot/compare.py RUN.jsonl --baseline OTHER.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def summarize(rows):
    """Same as harness/longbench.summarize: per-task mean (0-100) plus macro average."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row["score"])
    out = {t: round(100 * sum(s) / len(s), 2) for t, s in by_task.items()}
    out["average"] = round(sum(out.values()) / len(out), 2) if out else 0.0
    return out

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--baseline", default=str(REPO / "results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl"))
args = p.parse_args()

run = {r["record_id"]: r for r in map(json.loads, open(args.run))}
base = {r["record_id"]: r for r in map(json.loads, open(args.baseline))}
ids = sorted(set(run) & set(base))
if not ids:
    sys.exit("no shared record_ids")

s_run = summarize([run[i] for i in ids])
s_base = summarize([base[i] for i in ids])
print(f"{'task':22s} {'baseline':>9s} {'run':>9s} {'delta':>8s}   ({len(ids)} paired records)")
for t in s_base:
    if t in s_run:
        print(f"{t:22s} {s_base[t]:9.2f} {s_run[t]:9.2f} {s_run[t] - s_base[t]:+8.2f}")

rows = [run[i] for i in ids]
n = len(rows)
print(f"\nOOM records: {sum(r.get('oom', False) for r in rows)} / {n}")
print(f"mean kv_cache_gb (harness): {sum(r['kv_cache_gb'] for r in rows) / n:.4f}"
      f"   baseline: {sum(base[i]['kv_cache_gb'] for i in ids) / n:.4f}")
if "rdkv_kept_fraction_bits" in rows[0]:
    ok = [r for r in rows if "rdkv_kept_fraction_bits" in r]
    print(f"mean RDKV kept fraction (bits, incl. metadata): {sum(r['rdkv_kept_fraction_bits'] for r in ok) / len(ok):.4f}")
print(f"total time: {sum(r['seconds'] for r in rows) / 3600:.2f} h   "
      f"(baseline on its GPU: {sum(base[i]['seconds'] for i in ids) / 3600:.2f} h)")
