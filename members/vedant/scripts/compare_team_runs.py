#!/usr/bin/env python3
"""Compare actual shared-harness files; explicitly convert 0–1 to 0–100 for CIs."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path

MEMBER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBER / 'src'))
from turboquant.compare import compare_runs, _write_csv
from turboquant.protocol import DATA_DIR, validate_dataset, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=shared_harness.jsonl; include full=...')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--adaptive', action='store_true', help='exclude the frozen 32 calibration IDs from all runs')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    expected = {r['record_id']: r['task'] for r in validate_dataset(args.data)}
    excluded = set(json.loads((DATA_DIR/'adaptive_excluded_ids.json').read_text())['record_ids']) if args.adaptive else set()
    original = {}
    oom_counts = {}
    computed_memory = {}
    with tempfile.TemporaryDirectory(prefix='team6_score_adapter_') as folder:
        converted = {}
        for item in args.run:
            name, separator, path = item.partition('=')
            if not separator or name in converted:
                parser.error('each run must be a unique NAME=path')
            original[name] = Path(path)
            rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            if len(rows) != len(expected) or {r['record_id']: r['task'] for r in rows} != expected:
                parser.error(f'{name} must contain all 800 expected unique IDs/tasks before exclusion')
            if any(not 0 <= float(r['score']) <= 1 or r.get('score_scale','0-1') != '0-1' for r in rows):
                parser.error(f'{name} is not a shared-harness 0–1 score file')
            if any('press' not in r for r in rows):
                parser.error(f'{name} lacks shared-harness press metadata; use the shared runner and scorer')
            retained = [r for r in rows if r['record_id'] not in excluded]
            oom_counts[name] = sum(bool(r.get('oom')) for r in retained)
            memory = [r['kv_cache_gb'] for r in retained if r.get('kv_cache_gb') is not None]
            computed_memory[name] = sum(memory)/len(memory) if memory else None
            for r in rows:
                r['score'] = float(r['score'])*100
                r['source_score_scale'] = '0-1'
                r['score_scale'] = '0-100'
                r['scoring_policy'] = 'team-shared-harness-rapidfuzz'
            target = Path(folder)/(name+'.jsonl')
            target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            converted[name] = target
        report = compare_runs(converted, exclude_ids=excluded)
    report['runs'] = {name: str(p) for name,p in original.items()}
    report['source_sha256'] = {name: sha256(p) for name,p in original.items()}
    report['source_score_scale'] = '0-1'
    report['reported_score_scale'] = '0-100 (percentage points)'
    report['evaluation_set'] = 'adaptive_768' if args.adaptive else 'team_800'
    report['oom_counts_included_as_failures'] = oom_counts
    report['mean_computed_kv_cache_gb'] = computed_memory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'team_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for key in ('overall','per_task','paired'):
        _write_csv(args.output_dir/(key+'.csv'),report[key])
    print(json.dumps({'records':report['paired_record_count'],'overall':report['overall']},indent=2))


if __name__ == '__main__':
    main()
