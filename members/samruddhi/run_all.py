"""One command: run TriAttention + full-cache on LongBench, log everything, then build CSVs/plots/RESULTS.md.

Prerequisites (both are gated and must be satisfied first):
  * `huggingface-cli login` with a token that has accepted the Llama-3.1-8B license
  * data/processed_longbench/processed_longbench_dev.jsonl present
    (run members/samruddhi/build_dev_data.py, or unzip processed_longbench.zip)

    C:\\Python310\\python.exe members/samruddhi/run_all.py --compression-ratio 0.9 --per-task 50

Only TriAttention and the full-cache baseline are run; no other methods are touched.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNNER = REPO / "harness" / "run_longbench.py"
DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"
LOG = HERE / "results_out" / "run_log.txt"


def _run_once(cmd, logf):
    line = ">>> " + " ".join(str(c) for c in cmd) + "\n"
    print(line, end="")
    logf.write(line)
    logf.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for out_line in proc.stdout:
        print(out_line, end="")
        logf.write(out_line)
        logf.flush()
    proc.wait()
    return proc.returncode


def run(cmd, logf, max_restarts=12):
    """Run a ratio; if it crashes on a fatal CUDA error (exit 99), restart it. The harness is
    resumable (it skips records already written), so each restart continues past the bad record."""
    attempt = 0
    while True:
        rc = _run_once(cmd, logf)
        if rc == 0:
            return
        attempt += 1
        if rc == 99 and attempt <= max_restarts:
            msg = f"[run_all] ratio crashed (fatal CUDA); restart {attempt}/{max_restarts}, resuming...\n"
            print(msg, end=""); logf.write(msg); logf.flush()
            continue
        # non-fatal nonzero, or too many restarts: log and move on to the next ratio
        msg = f"[run_all] command exited {rc} after {attempt} attempt(s); continuing to next step\n"
        print(msg, end=""); logf.write(msg); logf.flush()
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compression-ratios", default="0.5,0.75,0.9,0.95",
                    help="comma-separated eviction fractions to sweep (0.9 = keep 10%)")
    ap.add_argument("--per-task", type=int, default=None, help="records per task (default: all 50)")
    ap.add_argument("--tasks", default="", help="comma-separated subset of tasks")
    ap.add_argument("--max-prompt-tokens", type=int, default=None)
    ap.add_argument("--chunk-prefill", type=int, default=None,
                    help="prefill long contexts in chunks of this many tokens (enables the full dev set)")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the full-cache 'none' run and reuse the shared naive baseline in results/longbench")
    args = ap.parse_args()
    ratios = [float(x) for x in args.compression_ratios.split(",") if x.strip()]

    if not DATA.exists():
        raise SystemExit(f"missing dev data: {DATA}\n"
                         f"run members/samruddhi/build_dev_data.py or unzip processed_longbench.zip first")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    common = [py, str(RUNNER)]
    if args.per_task is not None:
        common += ["--per-task", str(args.per_task)]
    if args.tasks:
        common += ["--tasks", args.tasks]
    if args.max_prompt_tokens is not None:
        common += ["--max-prompt-tokens", str(args.max_prompt_tokens)]
    if args.chunk_prefill is not None:
        common += ["--chunk-prefill", str(args.chunk_prefill)]

    with LOG.open("w", encoding="utf-8") as logf:
        # Full-cache baseline, then a sweep of TriAttention eviction ratios so the
        # flaws analysis can see where accuracy degrades.
        run(common + ["--press", "none"], logf)
        for r in ratios:
            run(common + ["--press", "triattention", "--compression-ratio", str(r)], logf)

    with open(LOG, "a", encoding="utf-8") as logf:
        run([py, str(HERE / "analyze_results.py")], logf)
        # Derive FLAWS.md from the measured results (sweep included).
        run([py, str(HERE / "write_flaws.py")], logf)


if __name__ == "__main__":
    main()
