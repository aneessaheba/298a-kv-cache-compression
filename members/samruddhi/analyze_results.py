"""Turn TriAttention LongBench runs into CSVs, plots, and a filled RESULTS.md.

Reads the harness output JSONL files produced by:
    python harness/run_longbench.py --press none
    python harness/run_longbench.py --press triattention --compression-ratio <r> [...]

and writes, under members/samruddhi/results_out/:
    per_record.csv          one row per (press, ratio, record): task, score, seconds, kv_cache_gb, tokens
    per_task_summary.csv     mean score per task for each run, plus the macro average
    run_summary.csv          one row per run: avg score, mean kv_cache_gb, mean seconds
    figures/*.png            per-task score bars, score-vs-kv_cache, latency
and regenerates members/samruddhi/RESULTS.md with the real numbers.

This script only looks at TriAttention and the full-cache (no-compression) baseline.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Both the naive baseline (press="none") and the TriAttention runs are written by the
# harness into results/longbench. KEEP_PRESSES filters to just those two.
LB_DIRS = [REPO / "results" / "longbench"]
OUT = Path(os.environ.get("SAM_OUT_DIR", str(HERE / "results_out")))
FIG = OUT / "figures"
# RESULTS.md / FLAWS.md live in the member folder.
DOC_DIR = HERE

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# TriAttention and the full-cache baseline only.
KEEP_PRESSES = {"none", "triattention"}

TASK_ORDER = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]


def run_label(press, ratio):
    if press == "none":
        return "Full cache"
    return f"TriAttention @{ratio:g}"


def load_runs():
    """Return {(press, ratio): [rows]} for the TriAttention + full-cache runs found."""
    runs = {}
    jsonls = []
    for d in LB_DIRS:
        if d.exists():
            jsonls.extend(sorted(d.glob("*.jsonl")))
    for jf in jsonls:
        rows = [json.loads(l) for l in jf.open(encoding="utf-8") if l.strip()]
        if not rows:
            continue
        press = rows[0].get("press")
        if press not in KEEP_PRESSES:
            continue
        ratio = rows[0].get("compression_ratio", 0.0)
        runs[(press, ratio)] = rows
    return runs


def write_csvs(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    # per_record.csv
    per_record = OUT / "per_record.csv"
    with per_record.open("w", encoding="utf-8") as f:
        f.write("press,compression_ratio,record_id,task,score,seconds,kv_cache_gb,prompt_tokens\n")
        for (press, ratio), rows in sorted(runs.items()):
            for r in rows:
                f.write(f"{press},{ratio},{r['record_id']},{r['task']},{r['score']:.6f},"
                        f"{r.get('seconds','')},{r.get('kv_cache_gb','')},{r.get('prompt_tokens','')}\n")

    # per_task_summary.csv + run_summary.csv
    per_task = defaultdict(dict)   # task -> {label: mean score}
    run_rows = []
    for (press, ratio), rows in sorted(runs.items()):
        label = run_label(press, ratio)
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r["score"])
        task_means = {t: 100 * sum(s) / len(s) for t, s in by_task.items()}
        for t, v in task_means.items():
            per_task[t][label] = v
        macro = sum(task_means.values()) / len(task_means) if task_means else 0.0
        kv = [r["kv_cache_gb"] for r in rows if r.get("kv_cache_gb") is not None]
        secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
        run_rows.append({
            "label": label, "press": press, "compression_ratio": ratio,
            "n_records": len(rows), "macro_avg": macro,
            "mean_kv_cache_gb": sum(kv) / len(kv) if kv else float("nan"),
            "mean_seconds": sum(secs) / len(secs) if secs else float("nan"),
        })

    labels = [r["label"] for r in run_rows]
    with (OUT / "per_task_summary.csv").open("w", encoding="utf-8") as f:
        f.write("task," + ",".join(labels) + "\n")
        for t in TASK_ORDER:
            if t in per_task:
                f.write(t + "," + ",".join(f"{per_task[t].get(l, ''):.2f}"
                        if isinstance(per_task[t].get(l), float) else "" for l in labels) + "\n")
        f.write("macro_average," + ",".join(f"{r['macro_avg']:.2f}" for r in run_rows) + "\n")

    with (OUT / "run_summary.csv").open("w", encoding="utf-8") as f:
        f.write("label,press,compression_ratio,n_records,macro_avg,mean_kv_cache_gb,mean_seconds\n")
        for r in run_rows:
            f.write(f"{r['label']},{r['press']},{r['compression_ratio']},{r['n_records']},"
                    f"{r['macro_avg']:.3f},{r['mean_kv_cache_gb']:.4f},{r['mean_seconds']:.3f}\n")
    return per_task, run_rows, labels


def make_plots(per_task, run_rows, labels):
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Per-task score bars, grouped by run
    tasks = [t for t in TASK_ORDER if t in per_task]
    x = range(len(tasks))
    width = 0.8 / max(1, len(labels))
    plt.figure(figsize=(14, 6))
    for i, lab in enumerate(labels):
        vals = [per_task[t].get(lab, 0.0) for t in tasks]
        plt.bar([xi + i * width for xi in x], vals, width, label=lab)
    plt.xticks([xi + width * (len(labels) - 1) / 2 for xi in x], tasks, rotation=45, ha="right")
    plt.ylabel("Score (0-100)")
    plt.title("TriAttention vs Full cache — LongBench per-task scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "per_task_scores.png", dpi=150)
    plt.close()

    # 2. Macro average vs mean KV-cache size (the equal-memory / degradation view).
    # With a ratio sweep this becomes the accuracy-vs-memory curve for TriAttention.
    plt.figure(figsize=(7, 5))
    tri = sorted([r for r in run_rows if r["press"] == "triattention"],
                 key=lambda r: r["mean_kv_cache_gb"])
    if len(tri) > 1:
        plt.plot([r["mean_kv_cache_gb"] for r in tri], [r["macro_avg"] for r in tri],
                 "-o", label="TriAttention sweep")
    for r in run_rows:
        plt.scatter(r["mean_kv_cache_gb"], r["macro_avg"], s=90, zorder=3)
        plt.annotate(r["label"], (r["mean_kv_cache_gb"], r["macro_avg"]),
                     textcoords="offset points", xytext=(6, 6))
    plt.xlabel("Mean KV cache kept (GB)")
    plt.ylabel("Macro-average score (0-100)")
    plt.title("Accuracy vs KV-cache memory")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "score_vs_kv_cache.png", dpi=150)
    plt.close()

    # 3. Macro average bar per run
    plt.figure(figsize=(7, 5))
    plt.bar([r["label"] for r in run_rows], [r["macro_avg"] for r in run_rows])
    plt.ylabel("Macro-average score (0-100)")
    plt.title("TriAttention vs Full cache - macro average")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "macro_average.png", dpi=150)
    plt.close()

    # 4. Per-task delta vs full cache (at the most aggressive TriAttention run) - shows the flaw
    tri_runs = [r for r in run_rows if r["press"] == "triattention"]
    base_label = next((r["label"] for r in run_rows if r["press"] == "none"), None)
    if tri_runs and base_label:
        worst = max(tri_runs, key=lambda r: r["compression_ratio"])["label"]
        deltas = []
        for t in tasks:
            b = per_task[t].get(base_label)
            v = per_task[t].get(worst)
            if isinstance(b, float) and isinstance(v, float):
                deltas.append((t, v - b))
        deltas.sort(key=lambda x: x[1])
        names = [d[0] for d in deltas]
        vals = [d[1] for d in deltas]
        colors = ["#C44E52" if v < 0 else "#55A868" for v in vals]
        plt.figure(figsize=(12, 6))
        plt.barh(names, vals, color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel(f"Score change vs full cache (points)  -  {worst}")
        plt.title("Per-task impact of TriAttention: where it hurts (red) and holds up (green)")
        plt.tight_layout()
        plt.savefig(FIG / "per_task_delta.png", dpi=150)
        plt.close()

    # 5. Memory saved vs accuracy kept, per run (dual view of the tradeoff)
    base = next((r for r in run_rows if r["press"] == "none"), None)
    if base:
        import numpy as np
        labels_b = [r["label"] for r in run_rows]
        acc_kept = [100 * r["macro_avg"] / base["macro_avg"] for r in run_rows]
        mem_saved = [100 * (1 - r["mean_kv_cache_gb"] / base["mean_kv_cache_gb"]) for r in run_rows]
        xb = np.arange(len(labels_b))
        w = 0.38
        plt.figure(figsize=(10, 5))
        plt.bar(xb - w / 2, acc_kept, w, label="Accuracy kept (% of full)", color="#4C72B0")
        plt.bar(xb + w / 2, mem_saved, w, label="KV memory saved (% vs full)", color="#DD8452")
        plt.xticks(xb, labels_b, rotation=15, ha="right")
        plt.ylabel("Percent")
        plt.title("The tradeoff: memory saved vs accuracy kept")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "tradeoff.png", dpi=150)
        plt.close()


def make_animations(per_task, run_rows, labels):
    """Animated GIFs: per-task bars morphing across the compression sweep, and the
    accuracy-vs-memory point walking down the curve."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    tasks = [t for t in TASK_ORDER if t in per_task]

    # Animation 1: per-task bars, one frame per run (full cache -> harder compression)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(tasks))

    def frame(i):
        ax.clear()
        lab = labels[i]
        vals = [per_task[t].get(lab, 0.0) for t in tasks]
        color = "#4C72B0" if run_rows[i]["press"] == "none" else "#C44E52"
        ax.bar(x, vals, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (0-100)")
        ax.set_title(f"LongBench per-task scores — {lab}  "
                     f"(macro {run_rows[i]['macro_avg']:.1f}, "
                     f"{run_rows[i]['mean_kv_cache_gb']:.3f} GB)")
        fig.tight_layout()

    anim = FuncAnimation(fig, frame, frames=len(labels), interval=1200)
    anim.save(FIG / "per_task_sweep.gif", writer=PillowWriter(fps=1))
    plt.close(fig)

    # Animation 2: accuracy-vs-memory curve, point walking from full cache down
    ordered = sorted(run_rows, key=lambda r: -r["mean_kv_cache_gb"])  # high memory -> low
    xs = [r["mean_kv_cache_gb"] for r in ordered]
    ys = [r["macro_avg"] for r in ordered]
    fig, ax = plt.subplots(figsize=(7, 5))

    def frame2(i):
        ax.clear()
        ax.plot(xs, ys, "-", color="#CCCCCC", zorder=1)
        ax.plot(xs[: i + 1], ys[: i + 1], "-o", color="#C44E52", zorder=2)
        ax.scatter([xs[i]], [ys[i]], s=140, color="#C44E52", zorder=3)
        ax.annotate(ordered[i]["label"], (xs[i], ys[i]),
                    textcoords="offset points", xytext=(8, 8))
        ax.set_xlim(min(xs) - 0.03, max(xs) + 0.05)
        ax.set_ylim(min(ys) - 5, max(ys) + 5)
        ax.set_xlabel("Mean KV cache kept (GB)")
        ax.set_ylabel("Macro-average score (0-100)")
        ax.set_title("Accuracy vs KV-cache memory")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    anim2 = FuncAnimation(fig, frame2, frames=len(ordered), interval=1200)
    anim2.save(FIG / "accuracy_vs_memory.gif", writer=PillowWriter(fps=1))
    plt.close(fig)


def write_results_md(per_task, run_rows, labels):
    tasks = [t for t in TASK_ORDER if t in per_task]
    lines = []
    lines.append("# TriAttention on LongBench - Results\n")
    lines.append("**Method:** TriAttention (Mao et al., ICML 2026, "
                 "[arXiv:2604.04921](https://arxiv.org/abs/2604.04921); "
                 "repo [WeianMao/triattention](https://github.com/WeianMao/triattention))  ")
    lines.append(f"**Model:** {MODEL} (bf16)  ")
    n_rec = run_rows[0]["n_records"] if run_rows else 0
    lines.append(f"**Benchmark:** LongBench v1 dev set, 16 tasks "
                 f"({n_rec} records per run, prompts capped at 6K tokens).  ")
    lines.append("**Baseline:** full cache (no compression). This document covers TriAttention only.\n")

    lines.append("## What TriAttention does\n")
    lines.append("TriAttention scores each cached key with a frequency-domain trigonometric series built "
                 "from **precomputed per-head query statistics** (the authors' calibration), then keeps the "
                 "highest-scoring tokens plus a recent window and the initial sink tokens. It is implemented "
                 "here as a kvpress `ScorerPress` (`members/samruddhi/triattention_press.py`) running the "
                 "authors' exact scoring over stats from `calibrate_triattention.py`, so it goes through the "
                 "shared harness with the same fair split and official metrics as the full-cache baseline.\n")

    lines.append("## Runs\n")
    lines.append("| Run | Records | Macro avg | Mean KV cache (GB) | Mean sec/record |")
    lines.append("|---|---|---|---|---|")
    for r in run_rows:
        lines.append(f"| {r['label']} | {r['n_records']} | {r['macro_avg']:.2f} | "
                     f"{r['mean_kv_cache_gb']:.4f} | {r['mean_seconds']:.2f} |")
    lines.append("")

    lines.append("## Per-task scores (0-100)\n")
    lines.append("| Task | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 1))
    for t in tasks:
        row = " | ".join(f"{per_task[t].get(l, float('nan')):.2f}"
                         if isinstance(per_task[t].get(l), float) else "-" for l in labels)
        lines.append(f"| {t} | {row} |")
    macro = " | ".join(f"{r['macro_avg']:.2f}" for r in run_rows)
    lines.append(f"| **macro average** | {macro} |")
    lines.append("")

    lines.append("## Figures\n")
    lines.append("![Per-task scores](results_out/figures/per_task_scores.png)\n")
    lines.append("![Accuracy vs KV cache](results_out/figures/score_vs_kv_cache.png)\n")
    lines.append("![Macro average](results_out/figures/macro_average.png)\n")

    lines.append("## Data files\n")
    lines.append("- `results_out/per_record.csv` — every record's task, score, latency, KV-cache size\n"
                 "- `results_out/per_task_summary.csv` — mean score per task per run\n"
                 "- `results_out/run_summary.csv` — macro average, mean KV cache, mean latency per run\n"
                 "- `results_out/run_log.txt` — full stdout/stderr of the harness runs\n")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No TriAttention/full-cache run files found in {LB_DIR}. "
                         f"Run harness/run_longbench.py first.")
    per_task, run_rows, labels = write_csvs(runs)
    make_plots(per_task, run_rows, labels)
    try:
        make_animations(per_task, run_rows, labels)
    except Exception as e:  # animation is a nicety; never fail the run over it
        print(f"animation step skipped: {e}")
    write_results_md(per_task, run_rows, labels)
    print(f"wrote CSVs + figures to {OUT}")
    print(f"regenerated {DOC_DIR / 'RESULTS.md'}")
    for r in run_rows:
        print(f"  {r['label']:22s} macro={r['macro_avg']:.2f}  kv={r['mean_kv_cache_gb']:.4f}GB")


if __name__ == "__main__":
    main()
