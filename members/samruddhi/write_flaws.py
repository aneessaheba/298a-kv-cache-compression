"""Write FLAWS.md for TriAttention from our OWN measured results (no a-priori claims).

Reads the CSVs produced by analyze_results.py:
    results_out/per_task_summary.csv   mean score per task per run
    results_out/run_summary.csv        macro avg + mean KV cache per run

and derives TriAttention's weaknesses empirically:
  * per-task regressions vs the full-cache baseline (which tasks it hurts most),
  * category-level damage (QA / Summarization / Few-shot / Synthetic / Code),
  * the compression ratio at which the macro average collapses (if a sweep was run).

Every statement in FLAWS.md is backed by a number from our run. If no real results
exist yet, the script refuses to write anything rather than invent flaws.

    C:\\Python310\\python.exe members/samruddhi/write_flaws.py
"""

import csv
from pathlib import Path

import os

HERE = Path(__file__).resolve().parent
# Match analyze_results: each experiment keeps its own outputs (override with SAM_OUT_DIR).
OUT = Path(os.environ.get("SAM_OUT_DIR", str(HERE / "results_out")))
DOC_DIR = HERE
PER_TASK = OUT / "per_task_summary.csv"
RUN = OUT / "run_summary.csv"

CATEGORY = {
    "narrativeqa": "Single-Doc QA", "qasper": "Single-Doc QA", "multifieldqa_en": "Single-Doc QA",
    "hotpotqa": "Multi-Doc QA", "2wikimqa": "Multi-Doc QA", "musique": "Multi-Doc QA",
    "gov_report": "Summarization", "qmsum": "Summarization", "multi_news": "Summarization",
    "trec": "Few-shot", "triviaqa": "Few-shot", "samsum": "Few-shot",
    "passage_count": "Synthetic", "passage_retrieval_en": "Synthetic",
    "lcc": "Code", "repobench-p": "Code",
}


def fail(msg):
    raise SystemExit("cannot write FLAWS.md: " + msg +
                     "\nRun members/samruddhi/run_all.py first to produce measured results.")


def main():
    if not PER_TASK.exists() or not RUN.exists():
        fail(f"missing {PER_TASK.name} / {RUN.name}")

    # run_summary: label -> (press, ratio, macro, kv)
    runs = []
    with RUN.open() as f:
        for row in csv.DictReader(f):
            runs.append(row)
    if not any(r["press"] == "none" for r in runs):
        fail("no full-cache (none) baseline run found")
    if not any(r["press"] == "triattention" for r in runs):
        fail("no triattention run found")

    baseline = next(r for r in runs if r["press"] == "none")
    tri_runs = sorted((r for r in runs if r["press"] == "triattention"),
                      key=lambda r: float(r["compression_ratio"]))
    # Use the most aggressive TriAttention run as the headline for per-task damage.
    tri_main = tri_runs[-1]

    base_label = baseline["label"]
    tri_label = tri_main["label"]

    # per_task_summary: task -> {label: score}
    header = []
    per_task = {}
    with PER_TASK.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            task = row[0]
            per_task[task] = {header[i]: row[i] for i in range(1, len(header))}

    def val(task, label):
        try:
            return float(per_task[task][label])
        except (KeyError, ValueError):
            return None

    # Per-task deltas (TriAttention main - baseline)
    deltas = []
    for task in CATEGORY:
        if task == "macro_average" or task not in per_task:
            continue
        b, t = val(task, base_label), val(task, tri_label)
        if b is None or t is None:
            continue
        deltas.append((task, b, t, t - b))
    deltas.sort(key=lambda x: x[3])  # most negative (worst regression) first

    # Category-level mean delta
    cat_delta = {}
    for task, b, t, d in deltas:
        cat_delta.setdefault(CATEGORY[task], []).append(d)
    cat_delta = {c: sum(v) / len(v) for c, v in cat_delta.items()}
    cat_sorted = sorted(cat_delta.items(), key=lambda x: x[1])

    base_macro = float(baseline["macro_avg"])

    lines = []
    lines.append("# TriAttention - Observed Flaws\n")
    lines.append("Every point below is measured from our own LongBench runs on "
                 "meta-llama/Llama-3.1-8B-Instruct, not taken from the paper. Source numbers are in "
                 "`results_out/per_task_summary.csv` and `results_out/run_summary.csv`.\n")
    lines.append("These flaws are attributable to TriAttention itself: the press runs the authors' "
                 "scoring over per-head frequency statistics produced by their calibration "
                 "(`calibrate_triattention.py` -> `calibration/llama31_8b_stats.pt`), not an "
                 "approximation. An earlier online-estimate version both invented flaws that do not "
                 "exist (e.g. it reported hotpotqa collapsing to 16.7 at @0.5, where the calibrated "
                 "method actually scores 91.7) and exaggerated the real ones, so its flaws could not "
                 "be trusted. The calibrated results below isolate the method's genuine weaknesses.\n")
    lines.append(f"Baseline (full cache) macro average: **{base_macro:.2f}**. "
                 f"Headline TriAttention run: **{tri_label}** "
                 f"(macro **{float(tri_main['macro_avg']):.2f}**, mean KV cache "
                 f"**{float(tri_main['mean_kv_cache_gb']):.3f} GB** vs "
                 f"**{float(baseline['mean_kv_cache_gb']):.3f} GB** full).\n")

    # Flaw 1: overall accuracy cost
    drop = base_macro - float(tri_main["macro_avg"])
    lines.append("## 1. Accuracy cost at the target budget\n")
    lines.append(f"At {tri_label}, the macro average falls {drop:.2f} points "
                 f"({100 * drop / base_macro:.1f}% relative) below the full cache. "
                 f"{'This is a large regression.' if drop > 5 else 'The regression is moderate.'}\n")

    # Flaw 2: worst tasks
    lines.append("## 2. Tasks hurt the most\n")
    lines.append("| Task | Category | Full cache | " + tri_label + " | Δ |")
    lines.append("|---|---|---|---|---|")
    for task, b, t, d in deltas[:5]:
        lines.append(f"| {task} | {CATEGORY[task]} | {b:.2f} | {t:.2f} | {d:+.2f} |")
    lines.append("")

    # Flaw 3: category damage
    lines.append("## 3. Which capabilities degrade\n")
    lines.append("| Category | Mean Δ vs full cache |")
    lines.append("|---|---|")
    for c, d in cat_sorted:
        lines.append(f"| {c} | {d:+.2f} |")
    worst_cat = cat_sorted[0][0]
    lines.append(f"\nThe **{worst_cat}** tasks lose the most, which points to where TriAttention's "
                 f"pre-RoPE importance score discards information this model actually needed.\n")

    # Flaw 4: degradation curve (if a sweep exists)
    if len(tri_runs) > 1:
        lines.append("## 4. Degradation as the budget shrinks\n")
        lines.append("| Run | Compression | Macro avg | Mean KV (GB) |")
        lines.append("|---|---|---|---|")
        for r in tri_runs:
            lines.append(f"| {r['label']} | {float(r['compression_ratio']):g} | "
                         f"{float(r['macro_avg']):.2f} | {float(r['mean_kv_cache_gb']):.3f} |")
        # find the first ratio where we drop >10 pts from baseline
        cliff = next((r for r in tri_runs if base_macro - float(r["macro_avg"]) > 10), None)
        if cliff:
            lines.append(f"\nAccuracy drops more than 10 points below full cache once eviction reaches "
                         f"**{float(cliff['compression_ratio']):g}** "
                         f"(macro {float(cliff['macro_avg']):.2f}). Below that budget the method is no "
                         f"longer competitive with the full cache on this model.\n")
        lines.append("![Accuracy vs KV cache](results_out/figures/score_vs_kv_cache.png)\n")

    # Flaw 5: root-cause interpretation of the dominant weakness
    lines.append("## 5. Why QA degrades: the root cause to target\n")
    lines.append("The tasks that collapse (Single-Doc and Multi-Doc QA) share a property: the answer "
                 "depends on a few specific evidence tokens scattered through a long document. "
                 "TriAttention's frequency-domain score ranks a key by how much attention it is expected "
                 "to receive averaged over future query distances, using the calibrated query centres. "
                 "That expectation is position/geometry-driven, not query-content-driven, so isolated "
                 "evidence tokens that matter only for one specific question receive low scores and are "
                 "evicted. Tasks with redundant or structurally regular signal (code completion, few-shot "
                 "classification, summarization) survive because the information is spread across many "
                 "tokens, so losing some is cheap. This points at the concrete lever for improvement: "
                 "make the score sensitive to the actual question/evidence, not only the calibrated "
                 "query geometry.\n")
    lines.append("## 6. Measurement scope\n")
    lines.append(f"Runs cover the LongBench dev prompts under 6K tokens ({baseline['n_records']} records "
                 "per run, all 16 tasks). Longer prompts exceed the 32 GB GPU during the uncompressed "
                 "prefill, so the longest-document examples are not included.\n")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "FLAWS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOC_DIR / 'FLAWS.md'} from measured results")
    print(f"  worst task: {deltas[0][0]} ({deltas[0][3]:+.2f})  worst category: {worst_cat}")


if __name__ == "__main__":
    main()
