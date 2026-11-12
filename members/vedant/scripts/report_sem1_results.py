"""Validate archived RTX 5090 runs and regenerate Semester 1 tables and figures.

Run from the repository root with the frozen team development dataset. This is
CPU-only analysis; it never loads model weights or changes recorded predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "harness"), str(REPO / "members/vedant/src")]
import longbench
from turboquant.protocol import fingerprint, sha256, validate_dataset
from turboquant.adaptive import _aggregate_reports, allocate_layers
from turboquant.splits import make_splits

SOURCE_COMMIT = "82d40f72eb8c3a8313c4da95d2dbe22913aaad7a"
SOURCE_HASHES = {
    "scorer_sha256": "afeeb573c1c5068b2c594d6a0584bc6649f46d6dae5596d9f1e63ae07abd0019",
    "runner_sha256": "c9ba5d5cd6dae2286fc4426ae0077645f0300c7196c765fd9b423b429f41ad2a",
    "implementation_sha256": "3c874f0c6c1a3f5eecf4c2e0b32d5f6f055488c4c64fb6a2d8f313339599eda0",
}
PREFIX = "Llama-3.1-8B-Instruct_"


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def macro(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row["score"])
    return float(np.mean([np.mean(values) for values in groups.values()]) * 100)


def paired(left, right, draws):
    """Left minus right; paired record resampling within each task."""
    lookup = {r["record_id"]: r for r in right}
    assert {r["record_id"] for r in left} == set(lookup)
    tasks = sorted({r["task"] for r in left})
    boot = np.zeros(draws)
    rng = np.random.default_rng(20260)
    for task in tasks:
        delta = np.array([r["score"] - lookup[r["record_id"]]["score"]
                          for r in left if r["task"] == task])
        boot += rng.choice(delta, size=(draws, len(delta))).mean(axis=1) * 100 / len(tasks)
    return {"difference_pp": macro(left) - macro(right),
            "ci95_pp": np.quantile(boot, [.025, .975]).tolist()}


def save_plot(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    svg = out / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("use at least 1000 bootstrap samples")
    evidence = REPO / "members/vedant/results/sem1"
    out = evidence / "analysis"
    out.mkdir(exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    data = validate_dataset(args.data)
    source = {r["record_id"]: r for r in data}
    # The scorer is fixed to the version used in the lab, even if main advances.
    assert sha256(REPO / "harness/longbench.py") == SOURCE_HASHES["scorer_sha256"]
    baseline_path = REPO / "results/longbench" / f"{PREFIX}none_0.0.jsonl"
    baseline = read_jsonl(baseline_path)
    paths = sorted((evidence / "fixed").glob("*.jsonl"),
                   key=lambda p: ("mse_" in p.name, float(p.name.split("turboquant_")[-1].replace("mse_", "").split("bit")[0])))
    paths += sorted((evidence / "adaptive/longbench").glob("*.jsonl"))
    assert len(paths) == 10
    runs, summaries = {}, []
    checksums = {}
    for path in [baseline_path] + paths:
        rows = read_jsonl(path)
        assert len(rows) == len({r["record_id"] for r in rows}) == 800
        assert {r["record_id"]: r["task"] for r in rows} == {r["record_id"]: r["task"] for r in data}
        assert not any(r.get("oom") or not r["prediction"].strip() for r in rows)
        for row in rows:
            record = source[row["record_id"]]
            rescored = longbench.score(row["task"], row["prediction"], record["answers"], record["all_classes"])
            assert np.isclose(rescored, row["score"], atol=1e-10, rtol=0)
        name = "full_cache" if path == baseline_path else rows[0]["press"]
        runs[name] = rows
        if name != "full_cache":
            manifest = json.loads(Path(str(path) + ".run.json").read_text())
            spec = manifest["specification"]
            assert fingerprint(spec) == manifest["run_fingerprint"]
            assert spec["dataset_sha256"] == sha256(args.data)
            assert all(spec[k] == v for k, v in SOURCE_HASHES.items())
            assert spec["press"] == name and spec["dtype"] == "bfloat16"
            if name == "turboquant_adaptive":
                assert spec["allocation_sha256"] == sha256(evidence / "adaptive/allocation/adaptive_allocation.json")
            for row in rows:
                assert row["run_fingerprint"] == manifest["run_fingerprint"]
                assert row["dtype"] == "bfloat16" and row["score_scale"] == "0-1"
                assert row["model_revision"] == "0e9e39f249a16976918f6564b8830bc894c89659"
        saved = json.loads(path.with_suffix(".summary.json").read_text())
        assert saved["n_records"] == 800
        assert saved["scores"] == longbench.summarize(rows)
        checksums[str(path.relative_to(REPO))] = sha256(path)
        summaries.append({"method": name, "records": 800,
            "score": saved["scores"]["average"], "raw_macro_score": macro(rows),
            "mean_computed_cache_gb": float(np.mean([r["kv_cache_gb"] for r in rows])),
            "peak_gpu_gb": max(r["peak_gpu_gb"] for r in rows),
            "summed_record_minutes": sum(r["seconds"] for r in rows) / 60,
            "vs_full_cache": paired(rows, baseline, args.bootstrap_samples)})
    base_memory = summaries[0]["mean_computed_cache_gb"]
    for row in summaries:
        row["computed_cache_percent"] = row["mean_computed_cache_gb"] / base_memory * 100
    splits = make_splits(data)
    excluded = {r["record_id"] for r in splits["calibration"]}
    split_manifest = json.loads((evidence / "adaptive/split_manifest.json").read_text())
    assert excluded == set(split_manifest["record_ids"])
    reports = defaultdict(list)
    for sample in range(16):
        folder = evidence / f"adaptive/calibration/sample_{sample}"
        meta = json.loads((folder / "calibration_kv.json").read_text())
        assert meta["source_record_id"] == splits["calibration"][2 * sample]["record_id"]
        assert meta["query_source"] == "model_post_rope" and meta["input_tokens"] <= 4096
        assert meta["dtype"] == "torch.bfloat16" and meta["prompt_format"] == "shared_kvpress"
        for layer in range(32):
            report = json.loads((folder / f"layer_{layer}/real_kv_report.json").read_text())
            assert report["query_source"] == "model_post_rope" and len(report["rows"]) == 6
            assert report["quantizer_seed"] == 42 + layer * 100003
            assert all(np.isfinite(r["key_normalized_mse"]) and np.isfinite(r["value_normalized_mse"]) for r in report["rows"])
            reports[layer].append(report)
    allocation = allocate_layers(_aggregate_reports(reports, key_weight=1, value_weight=1),
                                 reference_bits=3, reference_qjl=True, qjl_policy="on")
    saved_allocation = json.loads((evidence / "adaptive/allocation/adaptive_allocation.json").read_text())
    assert [(r["bits"], r["qjl_keys"]) for r in allocation["layers"]] == [(r["bits"], r["qjl_keys"]) for r in saved_allocation["layers"]]
    assert all(r["bits"] == 3 and r["qjl_keys"] for r in allocation["layers"])
    assert allocation["summary"]["selected_measured_bytes_per_vector"] <= allocation["budget"]["measured_reference_bytes_per_vector"]
    for row in runs["turboquant_adaptive"]:
        assert len(row["turboquant_stats"]["layer_settings"]) == 32
        assert all(r["bits"] == 3 and r["qjl_keys"] for r in row["turboquant_stats"]["layer_settings"])
    heldout = {name: [r for r in rows if r["record_id"] not in excluded] for name, rows in runs.items()}
    assert all(len(rows) == 768 and set(Counter(r["task"] for r in rows).values()) == {48} for rows in heldout.values())
    adaptive = {"calibration_pool": 32, "used_calibration_samples": 16, "layer_reports": 512,
        "heldout_records": 768, "allocation": saved_allocation["summary"],
        "heldout_scores": {name: macro(rows) for name, rows in heldout.items()},
        "vs_full_cache": paired(heldout["turboquant_adaptive"], heldout["full_cache"], args.bootstrap_samples),
        "vs_uniform_3bit": paired(heldout["turboquant_adaptive"], heldout["turboquant_3bit"], args.bootstrap_samples)}
    qjl_pairs = {bits: paired(runs[f"turboquant_mse_{bits}bit"], runs[f"turboquant_{bits}bit"], args.bootstrap_samples)
                 for bits in ("2.5", "3", "3.5", "4")}
    report = {"source_commit": SOURCE_COMMIT, "dataset_sha256": sha256(args.data),
        "fixed_settings": 9, "adaptive_settings": 1, "turboquant_predictions": 8000,
        "rescoring_mismatches": 0, "oom": 0, "blank_predictions": 0,
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": 20260,
        "summary_rounding": "shared summary averages rounded task means; raw_macro_score averages unrounded record scores",
        "runs": summaries, "adaptive": adaptive, "qjl_off_minus_on": qjl_pairs}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    # Hash all archived evidence; plots and generated analysis are reproducible outputs.
    for path in evidence.rglob("*"):
        if path.is_file() and out not in path.parents:
            checksums[str(path.relative_to(REPO))] = sha256(path)
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    table = ["# Verified RTX 5090 results", "", "Nine fixed settings and one adaptive experiment; 800 records each. Full-cache baseline is the team's existing run.", "",
        "| Setting | Score /100 | Computed cache % of full | Measured peak GPU GB |", "|---|---:|---:|---:|"]
    for row in summaries:
        table.append(f"| {row['method']} | {row['score']:.2f} | {row['computed_cache_percent']:.2f} | {row['peak_gpu_gb']:.2f} |")
    table += ["", "Cache percentages describe packed representation accounting, not physical VRAM savings.", "",
        "## Adaptive held-out comparison", "", "All 32 calibration-pool IDs are excluded from every method; 768 records remain (48 per task).", ""]
    for name in ("full_cache", "turboquant_3bit", "turboquant_adaptive"):
        table.append(f"- {name}: {adaptive['heldout_scores'][name]:.4f}.")
    table += ["", "The allocator selected 3-bit QJL in every layer: zero calibration-objective improvement. The small score difference from the earlier uniform run is a repeated-run difference, not an adaptive allocation gain.", "",
        "Paired 95% bootstrap intervals and all comparisons: [results.json](results.json)."]
    (out / "RESULTS.md").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    for qjl, color, marker in [(True, "#b64235", "o"), (False, "#177b89", "s")]:
        group = sorted([r for r in summaries if r["method"] != "full_cache" and "adaptive" not in r["method"]
                        and ("mse" not in r["method"]) == qjl], key=lambda r: r["computed_cache_percent"])
        ax.plot([r["computed_cache_percent"] for r in group], [r["raw_macro_score"] for r in group], marker=marker, color=color,
                label="QJL keys" if qjl else "MSE keys (QJL off)")
        for r in group:
            bits = r["method"].split("_")[-1].replace("bit", "")
            ax.annotate(bits + "b", (r["computed_cache_percent"], r["raw_macro_score"]), xytext=(3, 7 if qjl else -13), textcoords="offset points", fontsize=9)
    ax.axhline(macro(baseline), color="#4d4d4d", linestyle="--", label="Full-cache baseline")
    ax.set(xlabel="Computed packed-cache size (% of full cache)", ylabel="LongBench development score /100",
           title="TurboQuant quality versus computed cache size · 800 records")
    ax.legend(loc="lower right"); ax.grid(alpha=.15)
    fig.text(.12, -.01, "Decoded BF16 cache remains resident. These percentages are not measured VRAM savings.", fontsize=9)
    save_plot(fig, figures, "quality_vs_computed_cache")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (bits, pair) in enumerate(qjl_pairs.items()):
        d = pair["difference_pp"]; lo, hi = pair["ci95_pp"]
        ax.errorbar(d, i, xerr=[[d-lo], [hi-d]], fmt="o", color="#177b89", capsize=4)
    ax.set_yticks(range(4), [f"{b}-bit" for b in qjl_pairs]); ax.axvline(0, linestyle="--", color="gray")
    ax.set(xlabel="QJL-off minus QJL-on score (percentage points; paired 95% CI)", title="Key-quantizer ablation · 800 paired records")
    fig.text(.12, -.01, "Matched nominal rates; norm metadata makes total packed budgets slightly different.", fontsize=9)
    save_plot(fig, figures, "qjl_ablation")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["full_cache", "turboquant_3bit", "turboquant_adaptive"]
    values = [adaptive["heldout_scores"][n] for n in names]
    ax.bar(["Full cache", "Uniform 3-bit QJL", "Adaptive (all 3-bit)"], values, color=["#666666", "#b64235", "#177b89"])
    for i, value in enumerate(values): ax.text(i, value+.5, f"{value:.2f}", ha="center")
    ax.set(ylim=(0, 55), ylabel="Held-out score /100", title="Adaptive outcome · 768 records after calibration exclusions")
    fig.text(.12, -.01, "Same selected configuration as uniform 3-bit; no adaptive improvement established.", fontsize=9)
    save_plot(fig, figures, "adaptive_heldout")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tokens = [r["prompt_tokens_after"] for r in data]
    ax.hist(tokens, bins=20, color="#177b89", edgecolor="white")
    ax.set(xlabel="Prepared prompt tokens (up to 32,000)", ylabel="Records", title="Frozen development data · 800 records / 16 tasks / 50 per task")
    save_plot(fig, figures, "prepared_prompt_lengths")
    fig, ax = plt.subplots(figsize=(12, 3.5)); ax.set(xlim=(0, 12), ylim=(0, 3)); ax.axis("off")
    boxes = [(0.1, "Frozen team JSONL\n800 IDs + SHA256"), (3.1, "Shared prompts / BF16\nDocument | question"),
             (6.1, "Llama + TurboQuant\nDocument cache only"), (9.1, "JSONL artifact store\nPredictions + manifests")]
    for x, label in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.2), 2.7, 1.1, boxstyle="round,pad=.04", facecolor="#e9f4f4", edgecolor="#177b89"))
        ax.text(x+1.35, 1.75, label, ha="center", va="center", fontsize=10)
        if x < 9: ax.annotate("", xy=(x+3, 1.75), xytext=(x+2.75, 1.75), arrowprops={"arrowstyle": "->"})
    ax.text(6, .65, "Validation: IDs / hashes / OOM / canonical re-scoring → paired analysis → versioned figures and tables", ha="center", fontsize=10)
    ax.text(6, .15, "Adaptive branch: 32 calibration-pool IDs → 16 real-query samples → layer allocation → 768 held-out records", ha="center", fontsize=10)
    save_plot(fig, figures, "evaluation_pipeline")
    print(f"PASS: 8,000 TurboQuant predictions + 800 baseline predictions; 512 layer reports. Outputs: {out}")


if __name__ == "__main__":
    main()
