"""Allocate per-layer TurboQuant settings under a measured storage budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayerCandidate:
    layer: int
    bits: int
    qjl_keys: bool
    objective: float
    bytes_per_vector: float
    key_error: float
    value_error: float

    @property
    def name(self) -> str:
        return f"{self.bits}bit_{'qjl' if self.qjl_keys else 'mse'}"


def _aggregate_reports(
    layer_reports: dict[int, list[dict[str, Any]]],
    *,
    key_weight: float,
    value_weight: float,
) -> dict[int, list[LayerCandidate]]:
    if key_weight < 0 or value_weight < 0 or key_weight + value_weight <= 0:
        raise ValueError("objective weights must be non-negative and not both zero")
    candidates: dict[int, list[LayerCandidate]] = {}
    for layer, reports in sorted(layer_reports.items()):
        grouped: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
        for report in reports:
            for row in report["rows"]:
                qjl = row["method"] == "turboquant_product"
                grouped[(int(row["bits"]), qjl)].append(row)
        layer_candidates = []
        for (bits, qjl), rows in sorted(grouped.items()):
            key_error = sum(float(row["inner_product_error_mse"]) for row in rows) / len(rows)
            value_error = sum(float(row["value_normalized_mse"]) for row in rows) / len(rows)
            bytes_per_vector = sum(
                float(row["packed_bytes"]) / float(row["vectors"]) for row in rows
            ) / len(rows)
            layer_candidates.append(
                LayerCandidate(
                    layer=layer,
                    bits=bits,
                    qjl_keys=qjl,
                    objective=key_weight * key_error + value_weight * value_error,
                    bytes_per_vector=bytes_per_vector,
                    key_error=key_error,
                    value_error=value_error,
                )
            )
        if not layer_candidates:
            raise ValueError(f"layer {layer} has no candidates")
        candidates[layer] = layer_candidates
    expected = list(range(len(candidates)))
    if sorted(candidates) != expected:
        raise ValueError(f"layer reports must be contiguous from 0; received {sorted(candidates)}")
    return candidates


def allocate_layers(
    candidates: dict[int, list[LayerCandidate]],
    *,
    reference_bits: int = 3,
    reference_qjl: bool = True,
    qjl_policy: str = "on",
) -> dict[str, Any]:
    """Solve the multiple-choice storage-constrained allocation exactly.

    Storage is rounded upward to whole bytes per vector, making the selected
    allocation conservative relative to the measured reference budget.
    """

    if qjl_policy not in {"on", "off", "adaptive"}:
        raise ValueError("qjl_policy must be 'on', 'off', or 'adaptive'")
    filtered: dict[int, list[LayerCandidate]] = {}
    for layer, rows in sorted(candidates.items()):
        allowed = [
            row
            for row in rows
            if qjl_policy == "adaptive" or row.qjl_keys == (qjl_policy == "on")
        ]
        if not allowed:
            raise ValueError(f"layer {layer} has no candidates for QJL policy {qjl_policy}")
        filtered[layer] = allowed

    references = []
    for layer, rows in sorted(filtered.items()):
        matches = [row for row in rows if row.bits == reference_bits and row.qjl_keys == reference_qjl]
        if len(matches) != 1:
            raise ValueError(
                f"layer {layer} needs one {reference_bits}-bit "
                f"{'QJL' if reference_qjl else 'MSE'} reference candidate"
            )
        references.append(matches[0])
    budget_bytes = sum(math.ceil(row.bytes_per_vector) for row in references)

    # cost -> (objective, tuple of chosen candidates)
    states: dict[int, tuple[float, tuple[LayerCandidate, ...]]] = {0: (0.0, ())}
    for layer, rows in sorted(filtered.items()):
        next_states: dict[int, tuple[float, tuple[LayerCandidate, ...]]] = {}
        for used, (loss, chosen) in states.items():
            for row in sorted(rows, key=lambda item: (item.bits, item.qjl_keys)):
                cost = used + math.ceil(row.bytes_per_vector)
                if cost > budget_bytes:
                    continue
                proposal = (loss + row.objective, chosen + (row,))
                current = next_states.get(cost)
                if current is None or proposal[0] < current[0] - 1e-15:
                    next_states[cost] = proposal
        if not next_states:
            raise ValueError(f"storage budget is infeasible after layer {layer}")
        states = next_states

    selected_cost, (selected_loss, selected) = min(
        states.items(), key=lambda item: (item[1][0], item[0])
    )
    reference_loss = sum(row.objective for row in references)
    selected_bytes = sum(row.bytes_per_vector for row in selected)
    reference_bytes = sum(row.bytes_per_vector for row in references)
    layers = [
        {
            "layer": row.layer,
            "bits": row.bits,
            "qjl_keys": row.qjl_keys,
            "candidate": row.name,
            "objective": row.objective,
            "bytes_per_vector": row.bytes_per_vector,
            "key_error": row.key_error,
            "value_error": row.value_error,
        }
        for row in selected
    ]
    return {
        "schema_version": 1,
        "method": "adaptive_turboquant",
        "qjl_policy": qjl_policy,
        "budget": {
            "reference_bits": reference_bits,
            "reference_qjl_keys": reference_qjl,
            "rounded_bytes_per_vector": budget_bytes,
            "measured_reference_bytes_per_vector": reference_bytes,
        },
        "summary": {
            "layers": len(selected),
            "average_nominal_bits": sum(row.bits for row in selected) / len(selected),
            "qjl_layers": sum(row.qjl_keys for row in selected),
            "selected_rounded_bytes_per_vector": selected_cost,
            "selected_measured_bytes_per_vector": selected_bytes,
            "reference_objective": reference_loss,
            "selected_objective": selected_loss,
            "objective_reduction": reference_loss - selected_loss,
            "within_reference_storage": selected_bytes <= reference_bytes + len(selected),
        },
        "layers": layers,
        "candidates": [asdict(row) | {"name": row.name} for rows in filtered.values() for row in rows],
    }


def _write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adaptive_allocation.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "layer", "bits", "qjl_keys", "candidate", "objective", "bytes_per_vector",
        "key_error", "value_error",
    ]
    with (output_dir / "adaptive_allocation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["layers"])

    import matplotlib.pyplot as plt
    import numpy as np

    names = sorted({row["name"] for row in report["candidates"]})
    layers = report["summary"]["layers"]
    matrix = np.full((layers, len(names)), np.nan)
    for row in report["candidates"]:
        matrix[row["layer"], names.index(row["name"])] = row["objective"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=(2.2, 1))
    image = axes[0].imshow(np.log10(np.maximum(matrix, 1e-16)), aspect="auto", cmap="viridis_r")
    axes[0].set_xticks(range(len(names)), names, rotation=35, ha="right")
    axes[0].set_ylabel("transformer layer")
    axes[0].set_title("Layer sensitivity: log10 calibration objective (lower is better)")
    figure.colorbar(image, ax=axes[0], label="log10 objective")
    selected = report["layers"]
    x = [row["layer"] for row in selected]
    colors = ["#7b2cbf" if row["qjl_keys"] else "#2a9d8f" for row in selected]
    axes[1].bar(x, [row["bits"] for row in selected], color=colors)
    axes[1].set_yticks([2, 3, 4])
    axes[1].set_xlabel("transformer layer")
    axes[1].set_ylabel("selected bits")
    axes[1].set_title("Adaptive allocation (purple=QJL, green=MSE keys)")
    figure.tight_layout()
    figure.savefig(output_dir / "adaptive_sensitivity.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True,
        help="LAYER=real_kv_report.json; repeat a layer to average calibration prompts",
    )
    parser.add_argument("--reference-bits", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--reference-qjl", choices=["on", "off"], default="on")
    parser.add_argument(
        "--qjl-policy",
        choices=["on", "off", "adaptive"],
        default="on",
        help="keep QJL on by default; adaptive QJL selection is an exploratory ablation",
    )
    parser.add_argument("--key-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive"))
    args = parser.parse_args()
    reports: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for specification in args.report:
        if "=" not in specification:
            raise SystemExit("each --report must be LAYER=path.json")
        layer_text, path_text = specification.split("=", 1)
        reports[int(layer_text)].append(json.loads(Path(path_text).read_text(encoding="utf-8")))
    candidates = _aggregate_reports(
        reports, key_weight=args.key_weight, value_weight=args.value_weight
    )
    allocation = allocate_layers(
        candidates,
        reference_bits=args.reference_bits,
        reference_qjl=args.reference_qjl == "on",
        qjl_policy=args.qjl_policy,
    )
    _write_outputs(allocation, args.output_dir)
    print(json.dumps(allocation["summary"], indent=2))


if __name__ == "__main__":
    main()
