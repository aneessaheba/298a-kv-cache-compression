"""Final-workbook G3 correctness protocol for the TurboQuant implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .codebook import build_sphere_codebook
from .mse import TurboQuantMSE
from .product import TurboQuantProd
from .reference import scalar_mse_float64, scalar_quantize_float64
from .rotation import random_orthogonal_matrix


PAPER_PRODUCT_SCALED = {2: 0.56, 4: 0.047}


def _unit_vectors(samples: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(samples, dimension, generator=generator, dtype=torch.float64)
    return torch.nn.functional.normalize(x, dim=-1)


def _fixture_check(dimensions: Iterable[int], bits: Iterable[int]) -> dict[str, Any]:
    values = np.array(
        [-1.0, -0.91, -0.31, -0.125, -1e-12, 0.0, 1e-12, 0.125, 0.31, 0.91, 1.0],
        dtype=np.float64,
    )
    max_index_difference = 0
    max_value_difference = 0.0
    rows = []
    for dimension in dimensions:
        for bit_width in bits:
            codebook = build_sphere_codebook(dimension, bit_width)
            ref_idx, ref_values = scalar_quantize_float64(values, codebook.centroids.numpy())
            torch_values = torch.from_numpy(values)
            torch_idx = torch.bucketize(torch_values, codebook.boundaries).numpy()
            torch_reconstructed = codebook.centroids[torch.from_numpy(torch_idx)].numpy()
            index_difference = int(np.max(np.abs(ref_idx - torch_idx)))
            value_difference = float(np.max(np.abs(ref_values - torch_reconstructed)))
            max_index_difference = max(max_index_difference, index_difference)
            max_value_difference = max(max_value_difference, value_difference)
            rows.append(
                {
                    "dimension": dimension,
                    "bits": bit_width,
                    "index_difference": index_difference,
                    "max_abs_difference": value_difference,
                }
            )
    return {
        "rows": rows,
        "max_index_difference": max_index_difference,
        "max_abs_difference": max_value_difference,
        "threshold": 1e-6,
        "passed": max_index_difference == 0 and max_value_difference <= 1e-6,
    }


def _mean_ci95(seed_means: list[float]) -> tuple[float, float, float]:
    values = torch.tensor(seed_means, dtype=torch.float64)
    mean = values.mean().item()
    if values.numel() < 2:
        return mean, float("nan"), float("nan")
    half_width = 1.96 * values.std(unbiased=True).item() / math.sqrt(values.numel())
    return mean, mean - half_width, mean + half_width


def run_g3(
    *,
    dimensions: tuple[int, ...] = (128, 512),
    bits: tuple[int, ...] = (2, 4),
    seeds: int = 30,
    samples_per_seed: int = 1_000,
    base_seed: int = 17_029,
) -> dict[str, Any]:
    """Run the exact G3 protocol stated in the final Team 6 workbook."""

    if seeds < 1 or samples_per_seed < 1:
        raise ValueError("seeds and samples_per_seed must be positive")
    fixtures = _fixture_check(dimensions, bits)
    rotation_rows: list[dict[str, Any]] = []
    mse_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []

    for dimension in dimensions:
        codebooks = {b: build_sphere_codebook(dimension, b) for b in bits}
        product_codebooks = {b: build_sphere_codebook(dimension, b - 1) for b in bits}
        numpy_references = {
            b: dimension * scalar_mse_float64(codebooks[b].centroids.numpy(), dimension)
            for b in bits
        }
        rotation_errors: list[float] = []
        mse_seed_values: dict[int, list[float]] = {b: [] for b in bits}
        product_bias_seed_values: dict[int, list[float]] = {b: [] for b in bits}
        product_mse_seed_values: dict[int, list[float]] = {b: [] for b in bits}

        for seed_index in range(seeds):
            seed = base_seed + dimension * 10_000 + seed_index
            x = _unit_vectors(samples_per_seed, dimension, seed)
            y = _unit_vectors(samples_per_seed, dimension, seed + 1_000_000)
            rotation_seed = seed + 2_000_000
            projection_seed = seed + 3_000_000
            rotation = random_orthogonal_matrix(dimension, seed=rotation_seed)
            projection_generator = torch.Generator(device="cpu").manual_seed(projection_seed)
            projection = torch.randn(
                dimension,
                dimension,
                generator=projection_generator,
                dtype=torch.float64,
            )
            before = torch.linalg.vector_norm(x, dim=-1)
            after = torch.linalg.vector_norm(x @ rotation.T, dim=-1)
            rotation_errors.append(
                ((after - before).abs() / before.clamp_min(torch.finfo(before.dtype).tiny)).max().item()
            )

            for bit_width in bits:
                mse_quantizer = TurboQuantMSE(
                    dimension,
                    bit_width,
                    seed=rotation_seed,
                    codebook=codebooks[bit_width],
                    rotation=rotation,
                )
                mse_reconstructed = mse_quantizer.roundtrip(x)
                mse_seed_values[bit_width].append(
                    (x - mse_reconstructed).square().sum(dim=-1).mean().item()
                )

                product_quantizer = TurboQuantProd(
                    dimension,
                    bit_width,
                    seed=rotation_seed,
                    codebook=product_codebooks[bit_width],
                    rotation=rotation,
                    projection=projection,
                )
                product_reconstructed = product_quantizer.roundtrip(x)
                signed_error = (
                    (y * product_reconstructed).sum(dim=-1) - (y * x).sum(dim=-1)
                )
                product_bias_seed_values[bit_width].append(signed_error.mean().item())
                product_mse_seed_values[bit_width].append(signed_error.square().mean().item())

        max_rotation_error = max(rotation_errors)
        rotation_rows.append(
            {
                "dimension": dimension,
                "max_relative_norm_error": max_rotation_error,
                "threshold": 1e-5,
                "passed": max_rotation_error <= 1e-5,
            }
        )

        for bit_width in bits:
            observed = float(np.mean(mse_seed_values[bit_width]))
            reference = numpy_references[bit_width]
            relative_difference = abs(observed - reference) / reference
            mse_rows.append(
                {
                    "dimension": dimension,
                    "bits": bit_width,
                    "observed_distortion": observed,
                    "matched_numpy_reference": reference,
                    "relative_difference": relative_difference,
                    "threshold": 0.10,
                    "passed": relative_difference <= 0.10,
                }
            )

            bias, ci_low, ci_high = _mean_ci95(product_bias_seed_values[bit_width])
            product_mse = float(np.mean(product_mse_seed_values[bit_width]))
            paper_reference = PAPER_PRODUCT_SCALED[bit_width] / dimension
            product_rows.append(
                {
                    "dimension": dimension,
                    "bits": bit_width,
                    "mean_signed_inner_product_error": bias,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "equivalence_interval_low": -0.01,
                    "equivalence_interval_high": 0.01,
                    "observed_squared_error": product_mse,
                    "paper_approximation": paper_reference,
                    "passed": ci_low >= -0.01 and ci_high <= 0.01,
                }
            )

    overall = (
        fixtures["passed"]
        and all(row["passed"] for row in rotation_rows)
        and all(row["passed"] for row in mse_rows)
        and all(row["passed"] for row in product_rows)
    )
    return {
        "protocol": {
            "dimensions": list(dimensions),
            "bits": list(bits),
            "independent_seeds": seeds,
            "vectors_per_seed": samples_per_seed,
            "base_seed": base_seed,
            "confidence_interval_unit": "independent seed mean",
        },
        "fixtures": fixtures,
        "rotation": rotation_rows,
        "mse_distortion": mse_rows,
        "inner_product": product_rows,
        "overall_passed": overall,
    }


def _write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "g3_report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for section in ("rotation", "mse_distortion", "inner_product"):
        for row in report[section]:
            rows.append({"section": section, **row})
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "g3_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for dimension in report["protocol"]["dimensions"]:
        selected = [r for r in report["mse_distortion"] if r["dimension"] == dimension]
        axes[0].plot(
            [r["bits"] for r in selected],
            [r["observed_distortion"] for r in selected],
            "o-",
            label=f"observed d={dimension}",
        )
        axes[0].plot(
            [r["bits"] for r in selected],
            [r["matched_numpy_reference"] for r in selected],
            "x--",
            label=f"reference d={dimension}",
        )
        product = [r for r in report["inner_product"] if r["dimension"] == dimension]
        axes[1].errorbar(
            [r["bits"] for r in product],
            [r["mean_signed_inner_product_error"] for r in product],
            yerr=[
                [r["mean_signed_inner_product_error"] - r["ci95_low"] for r in product],
                [r["ci95_high"] - r["mean_signed_inner_product_error"] for r in product],
            ],
            marker="o",
            capsize=4,
            label=f"d={dimension}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bits per coordinate")
    axes[0].set_ylabel("normalized reconstruction error")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhspan(-0.01, 0.01, alpha=0.12, color="green", label="equivalence interval")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("total key bits")
    axes[1].set_ylabel("mean signed inner-product error (95% CI)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "g3_report.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--samples-per-seed", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, default=17_029)
    parser.add_argument("--output-dir", type=Path, default=Path("results/g3"))
    args = parser.parse_args()
    report = run_g3(
        dimensions=tuple(args.dimensions),
        bits=tuple(args.bits),
        seeds=args.seeds,
        samples_per_seed=args.samples_per_seed,
        base_seed=args.base_seed,
    )
    _write_outputs(report, args.output_dir)
    print(json.dumps({"overall_passed": report["overall_passed"], "protocol": report["protocol"]}, indent=2))
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
