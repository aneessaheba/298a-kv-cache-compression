"""Synthetic correctness gate for the Team 6 TurboQuant track."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from .mse import TurboQuantMSE
from .product import TurboQuantProd


PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.030, 4: 0.009}
PAPER_PRODUCT_SCALED = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}


def _unit_vectors(samples: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vectors = torch.randn(samples, dimension, generator=generator, dtype=torch.float64)
    return torch.nn.functional.normalize(vectors, dim=-1)


def validate(
    *,
    dimension: int = 128,
    samples: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    x = _unit_vectors(samples, dimension, seed + 10)
    y = _unit_vectors(samples, dimension, seed + 20)
    rows: list[dict[str, Any]] = []

    for bits in range(1, 5):
        quantizer = TurboQuantMSE(dimension, bits, seed=seed + 100 * bits)
        reconstructed = quantizer.roundtrip(x)
        per_item_mse = (x - reconstructed).square().sum(dim=-1)
        observed = per_item_mse.mean().item()
        reference = PAPER_MSE[bits]
        rows.append(
            {
                "variant": "mse",
                "bits": bits,
                "observed_distortion": observed,
                "paper_reference": reference,
                "relative_error": abs(observed - reference) / reference,
                "mean_inner_product_error": (
                    (y * reconstructed).sum(-1) - (y * x).sum(-1)
                ).mean().item(),
            }
        )

    for bits in range(2, 5):
        quantizer = TurboQuantProd(dimension, bits, seed=seed + 1000 * bits)
        reconstructed = quantizer.roundtrip(x)
        error = (y * reconstructed).sum(-1) - (y * x).sum(-1)
        observed = error.square().mean().item()
        reference = PAPER_PRODUCT_SCALED[bits] / dimension
        standard_error = error.std(unbiased=True).item() / math.sqrt(samples)
        bias = error.mean().item()
        rows.append(
            {
                "variant": "product",
                "bits": bits,
                "observed_distortion": observed,
                "paper_reference": reference,
                "relative_error": abs(observed - reference) / reference,
                "mean_inner_product_error": bias,
                "bias_standard_error": standard_error,
                "bias_z_score": bias / standard_error if standard_error else 0.0,
            }
        )

    return {
        "dimension": dimension,
        "samples": samples,
        "seed": seed,
        "paper": "arXiv:2504.19874",
        "rows": rows,
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in report["rows"] for key in row})
    with (output_dir / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])


def _plot(report: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install the 'plots' extra to use --plot") from exc

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, variant, title in [
        (axes[0], "mse", "MSE reconstruction distortion"),
        (axes[1], "product", "Inner-product squared error"),
    ]:
        rows = [row for row in report["rows"] if row["variant"] == variant]
        bits = [row["bits"] for row in rows]
        axis.semilogy(bits, [row["observed_distortion"] for row in rows], "o-", label="observed")
        axis.semilogy(bits, [row["paper_reference"] for row in rows], "s--", label="paper reference")
        axis.set_xlabel("bits per coordinate")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation.png", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    report = validate(dimension=args.dimension, samples=args.samples, seed=args.seed)
    _write_report(report, args.output_dir)
    if args.plot:
        _plot(report, args.output_dir)

    for row in report["rows"]:
        print(
            f"{row['variant']:>7} b={row['bits']}: "
            f"observed={row['observed_distortion']:.6g}, "
            f"paper={row['paper_reference']:.6g}, "
            f"relative_error={row['relative_error']:.1%}, "
            f"bias={row['mean_inner_product_error']:+.3e}"
        )


if __name__ == "__main__":
    main()
