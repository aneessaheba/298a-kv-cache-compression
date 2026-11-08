"""Evaluate TurboQuant on saved real-model key, value, and optional query tensors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mse import TurboQuantMSE
from .packing import pack_mse_codes, pack_product_codes
from .product import TurboQuantProd


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, torch.Tensor):
            return {"tensor": loaded}
        if not isinstance(loaded, dict):
            raise ValueError("PyTorch file must contain a tensor dictionary")
        return {str(key): value for key, value in loaded.items() if isinstance(value, torch.Tensor)}
    if suffix == ".npz":
        with np.load(path) as loaded:
            return {key: torch.from_numpy(loaded[key]) for key in loaded.files}
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    raise ValueError("supported tensor formats are .pt, .pth, .npz, and .safetensors")


def evaluate_real_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor | None = None,
    *,
    bit_widths: tuple[int, ...] = (2, 3, 4),
    seed: int = 42,
) -> dict[str, Any]:
    if keys.shape != values.shape or keys.ndim < 2:
        raise ValueError("keys and values must share a shape with vectors on the final axis")
    dimension = keys.shape[-1]
    keys = keys.to(torch.float64)
    values = values.to(torch.float64)
    if queries is None:
        generator = torch.Generator().manual_seed(seed + 99)
        queries = torch.randn(keys.shape, generator=generator, dtype=torch.float64)
        queries = torch.nn.functional.normalize(queries, dim=-1)
    elif queries.shape != keys.shape:
        raise ValueError("queries must have the same shape as keys")
    else:
        queries = queries.to(torch.float64)

    vector_count = keys.numel() // dimension
    rows = []
    for bits in bit_widths:
        value_quantizer = TurboQuantMSE(dimension, bits, seed=seed + bits)
        value_codes = value_quantizer.quantize(values)
        value_reconstructed = value_quantizer.dequantize(value_codes)
        value_packed = pack_mse_codes(value_codes, bits)

        for qjl in (False, True):
            if qjl and bits < 2:
                continue
            key_quantizer: TurboQuantMSE | TurboQuantProd
            if qjl:
                key_quantizer = TurboQuantProd(dimension, bits, seed=seed + 100 + bits)
                key_codes = key_quantizer.quantize(keys)
                key_packed = pack_product_codes(key_codes, bits - 1)
                method = "turboquant_product"
            else:
                key_quantizer = TurboQuantMSE(dimension, bits, seed=seed + 100 + bits)
                key_codes = key_quantizer.quantize(keys)
                key_packed = pack_mse_codes(key_codes, bits)
                method = "polarquant_mse"
            key_reconstructed = key_quantizer.dequantize(key_codes)
            signed_ip_error = (
                (queries * key_reconstructed).sum(-1) - (queries * keys).sum(-1)
            )
            compressed_bytes = key_packed.nbytes + value_packed.nbytes
            full_fp16_bytes = 2 * keys.numel() * 2
            rows.append(
                {
                    "method": method,
                    "bits": bits,
                    "dimension": dimension,
                    "vectors": vector_count,
                    "key_normalized_mse": (
                        (keys - key_reconstructed).square().sum(-1)
                        / keys.square().sum(-1).clamp_min(torch.finfo(torch.float64).tiny)
                    ).mean().item(),
                    "value_normalized_mse": (
                        (values - value_reconstructed).square().sum(-1)
                        / values.square().sum(-1).clamp_min(torch.finfo(torch.float64).tiny)
                    ).mean().item(),
                    "mean_signed_inner_product_error": signed_ip_error.mean().item(),
                    "inner_product_error_mse": signed_ip_error.square().mean().item(),
                    "packed_bytes": compressed_bytes,
                    "fp16_bytes": full_fp16_bytes,
                    "storage_ratio": compressed_bytes / full_fp16_bytes,
                    "compression_factor": full_fp16_bytes / compressed_bytes,
                }
            )
    return {"source_shape": list(keys.shape), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_file", type=Path)
    parser.add_argument("--key-name", default="keys")
    parser.add_argument("--value-name", default="values")
    parser.add_argument("--query-name")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_kv"))
    args = parser.parse_args()
    tensors = _load_tensors(args.tensor_file)
    missing = [name for name in (args.key_name, args.value_name) if name not in tensors]
    if missing:
        raise SystemExit(f"missing tensors {missing}; available names: {sorted(tensors)}")
    queries = tensors.get(args.query_name) if args.query_name else None
    report = evaluate_real_kv(
        tensors[args.key_name],
        tensors[args.value_name],
        queries,
        bit_widths=tuple(args.bits),
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "real_kv_report.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = sorted({key for row in report["rows"] for key in row})
    with (args.output_dir / "real_kv_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps({"source_shape": report["source_shape"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
