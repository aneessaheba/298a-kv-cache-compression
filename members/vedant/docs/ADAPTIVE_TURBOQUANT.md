# Adaptive layer allocation — experimental extension

The allocator chooses 2/3/4-bit layer configurations under the packed storage budget of uniform 3-bit QJL. It is a project extension, not a new published algorithm claim. Uniform mixed-channel rates remain the primary experiments.

The frozen task-stratified split has 32 calibration records, a 32-record context subset drawn from the evaluation pool, and 768 evaluation records. The default calibration takes 16 evenly spaced records from the 32 pool; all 32 remain excluded from adaptive comparisons. The split seed is 20,260. Generate the splits first:

```bash
cd members/vedant
turboquant-split --input ../../data/processed_longbench/processed_longbench_dev.jsonl --output-dir configs/splits
```

The command regenerates calibration/context JSONL using the frozen split seed. Then:

```bash
bash scripts/run_adaptive_calibration.sh configs/splits/adaptive_calibration.jsonl \
 meta-llama/Llama-3.1-8B-Instruct results/calibration 16
```

Extraction uses BF16 and shared kvpress prompt formatting. Hooks capture actual post-RoPE Llama query heads, preserving GQA query multiplicity. Calibration retains at most 4096 head/tail tokens per prompt instead of only the first 1024 raw tokens. Tensor loading is shared across all layers of each saved sample. Per-layer rotation seeds, FP32 quantization of BF16 inputs, rounded FP16 norm metadata and BF16 reconstruction match the shared press. This bounded window changes positions and cannot characterize the entire 32K sequence; report that limitation.

Both objective terms are dimensionless: `key_weight * key_normalized_mse + value_weight * value_normalized_mse`. Key and value squared reconstruction errors are divided by their own vector energy. Real-query inner-product error (normalized and absolute) is also recorded as a diagnostic, but is not mixed with value relative error in the objective. Position-aligned QK error is a proxy, not the full causal attention distortion. Synthetic-query diagnostics are explicitly labelled if real queries are absent; the all-layer calibration CLI requires saved real queries.

The dynamic program selects one candidate per layer, writes measured candidate sizes and the chosen allocation, and defaults to QJL enabled in every layer. QJL-off allocation is a separate ablation. Norm fields are included in representation bytes; the reference implementation's decoded BF16 and dense quantizer state do not count as compressed-representation savings.

Run the resulting `adaptive_allocation.json` with the shared harness's `--press turboquant_adaptive --turboquant-layer-config FILE`. Compare complete shared files with `scripts/compare_team_runs.py --adaptive` to exclude exactly the frozen calibration IDs from every run, including baseline and uniform controls. Optional standalone context scripts are diagnostic studies with their own scoring policy; they are not canonical team comparison results.

Success requires observed budget compliance and improved paired held-out quality. If the paired bootstrap interval includes zero, report the evidence as inconclusive; it does not prove equivalence. GPU execution is complete. Sixteen samples (one per task) from the 32-record pool produced 512 layer reports. The allocator selected 3-bit QJL in every layer, giving zero calibration-objective reduction at the uniform 3-bit packed budget. The 768-record held-out scores were 44.4970 (adaptive), 44.4096 (earlier uniform 3-bit), and 47.4485 (full cache). The adaptive-minus-uniform paired 95% interval is approximately [-0.4754, 0.6506] percentage points. This is a repeat of the same selected configuration, not evidence of adaptive benefit. See [verified analysis](../results/sem1/analysis/RESULTS.md).
