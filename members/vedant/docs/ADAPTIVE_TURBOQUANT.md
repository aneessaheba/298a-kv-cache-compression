# Adaptive TurboQuant Extension

## Contribution

The extension assigns a 2-, 3-, or 4-bit TurboQuant configuration to each
transformer layer under the measured storage budget of uniform 3-bit
TurboQuant. It is a project contribution built on the paper's quantizer; it is
not presented as a claim of inventing a new published algorithm.

The default, defensible experiment keeps QJL enabled in every layer so the
paper's unbiased inner-product construction remains intact. An optional
QJL-on/off allocator exists only as a clearly labeled exploratory ablation.

## Leakage control

The supplied 800-record development set has a deterministic task-stratified
split under `configs/splits/`:

- 32 calibration records: two per LongBench task;
- 32 context-scaling records: two records per task, drawn from the evaluation pool; and
- 768 held-out evaluation records, excluding calibration records.

The split seed is 20,260. The comparison command excludes the frozen
calibration IDs when it calculates final quality and bootstrap confidence
intervals. The predeclared primary calibration run uses 16 evenly spaced
records from the 32-record pool, which covers each of the 16 tasks once. All 32
pool records remain excluded from the final comparison as a conservative
leakage safeguard.

## Allocation method

For every layer and candidate bit width, calibration records provide:

- key inner-product squared error;
- value normalized reconstruction error; and
- physically packed bytes per vector, including stored norm metadata.

The default objective is the sum of key inner-product error and value
normalized error. Both weights are configurable. The allocator solves the
multiple-choice storage-constrained problem using dynamic programming and
writes the chosen layer configuration, every candidate measurement, a CSV
table, and a sensitivity heatmap.

The budget is defined by the measured uniform 3-bit QJL configuration rather
than nominal bit width alone. Final storage is measured again during model
generation; the allocation estimate is not substituted for the actual cache
measurement.

## Evaluation

The primary comparison is:

1. uniform 3-bit TurboQuant;
2. Adaptive TurboQuant under the uniform 3-bit storage budget; and
3. full precision, plus the previously defined 2/3/4-bit and QJL-off runs.

All methods use identical record IDs, prompts, decoding settings, model
revision, and scoring code. The report includes paired, task-stratified
bootstrap 95% confidence intervals for the macro LongBench score difference.

The extension also evaluates 4K, 8K, 16K, and 32K context caps on the frozen
32-record context subset. The generated plot shows LongBench quality and
measured total KV-cache MiB as context length increases.

## Success criterion

Adaptive TurboQuant succeeds if it stays within the measured uniform 3-bit
cache budget and improves the LongBench macro score. If its confidence interval
includes zero, it is reported as quality-equivalent rather than as a proven
improvement. A negative result is still useful evidence about layer
sensitivity and must not be hidden.
