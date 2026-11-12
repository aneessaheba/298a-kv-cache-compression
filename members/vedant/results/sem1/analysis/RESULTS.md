# Verified RTX 5090 results

Nine fixed settings and one adaptive experiment; 800 records each. Full-cache baseline is the team's existing run.

| Setting | Score /100 | Computed cache % of full | Measured peak GPU GB |
|---|---:|---:|---:|
| full_cache | 47.72 | 100.00 | 24.09 |
| turboquant_2bit | 14.94 | 15.04 | 24.09 |
| turboquant_2.5bit | 39.89 | 19.29 | 24.09 |
| turboquant_3bit | 44.48 | 21.21 | 24.09 |
| turboquant_3.5bit | 47.25 | 25.46 | 24.09 |
| turboquant_4bit | 47.67 | 27.38 | 24.09 |
| turboquant_mse_2.5bit | 46.44 | 18.52 | 24.09 |
| turboquant_mse_3bit | 47.90 | 20.82 | 24.09 |
| turboquant_mse_3.5bit | 47.84 | 24.69 | 24.09 |
| turboquant_mse_4bit | 47.56 | 26.99 | 24.09 |
| turboquant_adaptive | 44.57 | 21.21 | 24.09 |

Cache percentages describe packed representation accounting, not physical VRAM savings.

## Adaptive held-out comparison

All 32 calibration-pool IDs are excluded from every method; 768 records remain (48 per task).

- full_cache: 47.4485.
- turboquant_3bit: 44.4096.
- turboquant_adaptive: 44.4970.

The allocator selected 3-bit QJL in every layer: zero calibration-objective improvement. The small score difference from the earlier uniform run is a repeated-run difference, not an adaptive allocation gain.

Paired 95% bootstrap intervals and all comparisons: [results.json](results.json).
