# TurboQuant Implementation and Validation — Vedant Workstream

## Objective and scope

This workstream implements and validates TurboQuant as the advanced KV-cache
quantization component of the team project. It covers the paper's two
mathematical variants: TurboQuant-MSE, which applies a Haar-random orthogonal
rotation followed by a scalar quantizer optimized for the exact unit-sphere
coordinate density, and
TurboQuant-Prod, which adds a one-bit QJL sign sketch for inner-product
estimation while using the remaining bits for the MSE code. The implementation
is evaluated independently before it is connected to eviction and serving.

The reference path prioritizes correctness and reproducibility. It physically
packs the quantized codes so persistent-cache storage can be measured exactly.
It dequantizes cached tensors before calling ordinary attention, so its latency
and peak-memory measurements must not be presented as those of a fused
production kernel. A fused kernel remains stretch work.

## Method

For each head vector, a deterministic Haar-random orthogonal transform spreads
energy across coordinates while preserving the Euclidean norm. It is generated
by the Gaussian QR construction stated in the paper. TurboQuant-MSE
normalizes the rotated vector, assigns each coordinate to a Lloyd--Max
codeword designed for the exact dimension-dependent unit-sphere coordinate
density, stores the packed code
indices and norm, and reverses the operation during decoding.

TurboQuant-Prod uses a `(b-1)`-bit MSE code and a one-bit sign projection. It
stores the residual norm required by the paper's estimator. The implementation
also supports a QJL-off ablation, which uses all `b` bits for the MSE key and
therefore separates the sign sketch from the common rotation and scalar
quantization pipeline. Values use the MSE path.

The Hugging Face cache adapter appends physically packed chunks per layer and
reports code, norm, and metadata bytes separately. The model runner uses
greedy decoding, records model and tokenizer revisions, GPU type, timing, bit
width, method, and exact cache bytes, and resumes by stable record ID.

## Numerical gate protocol

The final workbook's G3 gate was implemented literally:

1. deterministic scalar fixtures are compared with an independent NumPy
   float64 implementation with tolerance `1e-6`;
2. rotation norm preservation must have relative error at most `1e-5`;
3. normalized isotropic Gaussian inputs are tested at dimensions 128 and 512;
4. every dimension/bit setting uses 30 independent seeds with 1,000 vectors
   per seed;
5. two- and four-bit MSE distortion must be within 10% of an independent
   numerical unit-sphere reference; and
6. the seed-level 95% confidence interval of mean signed product error must lie
   entirely inside `[-0.01, 0.01]`.

Separate random streams are used for data vectors, queries, rotations, and QJL
projections so the experiment cannot correlate an input with its transform.

## G3 results

All gate conditions passed.

| Dimension | Bits | Observed MSE distortion | Numerical reference | Relative difference |
|---:|---:|---:|---:|---:|
| 128 | 2 | 0.11578798 | 0.11600007 | 0.183% |
| 128 | 4 | 0.00928287 | 0.00931482 | 0.343% |
| 512 | 2 | 0.11710661 | 0.11711072 | 0.0035% |
| 512 | 4 | 0.00945831 | 0.00945438 | 0.0416% |

| Dimension | Bits | Mean signed product error | 95% confidence interval | Product MSE |
|---:|---:|---:|---:|---:|
| 128 | 2 | -0.0000519 | [-0.0007291, 0.0006253] | 0.0044543 |
| 128 | 4 | -0.0001298 | [-0.0003737, 0.0001141] | 0.0004123 |
| 512 | 2 | -0.0001008 | [-0.0005417, 0.0003402] | 0.0011158 |
| 512 | 4 | -0.0000322 | [-0.0001571, 0.0000926] | 0.0001061 |

The largest observed rotation norm error was `8.882e-16`. Independent scalar
fixtures differed by `0.0`. Machine-readable results and the figure are stored
under `results/g3/`.

## Cache-size projection

For Llama 3.1 8B, the calculation uses 32 layers, 8 KV heads, head dimension
128, packed codes, and FP16 norm metadata. An FP16 key-plus-value cache requires
131,072 bytes per token. The QJL-key/MSE-value path is projected to require
17,920, 26,112, and 34,304 bytes per token at 2, 3, and 4 bits respectively,
corresponding to 7.31x, 5.02x, and 3.82x persistent-cache compression. The
QJL-off MSE-key/MSE-value ablation is slightly smaller because it has no
residual-norm metadata.

These figures are exact for the repository's packed representation but exclude
one-time transform definitions and Python container overhead. They describe
persistent stored tensors, not peak CUDA allocation in the reference path.

## Model-level evaluation plan

The RTX 5090 experiment matrix uses the processed LongBench development split,
greedy decoding, stable record IDs, and the official task-specific maximum
generation lengths. It produces full precision, TurboQuant at 2/3/4 bits,
QJL-off at 3/4 bits, real-KV validation for early/middle/late layers, and paired
comparisons with KIVI files scored by the same official evaluator.

The comparison tool rejects unscored or unmatched records. The official KIVI
repository exposes 2- and 4-bit cache settings, so no KIVI 3-bit measurement
will be invented. The 3-bit TurboQuant point is reported as part of the
continuous rate-quality curve and compared with the available neighboring
KIVI settings.

## Original extension: Adaptive TurboQuant

The project adds a storage-constrained layerwise allocation experiment. A
task-stratified pool of 32 records is frozen for calibration and excluded from
the 768-record held-out comparison. The predeclared primary run uses 16 evenly
spaced records from that pool, covering every LongBench task once. For every
transformer layer, the allocator measures 2-, 3-, and 4-bit candidates and
solves a multiple-choice optimization problem under the measured storage
budget of uniform 3-bit TurboQuant. QJL is kept enabled in the primary
experiment to preserve the paper's unbiased inner-product construction;
adaptive QJL selection is reported only as an exploratory ablation.

The final comparison reports measured bytes, LongBench macro score, and a
paired task-stratified bootstrap 95% confidence interval. A separate frozen
32-record subset measures quality and total cache growth at 4K, 8K, 16K, and
32K context caps. The allocation, heatmap, and context-length results will be
inserted after the RTX 5090 execution; no model-level result is inferred from
the local tiny-model smoke test.

## Limitations and interpretation

The local G3 result validates the numerical implementation, not end-to-end
language-model quality. LongBench scores, real Llama KV statistics, decode
latency, and CUDA peak-memory figures require the university RTX 5090 and the
gated Llama checkpoint. The reference cache validates functionality and stored
bytes but creates transient dequantized tensors for stock attention. Serving
speed or peak-memory claims require a fused systems path and separate evidence.

Current vLLM TurboQuant is used only as a systems-oriented cross-check. Its
documentation describes Hadamard rotation, MSE keys, uniform values, and norm
correction, and explicitly omits QJL. It is not treated as an identical
implementation of the paper algorithm.

## Primary references

- TurboQuant paper: <https://arxiv.org/abs/2504.19874>
- KIVI paper: <https://arxiv.org/abs/2402.02750>
- Official KIVI implementation: <https://github.com/jy-yuan/KIVI>
- Official LongBench repository: <https://github.com/THUDM/LongBench>
- Hugging Face cache interface: <https://huggingface.co/docs/transformers/main/en/kv_cache>
- vLLM TurboQuant notes: <https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/>
