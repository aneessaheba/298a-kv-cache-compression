# Team harness integration status

This is Vedant's locally validated TurboQuant reference package, version 0.2.0.
It is an initial source-code contribution. RTX 5090 results and complete parity
with the shared harness are pending. The shared runners are not modified by this
contribution, and TurboQuant is not yet registered as a shared kvpress press.

## Checked against the current team repository

| Setting | Shared harness | Reference package |
|---|---|---|
| Dataset | Team processed LongBench dev file | Enforces frozen 800-record identity |
| Source prompt | Preprocessed record prompt | Same source field |
| Chat exceptions | trec, triviaqa, samsum, lcc, repobench-p | Same exceptions |
| Precision | BF16 | Runner supports BF16 via `--dtype bfloat16`; matrix scripts currently select FP16 |
| Compression boundary | Document compressible; question/instructions protected | Current adapter quantizes every cached token, including the question and generated suffix |
| Prompt construction | kvpress document/question pipeline, answer-cue handling | Single formatted prompt with Transformers generation |
| Scoring | Shared `harness/longbench.py`, scores in 0–1 | Pinned upstream metric functions, scores in 0–100 |
| Code similarity | rapidfuzz | upstream fuzzywuzzy; backend results can differ |
| Memory | Computed retained-token storage | Actual packed codes/norms plus separately reported quantizer overhead |

## Required before cross-method results

1. Use the same BF16 model revision, tokenizer, generation settings and exact input token IDs.
2. Integrate the shared document/question construction and protect the question
   and generated suffix, or agree on a common alternate protocol for all methods.
   Include all protected-token bytes in the memory budget.
3. Re-score all predictions with one chosen scorer and scale. Do not concatenate
   shared 0–1 scores with this package's 0–100 scores. Do not assume the two code
   similarity backends produce identical values.
4. Use all 800 IDs for the standard comparison. Adaptive evaluation excludes
   the frozen 32 calibration IDs from every method and its baseline (768 remain).
5. Compare observed memory budgets, not nominal bits versus eviction percentages.
6. Execute on the RTX 5090 and preserve environment, commands, predictions and manifests.

The checked-in shared baseline summary at
`results/longbench/Llama-3.1-8B-Instruct_none_0.0.summary.json` describes a
260-record run with `max_prompt_tokens=6000` and a 45.24 macro score. It is not
evidence of a full 800-record run or the reported 47.72 baseline. This contribution
does not change or reclassify the team's existing results.

## Running the reference package

Use an isolated environment in this directory so its Transformers pin does not
alter the team's kvpress environment:

```bash
cd members/vedant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf,plots,score]'
python -m pytest -q
```

The GPU runbook covers the standalone reference studies. Until the integration
steps above are complete, label their outputs as reference experiments rather
than shared-harness equal-memory results.
