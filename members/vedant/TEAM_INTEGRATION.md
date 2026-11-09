# Team integration status

The source is integrated in `harness/run_longbench.py` as `turboquant_[mse_]{2,2.5,3,3.5,4}bit` and `turboquant_adaptive`. Other presses retain their existing run path. TurboQuant uses BF16, shared prompts, protected question/generated tokens, the canonical rapidfuzz scorer and the exact 800-record dataset hash.

The saved full-cache result in `results/longbench/Llama-3.1-8B-Instruct_none_0.0.jsonl` is all 800 records; its rounded task macro score is 47.72. The earlier 260-record/45.24 description was obsolete. Standalone fuzzywuzzy scores use another policy and cannot replace the shared baseline.

Document prefill stays full precision; the press changes stored document K/V after that forward. Actual kvpress pipeline tests verify that question and decode forwards do not trigger compression. The cache compatibility API is tested under Transformers 5.2.0. CUDA execution on the lab's Torch 2.7.1+cu128 remains to be tested.

Every TurboQuant result includes the nominal setting, per-layer settings, exact logical packed bytes, dtype, scorer scale/policy, fingerprint and model revision. Logical cache sizes include channel selectors and protected-token overhead; decoded BF16 remains resident. Read measured peak GPU memory separately.

Use `scripts/compare_team_runs.py` for existing shared method outputs. Do not run `turboquant-score` over shared results. Mixed bit rates and eviction ratios are not automatically equal budgets.

The supplied Drive fix archive and notes were reviewed and incorporated selectively. The current branch adds fractional rates, stronger behavioral tests, normalized/real-query calibration and strict run fingerprints beyond that archive. The Drive guide describes an older five-setting runner; follow the repository's current GPU_RUNBOOK.md instead. It also names a different fork: use the team's authorized `aneessaheba/298a-kv-cache-compression` repository and this branch.
