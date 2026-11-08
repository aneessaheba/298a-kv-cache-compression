# GitHub source and later results

Authorized repository: https://github.com/aneessaheba/298a-kv-cache-compression

Source branch: `vedant/turboquant-reference-implementation`; draft PR #4. Version 0.3.0 adds the shared BF16 press, lab compatibility fixes, fractional presets, calibration corrections and behavioral tests. GPU results remain pending and belong in a later results commit.

Validate with the package test suite, shell syntax checks and `git diff --check`. Keep datasets, model weights, environment folders and credentials out of commits. Retain third-party notices.

Follow GPU_RUNBOOK.md in this checkout rather than overwriting it with the older Drive fix ZIP. Retrieve predictions, adjacent run manifests, summaries, environment, exact source commit and logs from the lab. Publish only intended result artifacts after checking them for credentials and size. Do not claim logical packed bytes are measured allocated GPU-memory savings.
