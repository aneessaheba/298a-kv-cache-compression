# Third party code

`src/turboquant/vendor/longbench/metrics.py` is copied without modification from
THUDM/LongBench, revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e`:
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py

Copyright (c) 2023 THU-KEG & Zhipu AI. MIT license retained beside the code.
The Team 6 wrapper applies `lstrip()` before first-line extraction for TREC,
TriviaQA and SAMSum instead of upstream `lstrip('\n')`. Metric functions are
unchanged. This policy is recorded as `longbench-v1-team6-leading-whitespace-v1`.
Scores are stored on a 0–100 scale. Use the same scoring policy for every method
and baseline; do not treat an unverified reported 47.72 as a reproduced baseline.

LongBench data and pretrained model weights are external inputs and are not
included in the GitHub source distribution. Follow the relevant dataset and
model licenses when obtaining those inputs.
