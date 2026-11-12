# GitHub source and Semester 1 evidence

Repository: https://github.com/aneessaheba/298a-kv-cache-compression

Source PR #4 has merged. The completed experiments used pinned commit `82d40f72eb8c3a8313c4da95d2dbe22913aaad7a`; archived fingerprints are intentionally not rewritten to a newer main commit. The later reporting branch is `vedant/turboquant-workbook2-results`.

The results release contains ten complete experiments (nine fixed and one adaptive), 8,000 per-record outputs with manifests/summaries, 512 real-KV layer reports, environment/log evidence, CPU-only validation/figure regeneration and Workbook 2 contribution text. [Entry point](../results/sem1/README.md).

Large weights, raw prompt datasets, real-KV tensors and full backup archives remain outside Git. The original archives are independently backed up and checked. Quantizer code and the shared evaluation harness are unchanged in this reporting release. Preserve the distinction between computed packed size and allocated GPU memory; preserve the null adaptive outcome.
