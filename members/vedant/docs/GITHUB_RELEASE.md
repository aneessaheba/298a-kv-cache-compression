# GitHub source release

This source release contains the local implementation and validation evidence.
RTX 5090 and full Llama 3.1 8B results are pending and belong in a later results
commit. The original project remains at the same path.

## Before publishing

```bash
python -m pytest -q
python -m build --no-isolation
 git status --short
 git diff --check
```

Use the prepared Git repository or copy this source into the team repository.
No GitHub remote is selected automatically because the destination repository
has not been specified. Configure the correct repository and push after review:

```bash
git add README.md STATUS.md THIRD_PARTY_NOTICES.md MANIFEST.in .gitignore pyproject.toml
git add src scripts tests docs configs results/g3
git diff --cached --stat
git commit -m "Prepare reproducible TurboQuant source and team evaluation pipeline"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not add the old transfer ZIP, `.venv`, credentials, model weights or dataset
text. The 800-record dataset remains a team input with a frozen identity.
Regenerate calibration/context JSONL using `turboquant-split` after obtaining it.
The vendored official LongBench code retains its upstream MIT license; this
notice does not assign a new license to the team's own code.

## Later GPU results commit

Include the source commit, environment, model revision, data identity, commands,
logs, paired predictions/scoring, observed storage and peak memory, reports and
confidence intervals. Remove secrets and model weights. Review file sizes and
sharing permissions. Intentionally allow only the result artifacts you want to
publish; do not disable the whole results ignore rule.
