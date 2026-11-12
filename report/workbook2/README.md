# Workbook 2 evidence and regeneration

`../T6.A6.docx` follows Template 5 from DATA 298A Student Templates v5. Sections 3.1–4.4 contain report prose; 4.5 and 4.6 remain headings as required by that template. This workbook references team main snapshot 39a1097 and TurboQuant evidence 9dc0d48 (PR 5). It does not declare unpushed GPU results complete.

The independent audit reads the public LongBench raw archive and the already frozen team full/development exports. It verifies counts and saved IDs, detects exact-context overlap, and preserves short matched source examples. It does not regenerate prompts or alter experiment inputs. Large datasets and model weights are not part of this commit.

From the repository root, with Python 3.12, NumPy and Matplotlib available:

```bash
python report/workbook2/audit_data.py --raw-archive /path/to/data.zip --full /path/to/processed_longbench.jsonl --dev /path/to/processed_longbench_dev.jsonl
python report/workbook2/make_figures.py --dev /path/to/processed_longbench_dev.jsonl
```

Download the raw archive from https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip if it is not cached. The prepared exports are the team's existing data handoff, not a new sample. The manifest and audit record their hashes.

All five workbook figures are produced by `make_figures.py`. The script deliberately plots RestoreKV and TurboQuant on the same 800-record scope and leaves TriAttention's 260-record, 12-task study separate. Raw source/archive terms and the model license remain applicable.

## Material findings

- Raw 3,750 → cleaned 3,731; development 800; 21 actual export fields.
- Exact token cap: 88 truncated in the full pool, 21 in development.
- Calibration/held-out IDs do not overlap; four document contexts do. This is record-held-out, not document-held-out.
- RDKV and ResidualKV are integrated; a completed shared 800-record outcome is not included in the audited snapshot.
- TurboQuant's adaptive allocator selected 3-bit QJL at all layers; no adaptive benefit is established.
- Computed cache bytes are distinct from allocated GPU memory.
