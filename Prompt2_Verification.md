# Prompt 2 verification: code, notebook, README and submission fixes

Scope: fixes to the Problem 1 and Problem 2 notebooks, the README and requirements, results consistency, and the submission folder. No ZIP was created. No training was rerun. The final report (`DNN_Assignment2_Report_Final.pdf`) was not modified.

## 1. Backups

All files changed in this revision were copied first to `archive_prompt2/` (inside the project, outside the submission folder). This includes both notebooks, `README_Final.md`, the old `problem2/README.md`, `problem2/results.csv`, `problem2/results_rows.csv`, the two 501-character sample files, the 4-head attention PNG, both requirements files, and `problem1/README.md`. A scratch copy of the original `problem2.ipynb` also exists in `/tmp/backup_prompt2/`.

## 2. Changes

### problem1/problem1.ipynb
| Cell | Change | Reason |
|---|---|---|
| 13 | `FIGURES_DIR = Path("problem1/figures")` → `Path("figures")` | Running from `problem1/` created `problem1/problem1/figures`. |
| 61 | `ARTIFACTS_DIR = Path("problem1/artifacts")` → `Path("artifacts")` | Same nested-path bug. |
| 60 (markdown) | `problem1/artifacts/` → `artifacts/` (inside `problem1/`) | Keeps the text consistent with the new path. |

Note: the audit's "cell 23" comparison is **not** in this notebook. Problem 1 cell 23 is the classical five-model validation table. The contaminated recurrent comparison is in `problem2/problem2.ipynb`, cell 23, and it was fixed there (section 2, Problem 2).

Not re-run: cells 13 and 61 were not executed (the HAR data is 386 MB and was not copied into a scratch folder). Their source edits were checked by string match only. The saved output of cell 61 still prints the old path `problem1/artifacts/best_har_model.joblib`. That output is stale until the notebook is re-run.

### problem2/problem2.ipynb
| Cell | Change |
|---|---|
| 5 | Shape assertions for Q, K, V, context and weights (attention). |
| 8 | Shape assertions for the 3-token example, plus a causal (upper-triangle zero) check. |
| 13 | Encoder-block shape checks: `(B, L, d) -> (B, L, d)`, attention weights `(B, heads, L, L)`, causal zeros. |
| 17 | HAR Transformer shape check `(B, 128, 9) -> (B, 6)` for `none`, `sinusoidal` and `learned`. |
| 23 | **Comparison fixed.** It now uses only the Problem 1 main runs (H = 64, 3 seeds; rows with `sweep` or `ablation` in the notes are excluded). It asserts 9 rows. Transformer values are read from `results.csv`, not hard-coded. |
| 24 (markdown) | Placeholder `<!--PATCH:comparison-->` replaced with the measured comparison. |
| 28 | Shape check for LM logits `(B, L, 65)`. |
| 36 | Samples are now exactly 500 generated characters, excluding the prompt (`ids[1:]`), with an assertion. Header updated. |
| 37 | Same convention for the no-mask sample. The model was not saved, so this cell was not re-run; see section 4. |
| 40 | Heatmap shows the **first two heads** of layer 1 (was four), with a shape assertion. |
| 41 (markdown) | Placeholder `<!--PATCH:heatmap-->` replaced with a qualitative description of the new figure. |
| 46 (markdown, §7.4) | Timing table changed to the canonical values from `results.csv` (0.0735, 0.1797, 0.4833, 1.9226). "Approximately quadratic" replaced with "faster than linear, between linear and quadratic". The earlier `results_rows.csv` values are named as unused. |
| 47 (markdown) | Placeholder `<!--PATCH:discussion-->` replaced with measured results and limitations. |

§7.5 of cell 46 already had the correct positional values (90.57, 88.53, 89.21). No change was needed.

Outputs: cells 5, 8, 13, 23, 28, 36 and 40 now carry the outputs of the verification run (section 3). Cell 37 has no output because it was not re-run. Cell 17 has no output, as before.

### Other files
| File | Change |
|---|---|
| `README.md` (root) | New. Replaces `README_Final.md` (which is kept). Setup, data, run order, outputs, hardware (only what was recorded), measured results, and limitations. |
| `problem2/README.md` | Replaced by a short folder note. The 951-line version (wrong 0.9057 attribution, contaminated Problem 1 values) is archived in `archive_prompt2/problem2_README_superseded.md`. |
| `problem1/README.md` | Structure tree: `class_balance.png` is now shown under `figures/`. |
| `requirements.txt` (root) | Unchanged. |
| `problem2/requirements.txt` | Pinned to the same versions as the root file (torch 2.15.0, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1). These match the project `.venv` (checked with `pip list`). |
| `results.csv` (root) | New. The 60 Problem 1 rows and the 12 Problem 2 rows, combined, with the 12 specified columns, 72 rows in total. Values are copied, not recomputed. |
| `problem2/generated_samples.txt` | Rewritten from the verified regeneration (section 3). Each sample is 500 generated characters, prompt excluded. |
| `problem2/generated_samples_nomask.txt` | Text-only edit: the prompt character was removed and the header updated. The no-mask model was never saved, so this cannot be regenerated without retraining. |
| `problem2/figures/attention_heads.png` | Replaced with the two-head figure from the verification run. The 4-head version is archived. |
| `problem2/results.csv` | Not changed. It is the canonical Problem 2 file. |
| `problem2/results_rows.csv` | Not changed, not deleted. Kept as evidence and not used. |
| Checkpoints (`*.pt`) | Not changed (timestamps unchanged from 21:03 on Nov 13, before this session). |

## 3. Validation performed

**Trimmed execution, no training.** `problem2/problem2.ipynb` was not run top to bottom. Instead, a scratch notebook was built from the edited cells (2, 5, 8, 11, 13, the HAR class block from cell 17, 26, 28, cell 23, cell 36 and cell 40), plus one verification cell that loads `transformer_lm.pt`. It was executed in a scratch copy of `problem2/` (`/tmp/p2wrap`) with the project venv kernel. It finished with **no errors**. Results:

- The reloaded checkpoint gives validation **1.6777 nats/char = 2.4204 bits/char**, matching the recorded value. This confirms the rebuilt model is the saved one.
- The 3-token example reproduces the recorded weights and outputs.
- The encoder block has 198,272 parameters, and all new shape checks pass.
- The HAR Transformer shape check passes for all three positional variants.
- The comparison table reproduces the 3-seed means: GRU 88.71 %, LSTM 88.66 %, RNN 85.04 %, with parameter counts 14,790, 19,590 and 5,190. It also reproduces the Transformer values: none 90.57, sinusoidal 88.53, learned 89.21.
- **Sample regeneration.** With the same seed and the same loop, the regenerated samples equal the stored samples exactly once the prompt character is removed (checked for 0.2, 0.8 and 1.5). The reproduction therefore confirms the stored text, and each sample is exactly 500 generated characters.
- The two-head figure was rendered from the checkpoint. Both heads are zero above the diagonal, as expected under the causal mask.

**Static checks.**
- Both notebooks pass `nbformat.validate` (problem1: 64 cells, problem2: 48 cells).
- No `TODO`, `FIXME`, `PATCH:` or `PLACEHOLDER` markers remain in either notebook.
- No remaining code path contains `problem1/problem1`. Code paths are relative to each notebook's folder, and Problem 2 reads `../problem1/...`.
- Sample file check: each of the three samples is 500 characters (checked by exact byte count). The no-mask sample is 500 characters.
- `results.csv` (root) has 72 rows and the 12 columns, in the specified order.

**Figures.** The 12 PNGs in `problem1/figures/` (7) and `problem2/figures/` (5) are present. Every figure the brief lists is covered except the memory-versus-context plot, which cannot be produced because memory was never measured (the time-versus-context plot is present). The two-head attention PNG was checked by viewing it.

## 4. Not done, and why

- **Full top-to-bottom run of either notebook:** not done. The verification above covers the edited cells only. Problem 1 cells 13 and 61 were not executed at all.
- **Problem 2 cell 37 (no-mask sample):** not re-run, because the no-mask model was not saved. Its file was changed by text edit only.
- **Problem 1 cell 61 saved output:** still shows the old path until a re-run.
- **Memory measurements, multi-seed Transformer runs, instructor's example, hardware and total runtime:** still missing, as before. No values were invented.

## 5. Conflicts with the final report (not changed)

The final report (`DNN_Assignment2_Report_Final.pdf`, unchanged, and byte-identical to the copy in the submission folder) still describes the old outputs:

1. **Figure 9 caption:** "Attention weights of the four layer-1 heads ... The brief asks for two heads; four are shown." The submitted figure now shows two heads.
2. **Table 8 note:** "Each sample is the start space plus 500 generated characters: 501 characters ... one more than the brief's 500." The submitted samples are now 500 generated characters each.
3. Any other "501" mention in the report should be checked against section 2 above.

These need a documented correction to the report (and a regenerated PDF) before submission. This was not done here because the report was to be left unchanged. Tell me if you want me to make that correction.

## 6. Remaining limitations (also in README.md)

- Every Transformer result uses one seed (42).
- The Transformer and Problem 1 recurrent models are not budget-matched.
- The LM ablations use 800 steps; the default uses 3000.
- Memory was not measured.
- The instructor's three-token example is not in the project.
- The no-mask sample is a text edit, not a regeneration.

## 7. Submission folder

`2026AJ05432_DNN_Assignment/` (27 files, 2.6 MB) contains `README.md`, `requirements.txt`, `results.csv`, `report.pdf`, `problem1/` (notebook, `results_problem1.csv`, four summary CSVs, seven figures) and `problem2/` (notebook, requirements, `results.csv`, both sample files, five figures). Excluded: datasets, checkpoints, `.venv`, backups, `archive_prompt2/`, `problem1/problem1/`, `results_rows.csv` and `problem2_results.csv`.
