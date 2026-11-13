# Problem 1 Report: Verification Notes

Companion to `Problem1_Report.docx` and `Problem1_Report.md`. Every number in the report was checked against `problem1/results_problem1.csv`, `problem1/har_summary_table.csv`, `problem1/adding_summary_table.csv`, `problem1/hidden_sweep_table.csv`, `problem1/ablation_table.csv`, the notebook outputs in `problem1/problem1.ipynb`, and the figures. No experiments were rerun.

## 1. Source files: what was found

| Requested | Status | Used instead / note |
|---|---|---|
| `problem1.ipynb` | Found (`problem1/problem1.ipynb`) | Primary source for code and printed outputs |
| `results_problem1.csv` | Found (`problem1/results_problem1.csv`) | Per-run values, 60 rows |
| `har_summary_table.csv` | Found (`problem1/`) | |
| `adding_summary_table.csv` | Found (`problem1/`) | |
| `hidden_sweep_table.csv` | Found (`problem1/`) | |
| `validation_results.csv`, `test_results.csv` | Found (`problem1/`) | Classical SVM etc.; not in the RNN report body |
| `test_confusion_matrix.csv` | Found (`problem1/`) | Classical SVM, not HAR RNN |
| `test_confusion_matrix.png` | Found (`problem1/`) | Classical SVM, not HAR RNN; not inserted (see §4) |
| `README.md` | Found (`problem1/README.md`) | Out of date on file locations (see §3) |
| `Pasted markdown (2).md` | **Not found** anywhere in the workspace | Used `problem1_report_data.md` (written in the previous step) as the detailed report data |
| `problem1_report_final_additions.md` | **Not found** | None |
| Assignment PDF | **The PDF on the Desktop (`2026AJ05432.pdf`) is the Steel-Plate ML assignment, a different course.** The DNN assignment is the `.docx` at `/Users/pavansai/Downloads/DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx` | Used the `.docx` for all assignment requirements |
| `report (1).pdf` | Found in `/Users/pavansai/Downloads/` | Not opened (not requested, and it is not a Problem 1 source file I could confirm) |

## 2. Verified experiments and values

Checked by recomputing from `results_problem1.csv`:

- **HAR (3 seeds, H=64, 30 epochs):** test accuracy, macro-F1, parameters and seconds per epoch match `har_summary_table.csv` and the per-run rows.
- **Adding problem (T = 50, 100, 200, 400; 3 seeds; 15 epochs):** all 36 main-run MSEs, the means and sample standard deviations, the solved counts, and the baselines match `adding_summary_table.csv` and the notebook output.
- **Parameter counts:** HAR totals 5190 / 19590 / 14790 (recurrent + 390 head); adding totals 4417 / 17473 / 13121 (recurrent + 65 head); sweep totals match the formulas for each H.
- **Cells vs PyTorch:** max differences 2.68e-07 (RNN), 5.96e-08 (LSTM), 9.69e-08 (GRU), from notebook output.
- **Hidden-size sweep:** all eight values match `hidden_sweep_table.csv`.
- **Ablation:** all four conditions match `ablation_table.csv`. Per-seed values match `results_problem1.csv` rows 27–29 and 51–59.
- **Gradient analysis:** the printed after-training ratios and t=T values match notebook output. The t=1 values in the report are derived as ratio × value and are not printed.
- **Confusion matrix (LSTM, 3 seeds summed):** values read from `figures/har_confusion_best.png`. Row totals are 1488, 1413, 1260, 1473, 1596, 1611 (sum 8841 = 3 × 2947). The diagonal sum 7838 / 8841 = 88.66%, which matches the LSTM mean test accuracy. This is a transcription from the image, not from a saved file.
- **Derived gradient numbers** and the rest of the figure descriptions were checked against the plots.

## 3. Conflicts and corrections found while verifying

1. **Ablation requirement.** The assignment assigns the ablation by the last digit of the student ID. For `2026AJ05432` the last digit is 2 (even), so the assigned variant is gradient clipping for the RNN at T=200. The report presents this as the assigned ablation. The LSTM forget-bias result is shown but labelled "not assigned".
2. **`problem1/README.md` project structure is out of date.** It lists `class_balance.png` at the top level of `problem1/`. The file is in `problem1/figures/`. It also omits several output files. Do not copy the README's file list into the report.
3. **Two different class-count sets.** The RNN class-balance plot uses 7352 full-training windows. The classical split uses 5952 windows. The markdown table in `problem1.ipynb` labels the 5952 counts as "Training Samples". The report uses the 7352 counts, which match `figures/class_balance.png`.
4. **SVM validation precision.** The notebook markdown states 0.8820. The stored output is 0.8861. The report does not use this classical number, so no change was needed in the body. The corrected value is listed in `problem1_report_data.md` §12.
5. **Best HAR model depends on the metric.** LSTM wins on validation macro-F1 (0.9129 vs 0.9116), GRU wins on mean test accuracy (88.71 vs 88.66), and LSTM wins on test macro-F1 (0.8867 vs 0.8863). The report calls these tied rather than picking one.
6. **Notebook working-directory inconsistency.** Some cells write to `problem1/figures` and `problem1/artifacts` relative to the project root, which created nested `problem1/problem1/` folders. The report does not depend on those files, but the notebook should be fixed before submission.

## 4. Figures inserted (all from the project, none synthesized)

| Figure | File | Experiment | Section |
|---|---|---|---|
| 1. HAR class balance | `problem1/figures/class_balance.png` | Data pipeline (full 7352 train vs 2947 test) | §2 |
| 2. LSTM HAR confusion matrix (3 seeds summed) | `problem1/figures/har_confusion_best.png` | HAR best model (by validation macro-F1) | §4 |
| 3. Adding test MSE vs T | `problem1/figures/adding_mse_vs_T.png` | Adding problem, all T and models | §5 |
| 4. Adding learning curves at T=200 | `problem1/figures/adding_learning_curves_T200.png` | Adding problem, T=200 | §5 |
| 5. Gradient norm vs time step (T=100, before/after) | `problem1/figures/grad_norm_vs_t_T100.png` | Gradient analysis | §6 |
| 6. HAR accuracy vs hidden size | `problem1/figures/hidden_sweep_har.png` | Hidden-size sweep (seed 0) | §7 |
| 7. T=200 ablation boxplots | `problem1/figures/ablation_T200.png` | Ablations (RNN clipping; LSTM forget bias) | §8 |

**Not inserted:**
- `test_confusion_matrix.png` (classical SVM, not an RNN result). Excluded to keep the report to Problem 1 RNN results.
- `har_class_balance.png` and `problem1/problem1/figures/class_balance.png` (classical split, 5952 windows). Duplicates of the RNN class-balance figure with different counts. Excluded to avoid confusion.

## 5. Missing information ([NEEDS VERIFICATION] in the report)

1. **HAR training and validation curves.** No per-epoch HAR histories are saved, and the notebook does not plot them. No figure exists, so none is inserted. The report flags this.
2. **Before-training gradient values.** The "before" profile is plotted but not printed or saved. The report describes the plot qualitatively and marks the numbers [NEEDS VERIFICATION].
3. **Ablation figure.** Figure 7 shows all four conditions together. The assigned clipping comparison is the left panel. No separate figure exists for the assigned variant alone.
4. **Variants table.** The assignment says hidden size and seeds come from an "Individualisation" table. That table is not in the brief copy I have. The report assumes H = 64 and seeds 0, 1, 2 and states this as an assumption (§3).
5. **Total Problem 1 runtime.** Only per-epoch times exist. The report does not give a total.
6. **Wearable recommendation.** The report includes one, labelled as a recommendation, not a measured result. Inference energy and on-device memory were not measured. One sentence (§10) about a smaller GRU (H=32) is marked [NEEDS VERIFICATION] because it comes from a single seed.
7. **Analysis text (Task 5d).** The report's §9 is new analysis written from the tables. The brief asks for a one-page analysis. Check whether the required content is covered.
8. **HAR confusion-matrix values.** The values were transcribed from the figure image and are not in a CSV. Check them against the notebook if you want an exact table.

## 6. Assignment requirements not yet satisfied

- **Page limit.** The brief caps the whole report at 6 pages, covering both problems. I could not render the document to count pages, because no PDF renderer (LibreOffice, Word, pandoc) is installed. My rough estimate is that Problem 1 alone is about 6 pages at the chosen figure sizes, so Problem 2 will not fit as is. Figure widths or text will need trimming once you have the full report. **Not verified.**
- **Problem 2.** Out of scope for this report.
- **AI-use statement.** Required in the report and not included here. The brief requires every AI use to be declared.
- **References section.** Not included.
- **Title block.** Name and student ID are included. Confirm the course title line.
- **Caption and reference check.** Every figure and table has a caption. Figures 1, 3, 5, 6 and 7 are referred to by number in the text. Figure 2 and Figure 4 are referred to only through their captions and nearby text, and Tables 1–5 are not all cross-referenced by number in the body. The brief requires every figure and table to be referred to in the text, so this needs fixing before submission.
- **Brief's figure list.** All seven Problem 1 figures are present (class balance, HAR confusion matrix, adding curves at T=200, MSE vs T, gradient norm vs t, hidden sweep, ablation). The HAR training/validation curves in the brief's list are **missing** (see §5 item 1).

## 7. Assumptions requiring your confirmation

1. **Hidden size H = 64 and seeds 0, 1, 2** are the student's variant. Confirm against the instructor's variants table.
2. **Student ID 2026AJ05432** is the one on the cover PDF and the folder name. The last digit (2) gives the even-ablation (RNN clipping). Confirm this is the ID the portal uses.
3. **The detailed data file** I used as the "Pasted markdown (2).md" stand-in is `problem1_report_data.md` (created in the previous step). If you have a different version of that file, send it and I will re-check.
4. **Wearable recommendation** (GRU) is my recommendation based on the tables. Keep, edit or remove it.
5. **Student name** "Pavansai Podichervu" comes from the cover PDF. Confirm the spelling for the portal.
6. **Figure sizes** were chosen to keep Problem 1 near one-third of the 6-page budget. Adjust after you assemble the full report.
