# Assignment Audit Checklist: DNN Programming Assignment-2

Audited against the assignment brief (`DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx`), the Problem 1 and Problem 2 notebooks, their CSV outputs and figures, `Problem1_Report_Verification.md` and `Problem2_Verification.md`. No experiments were rerun.

Status key: **PASS** = met and verified in project files. **PARTIAL** = some evidence exists but a stated requirement is unmet or unverified. **MISSING** = required and absent.

Summary (counted from the tables below): PASS 52, PARTIAL 23, MISSING 15, across 90 requirement rows.

---

## A. Deliverables and submission structure

| # | Requirement | Status | Evidence / what remains |
|---|---|---|---|
| A1 | Zip named `<StudentID>_DNN_Assignment.zip` | MISSING | Not created. Remaining: build the zip with the exact name `2026AJ05432_DNN_Assignment.zip`. |
| A2 | Folder `<StudentID>_DNN_Assignment/` at zip root | MISSING | Not created. Remaining: arrange the files into this folder before zipping. |
| A3 | `README.md` at folder root, one page max | PARTIAL | `README_Final.md` written. The old `problem2/README.md` is 951 lines and has wrong claims. Remaining: rename `README_Final.md` to `README.md` in the submission folder, and confirm it stays within one printed page. |
| A4 | `requirements.txt` with exact versions | PARTIAL | Root `requirements.txt` pins torch 2.15.0, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1. `problem2/requirements.txt` is unpinned (`torch>=2.0`, etc.). Remaining: pin the Problem 2 versions, or use one root file that covers both. |
| A5 | `report.pdf` at folder root, max 6 pages | MISSING | No PDF exists. Remaining: export `DNN_Assignment2_Report_Audited.docx` to PDF from Word or another renderer (none is installed here), then condense to 6 pages (see E1). |
| A6 | `results.csv` at folder root, one row per run, exact 12 columns (problem, model, seed, seq_len, test_acc, macro_f1, test_mse, params, sec_per_epoch, val_loss_nats, bits_per_char, notes) | MISSING | No root-level `results.csv`. `problem1/results_problem1.csv` (60 rows) and `problem2/results.csv` (12 rows) both have the 12 columns. Remaining: concatenate the two into one root `results.csv`, keeping the column order. Problem 2's `results_rows.csv` must not be merged (it has conflicting timings, see D4). |
| A7 | `problem1/problem1.ipynb` | PASS | Present. |
| A8 | `problem1/figures/` | PASS | Seven figures present. |
| A9 | `problem2/problem2.ipynb` | PASS | Present. |
| A10 | `problem2/generated_samples.txt` with three labelled samples of 500 characters | PARTIAL | Three labelled samples present, but each is 501 characters (start space plus 500 generated). Remaining: either regenerate with a 500-character count (this would be a rerun, which is not allowed here) or state the 501-character convention in the report and README (done in the report and README). Flagged as a deviation. |
| A11 | `problem2/figures/` | PASS | Five figures present. |
| A12 | Zip ≤ 25 MB; no datasets, checkpoints or virtual environment | MISSING | Not zipped. The working tree contains `.venv/`, `problem1/data/` (HAR dataset), `problem1/best_svm_model.pkl` (27.9 MB), `problem2/*.pt` checkpoints (3.7 MB and 3.3 MB) and `problem2/data/tinyshakespeare.txt`. Remaining: exclude all of these when zipping, then check the zip size. |
| A13 | Zip opens and contains `report.pdf` | MISSING | Depends on A1 and A5. |

## B. Problem 1 rubric (50 points)

| # | Requirement (rubric item) | Points | Status | Evidence / what remains |
|---|---|---|---|---|
| B1 | P1.1 HAR subject-wise validation split | 5 | PASS | `problem1.ipynb` cell 29 (`make_har_data`); per-seed validation subjects; asserts for no leakage. Report §2. |
| B2 | P1.1 Per-channel mean and std from training part only | 5 | PASS | Cell 29: statistics from `X_tr_all[tr]`. |
| B3 | P1.1 Class-balance plot | 5 | PASS | Report Figure 1 (`problem1/figures/class_balance.png`). Note: a second classical class-balance plot (5952-window split) exists at the root and in `problem1/problem1/`; it is not in the report. |
| B4 | P1.1 Adding-problem generator with its test (two markers, one per half, target = sum) | 5 | PASS | Cell 6 (generator) and cells 14 and 32 (asserts). |
| B5 | P1.2 All three cells match library layers within 1e-5 | 10 | PASS | Cell 9 output: max differences 2.68e-07 (RNN), 5.96e-08 (LSTM), 9.69e-08 (GRU). Report §3. Test uses a random batch, not HAR data; this is stated. |
| B6 | P1.2 Parameter counts derived by hand and reconciled | 10 | PASS | Cell 10 (markdown) and cell 11 (asserts): 4,736 / 18,944 / 14,208 by hand; 4,800 / 19,200 / 14,400 in PyTorch, one extra bias per gate block. |
| B7 | P1.3 Same hidden size, optimizer, batch and epoch budget for all three models | 10 | PASS | Cell 1 constants; report §4. Caveat: the variant hidden size is unconfirmed (see "Items that need student confirmation", item 1). |
| B8 | P1.3 Three seeds, mean ± std | 10 | PASS | `har_summary_table.csv`; recomputed from per-run CSV (report Table 1). |
| B9 | P1.3 Test accuracy, macro-F1, parameters, seconds per epoch | 10 | PASS | Report Table 1 and Table 2. |
| B10 | P1.3 Confusion matrix of best model | 10 | PASS | Report Figure 2 (LSTM, three seeds summed). The values were transcribed from the PNG; no CSV exists, so an exact table cannot be checked against a file. Remaining (optional): save the confusion matrix as CSV from the notebook. |
| B11 | P1.4 T = 50, 100, 200, 400 × three models × three seeds | 10 | PASS | `results_problem1.csv` rows 9–44 (36 rows). |
| B12 | P1.4 Test MSE against the about-0.167 baseline | 10 | PASS | Report Table 3 and Figure 3. Per-T baselines are 0.1754, 0.1697, 0.1679, 0.1527 (finite test sets); the population value is 1/6. The report explains this. |
| B13 | P1.4 Learning curves at T = 200 | 10 | PASS | Report Figure 4. Values are read from the plot; per-epoch numbers are not saved. |
| B14 | P1.4 Solved-run table (MSE < 0.05) | 10 | PASS | Report Table 3 (solved counts), `adding_summary_table.csv`. |
| B15 | P1.5(a) Gradient-norm plot ‖∂L/∂h_t‖ against t, log scale, before and after training, T = 100 | 15 | PASS | Report Figure 5. Caveat: before-training numeric values are not saved (see B16). |
| B16 | P1.5(a) Numeric gradient values before training | 15 | PARTIAL | Only the plot exists for before training; after-training ratios are printed. Remaining: rerun the gradient cell with output saved. This is a rerun and is not done here. The report marks the before-training numbers as [NEEDS VERIFICATION]. |
| B17 | P1.5(b) Hidden-size sweep {16, 32, 64, 128} for LSTM and GRU on HAR | 15 | PASS | Report Table 4 and Figure 6. Caveat: one seed only; flagged in the report. |
| B18 | P1.5(c) Individual ablation by the last digit of student ID; ID 2026AJ05432 ends in 2 (even) → gradient clipping for the vanilla RNN at T = 200 | 15 | PASS | Report §8, Table 5, Figure 7. Clipping variant run with three seeds. The LSTM forget-bias variant is also reported, labelled "not assigned". |
| B19 | P1.5(d) One-page analysis: what the gradient plot says about memory, where each model fails as T grows, accuracy–parameters–time trade-off | 15 | PASS | Report §5, §6 and §9. Length of §9 is about one page; check it again after condensation. |
| B20 | P1.5(d) Wearable recommendation justified from the tables | 15 | PASS | Report §10, labelled as a recommendation, with table-based reasons and caveats. The student may keep or edit it. |

## C. Problem 2 rubric (50 points)

| # | Requirement (rubric item) | Points | Status | Evidence / what remains |
|---|---|---|---|---|
| C1 | P2.1 Scaled dot-product attention with optional mask | 10 | PASS | Notebook cell 5. Masked positions are set to the dtype minimum before softmax. |
| C2 | P2.1 Unit tests: row sums equal 1 | 10 | PASS | Cell 5 assert (atol 1e-6). |
| C3 | P2.1 Output matches `F.scaled_dot_product_attention` within 1e-5 | 10 | PASS | Cell 5 assert (atol 1e-5); printed as passed. |
| C4 | P2.1 Causal mask gives zero weight to future positions | 10 | PASS | Cell 6 assert; printed as passed. |
| C5 | P2.1 Hand-checkable 3-token example, computed by hand and by code | 10 | PARTIAL | Hand and code agree (cells 8 and 9; report §2). The example is self-chosen. Remaining: the brief says the **instructor supplies** the example. Obtain it from the instructor and check the code against it. |
| C6 | P2.2 Multi-head attention with W_Q, W_K, W_V, W_O and head reshaping | 10 | PASS | Cell 11 `MultiHeadAttention`. |
| C7 | P2.2 Pre-LayerNorm encoder block, FFN width 4d, residuals, dropout | 10 | PASS | Cell 11 `EncoderBlock`: d_ff = 512 = 4·128; dropout 0.1; residuals. |
| C8 | P2.2 Sinusoidal positional encoding | 10 | PASS | Cell 11 `SinusoidalPositionalEncoding`. |
| C9 | P2.2 Weight count 4d² + 2·d·d_ff; reconcile with `sum(p.numel())` | 10 | PASS | Cell 13 assert: 198,272 per block; weights 196,608 + biases 1,152 + LayerNorm 512. |
| C10 | P2.2 Correct shapes | 10 | PARTIAL | Shape comments exist in the code. No explicit shape assertions or shape-test cells were found. Remaining: add shape asserts for the attention and encoder outputs (this is code, not a rerun of results). |
| C11 | P2.3 Embed each time step (or patch), add CLS token, classify with the encoder | 10 | PASS | Cell 17 `HARTransformer`: per-step embedding 9→128, CLS prepended, classification from CLS. |
| C12 | P2.3 Reuse Problem 1 data | 10 | PASS | Reads `../problem1/data/har/UCI HAR Dataset`. Caveat: uses the classical split (seed 42), not the per-seed Problem 1 splits. |
| C13 | P2.3 Compare accuracy, parameters and training time with best Problem 1 model | 10 | PARTIAL | Report §4 gives the comparison using the correct Problem 1 values (GRU 88.71%). Caveats: one Transformer seed versus three Problem 1 seeds; different epochs (15 vs 30), optimizer (AdamW vs Adam) and checkpoint rule (final epoch vs best validation). The notebook and README values for Problem 1 were wrong and are corrected in the report. |
| C14 | P2.3 Ablate positional information: none, sinusoidal, learned | 10 | PASS | Cell 19; report Table 6 and Figure 8 (default model learning curve). |
| C15 | P2.4 Tiny Shakespeare, context 128, d = 128, 4 heads, 4 layers | 10 | PASS | Cells 26–30; 801,664 parameters. |
| C16 | P2.4 Report validation loss and bits per character | 10 | PASS | Cell 30: 1.6777 nats/char; 2.4204 bits/char (= nats / ln 2). Report §5. |
| C17 | P2.4 Learning curve | 10 | PASS | Report Figure 9. |
| C18 | P2.4 Generate 500 characters at temperatures 0.2, 0.8 and 1.5 with top-k | 10 | PARTIAL | Three samples at top-k 40 exist, but each is 501 characters (see A10). Remaining: regenerate with 500 characters (a rerun, not allowed in this audit) or accept the 501-character convention and say so. |
| C19 | P2.4 Describe the differences between temperatures | 10 | PARTIAL | Qualitative descriptions are in report Table 8. These are the author's observations of the text, not measured quantities. Remaining: the student should check the descriptions against the samples. |
| C20 | P2.5(a) Heatmaps of two heads on one sentence, with interpretation | 10 | PARTIAL | Report Figure 10 shows four heads for "To be, or not to be". The brief asks for two heads; four is more than required. The notebook's own interpretation (cell 39) is generic. The report's observations are qualitative, drawn from the figure. |
| C21 | P2.5(b) Heads = 1, 4 and 8 at fixed d | 10 | PASS | Cell 33; report Table 9 and Figure 11. One seed; 800 steps. |
| C22 | P2.5(c) Remove the causal mask; show training loss and generation | 10 | PASS | Cell 33 (training loss); cell 37 and `generated_samples_nomask.txt` (generation); report Table 10 and Figure 11. |
| C23 | P2.5(d) Time and memory at context 64, 128, 256 and 512; relate growth to O(n²·d) | 10 | PARTIAL | Time at all four lengths: report Table 11 and Figure 12. Memory: **not measured** (n/a on CPU). See E3. |
| C24 | Bonus: key-value cache or LSTM-versus-Transformer character model | up to 5 | MISSING | Not attempted. It is optional; no points are claimed. |

## D. Report content and formatting

| # | Requirement | Status | Evidence / what remains |
|---|---|---|---|
| D1 | Title block with name and student ID | PASS | Title and header lines in `DNN_Assignment2_Report_Audited.docx`. |
| D2 | Problem 1: setup (datasets, split, preprocessing) | PASS | Report §2. |
| D3 | Problem 1: cell verification and parameter counts | PASS | Report §3. |
| D4 | Problem 1 HAR results table; adding-problem table and curves; experiments 5(a)–(c); analysis | PASS | Report §4–§9, Tables 1–5, Figures 2–7. |
| D5 | Problem 2: attention and encoder-block tests; parameter derivation | PASS | Report Problem 2 §2–§3. |
| D6 | Problem 2: Task A table; Task B learning curve and bits per character; heatmaps and ablations; analysis | PASS | Report Problem 2 §4–§7; Tables 6–11; Figures 8–12. |
| D7 | Every figure and table numbered, captioned and referred to in the text | PASS | All 12 figures and 14 tables carry captions. In-text references were added for each one in this audit. |
| D8 | AI-use statement declaring every tool use | PARTIAL | A draft statement is in the report with [NEEDS VERIFICATION] markers. The student must state which tools were used for what, including whether any AI tool wrote the from-scratch cells (the brief forbids this). Remaining: the student completes and confirms it. |
| D9 | References | PARTIAL | A reference list is in the report (12 items, standard sources and the brief). Marked [NEEDS VERIFICATION]. Remaining: the student checks each entry's details and formatting. No citation was invented, but the entries have not been checked against the original documents here. |
| D10 | Report ≤ 6 pages, A4, 11 pt, margins ≥ 2 cm | MISSING | The combined report is not condensed (the user asked to leave it long for now). Page count not verified: no renderer is installed. Margins are 2.2 cm; font is 10.5 pt (not 11 pt). Remaining: condense to 6 pages, use 11 pt, and check the page count on the PDF. |
| D11 | Problem 2 HAR results and comparison use correct values | PASS | Problem 1 comparison values corrected from contaminated notebook/README values; Problem 2 positional values taken from CSV rows. |
| D12 | Assumptions clearly stated | PASS | Report §3 (Problem 1 assumptions) and Problem 2 §4 (budget and split). |

## E. Missing evidence (explicitly required or flagged)

| # | Item | Status | Remaining |
|---|---|---|---|
| E1 | Six-page limit for the combined report | MISSING | Condense after all content is final. Not done here by instruction. |
| E2 | Instructor-supplied attention example | MISSING | Obtain from the instructor; check the code and the hand computation against it. |
| E3 | Context-length memory measurements (64, 128, 256, 512) | MISSING | Requires a rerun on a machine that reports memory (CUDA `max_memory_allocated`, or an equivalent CPU measure). Not done here. |
| E4 | Multi-seed evaluation of the Transformer models | MISSING | Requires reruns with several seeds (the notebook uses seed 42 only). Not done here. |
| E5 | Problem 1 before-training gradient values | PARTIAL | Only the plot exists (see B16). |
| E6 | HAR training and validation curves for Problem 1 | MISSING | Not required by the brief's figure list, but the report's own outline asked for them. Histories were not saved. Remaining: save per-epoch histories in a rerun. |
| E7 | Total runtime per notebook on the hardware used | MISSING | Not recorded. Remaining: record the wall-clock time for each notebook on the machine actually used, and the CPU model, core count and memory. These are not invented here; the README says so. |
| E8 | Hardware details (CPU model, RAM) | MISSING | Not recorded anywhere in the project. |
| E9 | Notebook outputs saved and run top to bottom | PARTIAL | Problem 1: 39 of 41 code cells have saved outputs; two cells without outputs were not identified. Problem 2: not counted. Restart and Run All was not verified in this audit. Remaining: restart and run both notebooks in order, which is a rerun and must be done by the student. |

## F. Code requirements

| # | Requirement | Status | Evidence / what remains |
|---|---|---|---|
| F1 | Each notebook runs top to bottom with Restart and Run All | PARTIAL | See E9. |
| F2 | Unit tests are assert cells placed directly after the function they test | PASS | Problem 1 cells 6→14/32 and 29; Problem 2 cells 5, 6, 8, 13. |
| F3 | Relative paths only; no `/content/drive/...` or other account-specific paths | PARTIAL | Problem 2 and the Problem 1 RNN section use relative paths. Problem 1 classical cells 13 and 61 write to `problem1/...`, relative to the project root, so running from the project root creates nested folders (`problem1/problem1/`). Remaining: change those paths to be relative to `problem1/`. |
| F4 | Comments explain tensor shapes in attention and LSTM code | PASS | Problem 1 cell 8 (LSTM shapes); Problem 2 cell 5 (attention shapes). |
| F5 | Fixed seeds for all of Python, NumPy and PyTorch | PASS | `set_seed` in both notebooks. |

## G. Problem 1 and Problem 2 consistency and factual checks

| # | Check | Status | Evidence / what remains |
|---|---|---|---|
| G1 | HAR Transformer accuracy attribution in the README (0.9057 called the sinusoidal model) | PASS (corrected) | 0.9057 is the **no-positional-encoding** model; the sinusoidal model is 0.8853 (`problem2/results.csv` row 2). The corrected README (`README_Final.md`) does not repeat the claim. The old `problem2/README.md` still contains it. Remaining: replace or delete the old README in the submission. |
| G2 | Problem 1 values in the notebook comparison (cell 23) and README | PARTIAL | The notebook and README values (LSTM 0.8824, GRU 0.8932, params 23,041 and 17,374) are contaminated by sweep rows. The report uses correct values (LSTM 88.66, GRU 88.71). Remaining: fix cell 23 to use only the H = 64 rows, and fix the README; these are code and text edits, not reruns. |
| G3 | Timing conflict: `results.csv` and cell 43 vs `results_rows.csv` and notebook §7.4 | PARTIAL | Canonical source chosen: cell 43 live output, which matches `results.csv`. `results_rows.csv` and §7.4 are not reconciled. Remaining: decide which run to keep and delete or regenerate the other. The source of the second set is not recorded. |
| G4 | LM training time: 10.1 min (`results.csv`) vs 10.0 min (`results_rows.csv`) | PARTIAL | Conflict noted in the report. Canonical source: `results.csv`. |
| G5 | Problem 1 `class_balance.png` in the README project structure | PARTIAL | README lists it at the top level; it is in `problem1/figures/`. The corrected README does not repeat the error. |
| G6 | Problem 1 HAR best-model artifact | PARTIAL | `problem1/problem1/artifacts/best_har_model.joblib` holds the classical SVM, not a recurrent model. Noted in `Problem1_Report_Verification.md`. Remaining: rename the file or save the recurrent model. |
| G7 | Tiny Shakespeare sample length | PARTIAL | See A10 and C18. |

---

## Items that need student confirmation

1. The variant hidden size (H = 64) and seeds (0, 1, 2) for Problem 1; the variants table is not in the brief copy provided (`Problem1_Report_Verification.md`).
2. The student ID is taken from the cover PDF and folder name (2026AJ05432); last digit 2 gives the clipping ablation.
3. The AI-use statement (D8).
4. The reference list (D9).
5. The GRU wearable recommendation (Problem 1 report §10).
6. The 501-character sample convention (A10, C18) and whether to keep it or regenerate.
