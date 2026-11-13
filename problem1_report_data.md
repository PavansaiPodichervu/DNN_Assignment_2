# Problem 1 — Report Data (RNN, LSTM, GRU from scratch; HAR; Adding Problem)

Compiled from project files only. **No experiments were rerun.** Values are copied from the CSV files and notebook outputs listed below, or recomputed from `results_problem1.csv` where stated.

Conventions used in this file:
- **[MEASURED]** = value stored in a CSV or printed in a notebook output cell.
- **[RECOMPUTED]** = value I recomputed from the per-run CSV (matches the stored summary).
- **[DERIVED]** = arithmetic from measured values (shown explicitly).
- **[INTERPRETATION]** = my reading, not a measured quantity.
- **[MISSING]** = not present in the project.

Project root: `/Users/pavansai/Desktop/2026AJ05432_DNN_Assignment2/`
Problem 1 directory: `problem1/`
Main notebook: `problem1/problem1.ipynb` (64 cells: 41 code, 23 markdown; 39 of 41 code cells have saved outputs).

---

## 0. Source files and their roles

| File | Role | Status |
|---|---|---|
| `problem1/problem1.ipynb` | Combined notebook: Part A (classical baselines), Part B (RNN/LSTM/GRU, adding problem, gradients, sweep, ablation), Part C (model selection, classical). | Executed, outputs saved |
| `problem1/classical_baselines.ipynb` | Earlier classical-ML notebook (LR, DT, KNN, NB, SVM). Writes root-level `validation_results.csv`, `test_results.csv`, `test_confusion_matrix.csv`/`.png`, `best_svm_model.pkl`. | Executed, outputs saved (last modified Nov 9) |
| `problem1/data/problem1.ipynb` | Template/brief notebook: "Assumptions and deviations" cell and task skeleton. **No outputs.** | Not an executed result |
| `problem1/data/problem1_continuation_clean.ipynb` | Continuation notebook for classical model selection. **No outputs.** | Not an executed result |
| `problem1/results_problem1.csv` | Per-run results (60 rows, 12 columns). | Source of truth for RNN/LSTM/GRU numbers |
| `problem1/har_summary_table.csv` | HAR mean ± std table. | Summary |
| `problem1/adding_summary_table.csv` | Adding-problem mean ± std and solved counts. | Summary |
| `problem1/hidden_sweep_table.csv` | Hidden-size sweep (seed 0). | Summary |
| `problem1/ablation_table.csv` | T=200 ablations. | Summary |
| `problem1/test_results.csv`, `validation_results.csv`, `test_confusion_matrix.csv` | Classical SVM/LR/DT/KNN/NB results. | From `classical_baselines.ipynb` |
| `problem1/problem1/artifacts/*` | Classical model (joblib) and metrics copies. | Path is nested (see §9) |
| `problem1/verifyproblem1.py` | Asserts the 6×6 classical test confusion matrix sums to 2947. | Not run here |
| `problem1/README.md` | Problem 1 README (untracked in git). | See conflicts §9 |
| `requirements.txt` (root) | Pins `torch==2.15.0`, `numpy==2.5.4`, `pandas==3.1.1`, `matplotlib==3.11.2`, `scikit-learn==1.9.1`; notebook-only `nbformat`, `nbconvert`, `ipykernel`. | Present |

Environment printed by the notebook (cell 2, `problem1.ipynb`): `NumPy: 2.5.4`, `PyTorch: 2.15.0`, `CUDA available: False`. Python interpreter used: `.venv/lib/python3.13` (`.venv` path in the sklearn warning, cell 20). Torch thread count set to 4 (cell 27 output: `torch 2.15.0 | threads 4`).

---

## 1. Dataset details

### 1.1 UCI HAR (Human Activity Recognition Using Smartphones)

- Source download in notebook: `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip` (`problem1.ipynb`, cell 3, "Download and extract HAR"). A nested `UCI HAR Dataset.zip` is extracted (cell 4).
- Local path used by RNN section: `data/har/UCI HAR Dataset/` (relative to `problem1/`; asserted in cell 29).
- Input: 9 channels × 128 time steps; 6 classes.
- Channel order (cell 29, `SIGNALS`): `body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z`.
- Loaded shapes [MEASURED, cell 6/cell 29]: train `(7352, 128, 9)`, test `(2947, 128, 9)`.
- Labels loaded as `y - 1` → classes 0..5 (cell 29).

**Class mapping** (cell 29 and `problem1.ipynb` cell 62 / continuation step notebook):

| Index (0-based in code) | Raw label (file) | Activity name used in plots/tables |
|---|---|---|
| 0 | 1 | Walking |
| 1 | 2 | Walking Upstairs |
| 2 | 3 | Walking Downstairs |
| 3 | 4 | Sitting |
| 4 | 5 | Standing |
| 5 | 6 | Laying |

Note: `test_confusion_matrix.png` (root) labels its axes `0`–`5` (numeric), not activity names. The combined notebook's classical confusion-matrix tables use the names above.

**Subjects** [MEASURED, cell 7 output]:
- Train-file subjects (21): `[1 3 5 6 7 8 11 14 15 16 17 19 21 22 23 25 26 27 28 29 30]`
- Test-file subjects (9): `[2 4 9 10 12 13 18 20 24]`
- The train and test subject sets do not overlap.

### 1.2 Subject-wise validation split — RNN/LSTM/GRU section

Method (cell 29, `make_har_data(seed)`): `val_subjects = np.random.default_rng(seed).choice(np.unique(subj_tr), 4, replace=False)`; seeds 0, 1, 2.

| Seed | Validation subjects (4) | Train windows | Validation windows | Test windows |
|---|---|---|---|---|
| 0 | [8, 17, 25, 21] | 5886 | 1466 | 2947 |
| 1 | [25, 16, 29, 15] | 5905 | 1447 | 2947 |
| 2 | [11, 25, 5, 7] | 6017 | 1335 | 2947 |

[MEASURED, cell 29 output.] Check: 5886+1466 = 7352; 5905+1447 = 7352; 6017+1335 = 7352 [DERIVED].

Leakage asserts (cell 29): `set(subj_tr[tr]).isdisjoint(val_subjects)` and `set(val_subjects).isdisjoint(subj_te)` — both present as `assert` statements.

### 1.3 Subject-wise validation split — classical section (different split)

From `problem1.ipynb` cell 8 and `classical_baselines.ipynb` cell 6 [MEASURED]:
- `subject_wise_split(subjects_train, seed=42)` → validation subjects `[16, 3, 22, 23]`.
- Train shape `(5952, 128, 9)`; validation shape `(1400, 128, 9)`; flattened to `(5952, 1152)`, `(1400, 1152)`, test `(2947, 1152)` (cell 16).
- **This split is different from the RNN per-seed splits in §1.2.** See conflicts §9.

### 1.4 Preprocessing and normalization

- **RNN section (cell 29):** per-channel mean and standard deviation computed on the training part only, over axes (0, 1): `mu = X_tr_all[tr].mean(axis=(0,1), keepdims=True)`, `sd = X_tr_all[tr].std(axis=(0,1), keepdims=True) + 1e-6`. Validation and test are normalized with these training statistics. Output dtype float32. Shapes of normalized arrays printed for seed 0: train `(5886,128,9)`, val `(1466,128,9)`, test `(2947,128,9)`.
- **Classical section (cell 10):** normalization stats from training part; "Mean shape: (9,)", "Std shape: (9,)", "Normalized train: (5952, 128, 9)".
- Windows are fed as-is (no windowing/overlap processing beyond the provided 128-step windows).

### 1.5 Class distribution

Two different class-count sets exist in the project. They are **not the same data**:

**(a) Full 21-subject training file, 7352 windows** — printed in `problem1.ipynb` cell 30 (RNN section) and plotted in `problem1/figures/class_balance.png`:

| Activity | Count |
|---|---|
| Walking | 1226 |
| Upstairs | 1073 |
| Downstairs | 986 |
| Sitting | 1286 |
| Standing | 1374 |
| Laying | 1407 |

Sum = 7352 [DERIVED].

**(b) Classical train split, 5952 windows** (after removing validation subjects [16,3,22,23]) — printed in `problem1.ipynb` cell 62 (Part C) and plotted in root `har_class_balance.png`:

| Activity | Training | Validation | Test |
|---|---|---|---|
| Walking | 1012 | 214 | 496 |
| Walking Upstairs | 870 | 203 | 471 |
| Walking Downstairs | 800 | 186 | 420 |
| Sitting | 1035 | 251 | 491 |
| Standing | 1104 | 270 | 532 |
| Laying | 1131 | 276 | 537 |

Training sum = 5952; validation sum = 1400; test sum = 2947 [DERIVED from cell 62 output].

**Test class counts are identical in both splits** (496, 471, 420, 491, 532, 537; total 2947) [MEASURED].

Imbalance: moderate. Ratio max/min in (a) = 1407/986 = 1.43; in (b) = 1131/800 = 1.41 [DERIVED]. Markdown in `problem1.ipynb` cell 63 §5.2 describes it as "moderate class imbalance rather than extreme imbalance" [INTERPRETATION by the notebook author].

### 1.6 Adding-problem dataset

Generator (`problem1.ipynb` cell 6; identical to the brief's function):
- `x[:, :, 0] = rng.random((n, T))` (values in [0,1)); `x[:, :, 1]` = marker channel; exactly two markers: `i1 ∈ [0, T//2)`, `i2 ∈ [T//2, T)`; target `y = x[i1,0] + x[i2,0]`.
- Seeds: train n=10000 seed 0; validation n=1000 seed 1; test n=1000 seed 2 (cell 43: `ADD_DATA[T] = (make_adding(10000,T,0), make_adding(1000,T,1), make_adding(1000,T,2))`).
- Shapes printed for T=50 (cell 14): train `(10000, 50, 2)`, val `(1000, 50, 2)`, test `(1000, 50, 2)`.
- Generator test (cell 14 / cell 32): asserts every sequence has 2 markers, one in each half, and target equals sum of marked values. Output: "All adding problem tests passed!" / "adding-problem generator tests passed".
- Validation set is generated but **not used for early stopping in the adding-problem runs beyond the model-selection rule in §2.4**; it is used for the best-epoch selection (`run_model`).

---

## 2. Recurrent cells from scratch: equations, implementation, verification

### 2.1 Equations as implemented (`problem1.ipynb` cell 8, Task 2)

Helper: `_affine(x, W, b) = x @ W.T + b` with `x: (B, in)`, `W: (out, in)`.

**Vanilla RNN — `MyRNNCell`** (tanh):
h_t = tanh( W_ih x_t + b_ih + W_hh h_{t−1} + b_hh )
- `weight_ih` (h, d), `weight_hh` (h, h), `bias_ih` (h), `bias_hh` (h).

**LSTM — `MyLSTMCell`**, gate order (i, f, g, o) as PyTorch:
gates = W_ih x_t + b_ih + W_hh h_{t−1} + b_hh  (shape (B, 4h)); split into i, f, g, o.
i = σ(·), f = σ(·), o = σ(·), g = tanh(·)
c_t = f ⊙ c_{t−1} + i ⊙ g
h_t = o ⊙ tanh(c_t)
- `weight_ih` (4h, d), `weight_hh` (4h, h), `bias_ih` (4h), `bias_hh` (4h).

**GRU — `MyGRUCell`**, gate order (r, z, n) as PyTorch:
gi = W_ih x_t + b_ih (3h); gh = W_hh h_{t−1} + b_hh (3h)
r = σ(i_r + h_r); z = σ(i_z + h_z)
n = tanh(i_n + r ⊙ h_n)   ← reset gate applied to (W_hn h + b_hn), PyTorch convention
h_t = (1 − z) ⊙ n + z ⊙ h_{t−1}
- `weight_ih` (3h, d), `weight_hh` (3h, h), `bias_ih` (3h), `bias_hh` (3h).

**Unrolling (`Unrolled` class, cell 8):** loops over T steps, stores each `h_t` in `self.states` (used for gradient retention in Task 5a). Returns `(outputs (B,T,h), final_state)`. Initial state zeros.

**Weight init** in the from-scratch cells: uniform(−1/√h, 1/√h) (RNN) or uniform(−1/√h, 1/√h) for all LSTM/GRU tensors (cell 8). Library layers use PyTorch default init; the from-scratch cells are **copied from the library layer** (`copy_from_library`) for verification.

Library-to-cell mapping: `lib.weight_ih_l0 → c.weight_ih`, `weight_hh_l0 → weight_hh`, `bias_ih_l0 → bias_ih`, `bias_hh_l0 → bias_hh` (cell 8).

### 2.2 Verification against PyTorch (`problem1.ipynb` cell 9 output) [MEASURED]

Setting: d = 9, h = 64, batch B = 32, T = 128, random input `torch.randn`, seed 0, `eval()` mode, tolerance 1e-5.

| Cell | max \|output diff\| | max \|final state diff\| |
|---|---|---|
| RNN | 2.68e-07 | 1.79e-07 |
| LSTM | 5.96e-08 | 8.94e-08 |
| GRU | 9.69e-08 | 5.96e-08 |

Printed: "All three cells match nn.RNN / nn.LSTM / nn.GRU within 1e-5."

Gradient-profile check (cell 46 code, Task 5a): the from-scratch cell with trained weights reproduces the trained library layer outputs; assertion `diff < 1e-5` passed (cell did not raise; no printed diff value).

### 2.3 Parameter counts (d = 9, h = 64)

Hand derivation (cell 10 markdown; cell 11 asserts):
- One bias vector per gate block: RNN h(d+h)+h = 64·73+64 = **4,736**; LSTM 4× = **18,944**; GRU 3× = **14,208**.
- PyTorch / my cells keep two bias vectors: RNN **4,800** (adds h), LSTM **19,200** (adds 4h), GRU **14,400** (adds 3h).

Measured (cell 11 output):

| Model | Hand (one bias) | My cells | nn.* layer |
|---|---|---|---|
| RNN | 4736 | 4800 | 4800 |
| LSTM | 18944 | 19200 | 19200 |
| GRU | 14208 | 14400 | 14400 |

Printed: "Parameter counts reconcile: hand derivation + one extra bias vector per gate block = PyTorch."

Recurrent-layer-only counts equal those in the table; the classifier head is not included in these numbers.

### 2.4 Model-selection rule used everywhere in the RNN section

`run_model` (cell 39): Adam, lr = 1e-3, batch 64, gradient-norm clip (1.0 unless stated), keeps the **epoch with lowest validation loss** (`best["val"]`). Test is evaluated once with those weights (`predict(model, te, ...)`).

### 2.5 Verification results — summary
- All three cells match library layers within 1e-5 (max diff ~2.7e-7). [MEASURED]
- Parameter reconciliation matches. [MEASURED]
- Verification was done on random inputs, in eval mode, not on the HAR data. [INTERPRETATION of scope]

---

## 3. HAR classification experiments

### 3.1 Configuration (`problem1.ipynb` cells 12–13, 39; markdown cell 26 "Assumptions and deviations")

| Setting | Value | Source |
|---|---|---|
| Architecture | `RecClassifier`: `nn.RNN` / `nn.LSTM` / `nn.GRU` (library, batch_first) → last hidden state `out[:, -1]` → `nn.Linear(h, 6)` | cell 13 |
| Hidden size H | 64 | cell 1 `H = 64` |
| Input | 128 × 9 | cell 13 |
| Optimizer | Adam, lr 1e-3 | cell 1 `LR = 1e-3` |
| Batch size | 64 | cell 1 |
| Epochs | 30 (HAR) | `HAR_EPOCHS = 30` |
| Gradient clipping | norm 1.0 | `CLIP = 1.0` |
| Loss | cross-entropy (`F.cross_entropy`) | `CE_LOSS` |
| Seeds | 0, 1, 2 (`set_seed(s)`) | `SEEDS = [0, 1, 2]` |
| Validation subjects | per seed, see §1.2 | `make_har_data` |
| Model selection | lowest validation loss epoch | `run_model` |
| Device | CPU (`DEVICE = cpu`; "no CUDA here") | cell 1 |
| Threads | `torch.set_num_threads(4)` | cell 1 |

Brief-compliance note: identical hidden size, optimizer, batch size, and epoch budget for all three models (per notebook markdown) [INTERPRETATION of code; confirmed by cell 1 constants].

### 3.2 Per-seed results [MEASURED, `results_problem1.csv` rows 0–8; matches notebook cell 39 printed output]

| Model | Seed | Test acc (fraction) | Test macro-F1 | Best epoch (0-based) | Val macro-F1 | Params (total) | Sec/epoch |
|---|---|---|---|---|---|---|---|
| RNN | 0 | 0.854089 | 0.850307 | 13 | 0.8389 | 5190 | 0.518100 |
| RNN | 1 | 0.847302 | 0.848570 | 15 | 0.8483 | 5190 | 0.496512 |
| RNN | 2 | 0.849678 | 0.849759 | 25 | 0.9173 | 5190 | 0.484619 |
| LSTM | 0 | 0.888022 | 0.888993 | 29 | 0.9442 | 19590 | 1.375927 |
| LSTM | 1 | 0.885986 | 0.885419 | 25 | 0.8822 | 19590 | 1.382717 |
| LSTM | 2 | 0.885646 | 0.885624 | 6 | 0.9122 | 19590 | 1.413192 |
| GRU | 0 | 0.892772 | 0.892228 | 19 | 0.9339 | 14790 | 1.135861 |
| GRU | 1 | 0.888700 | 0.887604 | 5 | 0.8569 | 14790 | 1.137411 |
| GRU | 2 | 0.879878 | 0.879162 | 5 | 0.9440 | 14790 | 1.159769 |

"Best epoch" is the 0-based index stored in `notes` (`best_epoch=`), so 29 means the 30th epoch.

### 3.3 Mean ± sample std (ddof = 1) over 3 seeds

| Model | Test acc (%) | Test macro-F1 | Params (total) | Sec/epoch (mean of per-run) | Val macro-F1 (mean) |
|---|---|---|---|---|---|
| RNN | 85.04 ± 0.34 | 0.8495 ± 0.0009 | 5190 | 0.50 | 0.8682 |
| LSTM | 88.66 ± 0.13 | 0.8867 ± 0.0020 | 19590 | 1.39 | 0.9129 |
| GRU | 88.71 ± 0.66 | 0.8863 ± 0.0066 | 14790 | 1.14 | 0.9116 |

Sources: `har_summary_table.csv` [MEASURED]; recomputed from per-run CSV with ddof=1 [RECOMPUTED]: RNN acc 85.0356 ± 0.3444; LSTM 88.6551 ± 0.1285; GRU 88.7117 ± 0.6591 (percent). F1 recomputed: RNN 0.84955 ± 0.00089; LSTM 0.88668 ± 0.00201; GRU 0.88633 ± 0.00663.

Note the summary table prints `sec/epoch` rounded to 2 decimals (0.5, 1.39, 1.14); the unrounded per-run values are in the CSV.

Parameter breakdown [DERIVED]:
- RNN: recurrent 4800 + head (64·6+6 = 390) = 5190 ✓
- LSTM: 19200 + 390 = 19590 ✓
- GRU: 14400 + 390 = 14790 ✓

### 3.4 Best-model selection and confusion matrix

- Best model chosen by **mean validation macro-F1**: LSTM (0.9129) over GRU (0.9116) over RNN (0.8682). Printed: "Best HAR model (by validation macro-F1): LSTM" (cell 40). [MEASURED]
- Note: on **test** mean accuracy, GRU is 88.71% vs LSTM 88.66% (difference 0.05 pp, well inside the std). On test macro-F1, LSTM 0.8867 vs GRU 0.8863. [MEASURED] The ranking depends on the metric; see conflicts §9.
- Confusion matrix: `figures/har_confusion_best.png` — **sum over 3 seeds of the LSTM test predictions** (title: "LSTM: test confusion matrix (3 seeds summed)"). Total = 8841 = 3 × 2947 [DERIVED from figure cells].

Values read from `figures/har_confusion_best.png` (rows = true, cols = predicted):

| True \ Pred | Walking | Upstairs | Downstairs | Sitting | Standing | Laying | Row total |
|---|---|---|---|---|---|---|---|
| Walking | 1308 | 45 | 133 | 2 | 0 | 0 | 1488 |
| Upstairs | 13 | 1306 | 85 | 8 | 1 | 0 | 1413 |
| Downstairs | 6 | 18 | 1236 | 0 | 0 | 0 | 1260 |
| Sitting | 3 | 27 | 2 | 1174 | 267 | 0 | 1473 |
| Standing | 30 | 1 | 1 | 280 | 1284 | 0 | 1596 |
| Laying | 0 | 81 | 0 | 0 | 0 | 1530 | 1611 |

Row totals match 3 × the per-class test counts [DERIVED]. Diagonal sum = 7838 → 7838 / 8841 = 88.66% [DERIVED; matches LSTM mean test accuracy].

No numeric CSV of this matrix exists; the values above were read from the figure image [MEASURED from figure, read visually — treat as transcription].

### 3.5 Confusion-matrix observations [INTERPRETATION of the table above]
- Largest off-diagonal cells: Standing→Sitting (280), Sitting→Standing (267), Walking→Downstairs (133), Upstairs→Downstairs (85), Laying→Upstairs (81).
- The static-posture confusion (Sitting ↔ Standing) and the walking-direction confusion (Walking/Upstairs/Downstairs) dominate the errors.

### 3.6 Runtime notes
- Per-epoch times are per-run and reflect this CPU run; total wall-clock for Problem 1 is **[MISSING]** (not recorded anywhere in the project).

---

## 4. Adding problem (T = 50, 100, 200, 400)

### 4.1 Configuration (`problem1.ipynb` cells 17, 43)

- Same architecture pattern as HAR, but input d = 2 and output 1: `RecClassifier(kind, 2, 64, 1)` with MSE loss `F.mse_loss(p.squeeze(-1), y)`.
- Hidden H = 64; Adam lr 1e-3; batch 64; clip 1.0; **15 epochs** (`ADD_EPOCHS = 15`); seeds 0, 1, 2; model selection by lowest validation MSE.
- Baseline printed for each T (cell 43): constant prediction 1.0 test MSE.
- "Solved" = test MSE < 0.05 (cell 43).

### 4.2 Baselines [MEASURED, cell 43]

| T | Constant-1.0 baseline test MSE |
|---|---|
| 50 | 0.1754 |
| 100 | 0.1697 |
| 200 | 0.1679 |
| 400 | 0.1527 |

- Brief says "about 0.167" [brief docx, Dataset 3]. Theoretical value for a constant prediction 1.0 against y = U₁ + U₂ (U ~ Uniform[0,1), independent): E[y] = 1 and Var[y] = 2/12 = 1/6 ≈ 0.1667, so the population baseline is 1/6 [DERIVED — my calculation]. The per-T values above differ from 1/6 because they are computed on finite 1000-sample test sets [INTERPRETATION]. Do not present the per-T values as theoretical baselines.

### 4.3 Per-run test MSE, all 36 main runs [MEASURED, `results_problem1.csv` rows 9–44; per-run values match cell 43 printouts to 4 dp]

| Model | T | Seed 0 | Seed 1 | Seed 2 | Solved (of 3) | Best epoch (0-based) per seed |
|---|---|---|---|---|---|---|
| RNN | 50 | 0.174833 | 0.174711 | 0.174891 | 0 | 8, 13, 10 |
| RNN | 100 | 0.170502 | 0.170042 | 0.170104 | 0 | 12, 7, 13 |
| RNN | 200 | 0.168219 | 0.168879 | 0.168448 | 0 | 10, 11, 10 |
| RNN | 400 | 0.153975 | 0.154310 | 0.153310 | 0 | 14, 11, 10 |
| LSTM | 50 | 0.016961 | 0.049495 | 0.010998 | 3 | 14, 14, 14 |
| LSTM | 100 | 0.169427 | 0.169522 | 0.170031 | 0 | 2, 0, 13 |
| LSTM | 200 | 0.168242 | 0.168363 | 0.168227 | 0 | 3, 14, 11 |
| LSTM | 400 | 0.153571 | 0.153677 | 0.153697 | 0 | 7, 11, 8 |
| GRU | 50 | 0.002808 | 0.001501 | 0.002134 | 3 | 14, 14, 12 |
| GRU | 100 | 0.002064 | 0.001906 | 0.001991 | 3 | 14, 14, 14 |
| GRU | 200 | 0.002572 | 0.002751 | 0.004177 | 3 | 14, 14, 13 |
| GRU | 400 | 0.078280 | 0.023869 | 0.021649 | 2 | 14, 14, 13 |

Note the LSTM T=50 seed-1 run (0.049495) is just under the 0.05 threshold.

### 4.4 Mean ± sample std, solved counts, and per-run sec/epoch [RECOMPUTED ddof=1; matches `adding_summary_table.csv`]

| Model | T | Test MSE mean ± std | Solved (of 3) | Baseline MSE (this T) | Mean sec/epoch (per-run mean) | Params |
|---|---|---|---|---|---|---|
| RNN | 50 | 0.174811 ± 0.000092 | 0 | 0.1754 | 0.332 | 4417 |
| RNN | 100 | 0.170216 ± 0.000250 | 0 | 0.1697 | 0.622 | 4417 |
| RNN | 200 | 0.168515 ± 0.000335 | 0 | 0.1679 | 1.226 | 4417 |
| RNN | 400 | 0.153865 ± 0.000509 | 0 | 0.1527 | 2.558 | 4417 |
| LSTM | 50 | 0.025818 ± 0.020720 | 3 | 0.1754 | 0.882 | 17473 |
| LSTM | 100 | 0.169660 ± 0.000325 | 0 | 0.1697 | 1.809 | 17473 |
| LSTM | 200 | 0.168278 ± 0.000075 | 0 | 0.1679 | 3.656 | 17473 |
| LSTM | 400 | 0.153648 ± 0.000068 | 0 | 0.1527 | 7.494 | 17473 |
| GRU | 50 | 0.002148 ± 0.000654 | 3 | 0.1754 | 0.708 | 13121 |
| GRU | 100 | 0.001987 ± 0.000079 | 3 | 0.1697 | 1.459 | 13121 |
| GRU | 200 | 0.003167 ± 0.000880 | 3 | 0.1679 | 3.069 | 13121 |
| GRU | 400 | 0.041266 ± 0.032074 | 2 | 0.1527 | 6.373 | 13121 |

The stored `adding_summary_table.csv` reports these to 4 dp (e.g. "0.0258 ± 0.0207"; "0.0021 ± 0.0007"; "0.0413 ± 0.0321"). Stored `baseline MSE` is rounded to 4 dp.

Parameter counts [DERIVED]: RNN recurrent = 64·2 + 64·64 + 64 + 64 = 4352, + head 65 = **4417** ✓. LSTM 4·64·(2+64) + 2·4·64 = 17408, + 65 = **17473** ✓. GRU 3·64·66 + 2·3·64 = 13056, + 65 = **13121** ✓.

### 4.5 Learning curves at T = 200 (`figures/adding_learning_curves_T200.png`)
- Mean ± std band over 3 seeds, log-scale y-axis; training MSE (left) and validation MSE (right); epochs 1–15.
- Values are not stored numerically. Visual reading: GRU training and validation MSE fall from ~0.16 to ~0.005 (train) / ~0.004 (val) by epoch 15; the GRU curve drops sharply around epochs 8–11. RNN and LSTM curves stay around 0.16–0.17 across all epochs (the two curves overlap heavily in the plot). [INTERPRETATION of plot]

### 4.6 Interpretation supported by the results [INTERPRETATION]
- At every T, GRU reached test MSE far below the baseline (~0.002–0.004 at T ≤ 200), and 2–3 of 3 seeds were solved at each T.
- LSTM is solved at T = 50 (3/3) but is at baseline level at T = 100, 200, 400 with this 15-epoch budget.
- RNN is at baseline level at all T.
- At T = 400, GRU degrades (mean 0.0413, one seed at 0.0783 and two at 0.0239/0.0216), and LSTM and RNN are at baseline.
- These are results of a 15-epoch, H = 64, clip = 1.0 budget. They do not show that LSTM cannot solve T = 100+ with more epochs; the notebook did not test that.

---

## 5. Gradient analysis (Task 5a) and the related plot

### 5.1 Setup (`problem1.ipynb` cell 20 code; markdown cell 19)
- Sequence length **T = 100**.
- Gradient profile = mean over **256 held-out sequences** (`make_adding(256, 100, seed=3)`), i.e. seed 3 — distinct from train/val/test seeds 0/1/2.
- Quantity: ‖∂L/∂h_t‖ (L-2 norm over the hidden dimension, per sequence, then averaged over the 256 sequences) for every h_t, via `retain_grad()` on the states of the from-scratch `Unrolled` model with library weights copied in.
- **Before training:** `INIT_SNAPSHOT[kind]` = the untrained model for seed 0 at T = 100 (deep-copied at construction, cell 17).
- **After training:** the seed-0, T = 100 model from Task 4 (best-validation-epoch weights).
- Loss: MSE on `head(out[:, -1])`.

### 5.2 Measured values
Only the **after-training** summary values were printed (cell 46 output):

| Model | \|grad\| at t=1 / \|grad\| at t=T (after) | \|grad\| at t=T (after) |
|---|---|---|
| RNN | 7.233786e-09 | 0.001519 |
| LSTM | 2.885134e-14 | 0.001679 |
| GRU | 8.331016e-01 | 0.000265 |

[MEASURED]. Derived t=1 values [DERIVED = ratio × t=T value]: RNN ≈ 1.10e-11; LSTM ≈ 4.84e-17; GRU ≈ 2.21e-4.

- **Before-training numeric values are [MISSING]:** the notebook plots them but does not print or save them. Only the figure shows them.

### 5.3 Observations from `figures/grad_norm_vs_t_T100.png` [INTERPRETATION of plot]
- **Before training (left panel):** all three curves are nearly parallel on a log scale, rising from very small values at small t to ~1e-3 at t = 100. The RNN and LSTM curves begin at later t (≈ t=5 and ≈ t=2) because values at earlier steps are below the plotted range or zero; the plot does not show which. GRU begins at t=1 near 1e-21 to 1e-22.
- **After training (right panel):** GRU gradient norm is essentially flat across t (~2e-4 to 1e-3), i.e. information from early steps still receives gradient. RNN rises roughly log-linearly from ~1e-11 at t=1 to ~1e-3 at t=100. LSTM rises from ~5e-17 at t=1 to ~1e-3 at t=100.
- Interpretation: after training, the GRU keeps gradient flow from early time steps, while the RNN and LSTM show strong decay toward early steps (ratio ~7e-9 and ~3e-14). This is consistent with the T=100 results in §4 (GRU solved; RNN and LSTM at baseline).

---

## 6. Hidden-size sweep (Task 5b)

### 6.1 Setup (`problem1.ipynb` cells 47–48)
- Models: LSTM and GRU. HAR dataset. H ∈ {16, 32, 64, 128}.
- **One seed only (seed 0)** (`SWEEP_SEEDS = [0]`, cell 47 comment: "one seed for the sweep to save time").
- H = 64 point is the main-run seed-0 result (not re-trained), per the notebook.
- Same epochs (30), clip 1.0, Adam 1e-3, batch 64.

### 6.2 Results [MEASURED, `hidden_sweep_table.csv`; `results_problem1.csv` rows 45–50 for H ≠ 64]

| Model | H | Params (total) | Test acc (%) seed 0 | Sec/epoch |
|---|---|---|---|---|
| LSTM | 16 | 1830 | 86.12 | 0.62 |
| LSTM | 32 | 5702 | 87.07 | 0.81 |
| LSTM | 64 | 19590 | 88.80 | 1.38 |
| LSTM | 128 | 71942 | 90.26 | 2.27 |
| GRU | 16 | 1398 | 87.44 | 0.66 |
| GRU | 32 | 4326 | 90.30 | 0.76 |
| GRU | 64 | 14790 | 89.28 | 1.14 |
| GRU | 128 | 54150 | 92.06 | 1.85 |

Per-run exact accuracies (fractions): LSTM H16 0.861215, H32 0.870716, H128 0.902613; GRU H16 0.874449, H32 0.902952, H128 0.920597. H=64 values: LSTM 0.888022, GRU 0.892772.

Parameter check [DERIVED]:
- LSTM H=16: 4·16·(9+16) + 2·4·16 = 1600 + 128 = 1728; head 16·6+6 = 102 → 1830 ✓
- LSTM H=128: 4·128·137 + 2·4·128 = 70144 + 1024 = 71168; head 774 → 71942 ✓
- GRU H=16: 3·16·25 + 2·3·16 = 1296; head 102 → 1398 ✓
- GRU H=128: 3·128·137 + 2·3·128 = 53376; head 774 → 54150 ✓

### 6.3 Observations [INTERPRETATION]
- LSTM accuracy increases monotonically with H (86.12 → 87.07 → 88.80 → 90.26) while parameters grow from 1830 to 71942.
- GRU is not monotonic: 87.44 (H16) → 90.30 (H32) → 89.28 (H64) → 92.06 (H128). The H=64 dip is a single-seed result.
- At one seed, differences of ~1 pp are within the 3-seed spread seen in §3 (LSTM std 0.13 pp, GRU std 0.66 pp, RNN std 0.34 pp), but the spread at a single fixed H is not measured here. Treat the sweep as indicative only.
- Runtime grows roughly with H (sec/epoch 0.62 → 2.27 for LSTM; 0.66 → 1.85 for GRU).

---

## 7. Student-specific individual ablation (Task 5c)

### 7.1 Which variant applies
- The brief (`DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx`, Problem 1 Tasks, item (c)): "chosen by the last digit of your student ID: **even = gradient clipping (norm 1.0 versus none) for the vanilla RNN at T = 200; odd = LSTM forget-gate bias initialised to 0 versus 1 at T = 200**."
- Student ID from the project folder: `2026AJ05432` → last digit **2 → even → gradient clipping (RNN)**. [DERIVED from folder name and the cover PDF `/Users/pavansai/Desktop/2026AJ05432.pdf`, which shows "BITS ID: 2026AJ05432".]
- The notebook does not know the ID. Both variants are run (cell 49 markdown). [MEASURED — both are in the CSV.]
- **Caveat:** the brief's "Individualisation" / variants table, referred to in the docx ("see Individualisation"), is **[MISSING] from the brief copy available here** (`/Users/pavansai/Downloads/DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx`, same content as "(1) (1)" copy). The hidden size and seeds for this student are therefore not confirmed by the provided files (§10).

### 7.2 Design (`problem1.ipynb` cell 24)
- T = 200, seeds 0, 1, 2 for each condition, 15 epochs, H = 64, Adam 1e-3, batch 64.
- RNN clip 1.0 = the Task-4 main RNN runs (reused, not retrained). RNN no-clip = new runs with `clip=None`.
- LSTM forget-bias ablation: forget-gate bias set to 0.0 or 1.0 in `bias_ih_l0[H:2H]` with `bias_hh_l0[H:2H] = 0` (effective forget bias = value), via `set_lstm_forget_bias` (cell 13). Seeds 0, 1, 2.

### 7.3 Per-seed test MSE [MEASURED, `results_problem1.csv` rows 27–29, 51–59]

| Condition | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| RNN, clip 1.0 (main run) | 0.168219 | 0.168879 | 0.168448 |
| RNN, no clip | 0.168045 | 0.168445 | 0.168679 |
| LSTM, forget bias 0 | 0.168250 | 0.168369 | 0.168214 |
| LSTM, forget bias 1 | 0.168575 | 0.168462 | 0.168293 |

Note: the LSTM forget-bias runs are not the same as the main LSTM T=200 runs (rows 30–32: 0.168242, 0.168363, 0.168227), which use PyTorch default bias init.

### 7.4 Summary table [MEASURED, `ablation_table.csv`]

| Ablation | Condition | Test MSE (mean ± std) | Solved (of 3) |
|---|---|---|---|
| rnn_clip | clip 1.0 | 0.1685 ± 0.0003 | 0 |
| rnn_clip | no clip | 0.1684 ± 0.0003 | 0 |
| lstm_forget | forget bias 0 | 0.1683 ± 0.0001 | 0 |
| lstm_forget | forget bias 1 | 0.1684 ± 0.0001 | 0 |

### 7.5 Observations [INTERPRETATION]
- No condition is solved (all MSEs ≈ 0.168, baseline at T=200 is 0.1679). Clipping does not change the RNN result measurably at this budget (difference 0.0001, within seed spread).
- The even/clip ablation was **not** the one that produced a solved run: the RNN is at baseline with or without clipping.
- `figures/ablation_T200.png` (boxplots with 3 points each, 0.05 threshold dotted line) shows all four conditions clustered at ~0.168, far above the 0.05 line.
- The notebook does not test whether clipping matters at larger epochs or other learning rates.

---

## 8. All Problem 1 figures

### 8.1 Figures in `problem1/figures/` (the notebook's figure directory)

| # | File | Pixels | What it depicts | Experiment | Factual observations |
|---|---|---|---|---|---|
| 1 | `figures/class_balance.png` | 1280×560 (saved at dpi 160; the figure shows 1280×560 px) | Grouped bars: train (21 subjects) vs test (9 subjects) window counts per activity | §1.5(a); Task 1 | Train: Walking ~1226, Upstairs ~1073, Downstairs ~986, Sitting ~1286, Standing ~1374, Laying ~1407 (matches cell 30 printout). Test: ~496, 471, 420, 491, 532, 537. Train has ~2.5× the test counts per class. |
| 2 | `figures/har_confusion_best.png` | 1024×768 | Confusion matrix (counts) for LSTM, 3 seeds summed | Task 3 | Values in §3.4. Largest errors: Sitting↔Standing (267, 280), Walking→Downstairs (133). |
| 3 | `figures/adding_mse_vs_T.png` | 1040×672 | Test MSE mean ± std (3 seeds) vs T (log x-axis 50–400) for RNN, LSTM, GRU; dashed black = predict-1.0 baseline; dotted grey = 0.05 solved threshold | Task 4 | RNN and LSTM track the baseline for T ≥ 100; LSTM at T=50 is ~0.026 with a large error bar reaching ~0.047; GRU ~0.002 at T=50–200 and ~0.041 at T=400 with error bar down to ~0.01. |
| 4 | `figures/adding_learning_curves_T200.png` | 1760×640 | Train MSE (left) and validation MSE (right) vs epoch 1–15 at T = 200, mean ± std band, log y-axis | Task 4 | GRU drops from ~0.16 to ~0.005 train / ~0.004 val; RNN and LSTM stay ~0.16–0.17 (curves overlap). |
| 5 | `figures/grad_norm_vs_t_T100.png` | 1760×640 | ‖∂L/∂h_t‖ vs t (1–100), log y-axis, before (left) and after (right) training, T=100, mean over 256 sequences | Task 5a | See §5.3. |
| 6 | `figures/hidden_sweep_har.png` | 960×640 | HAR test accuracy (%) vs H ∈ {16,32,64,128}, LSTM and GRU, seed 0 | Task 5b | LSTM monotonic (86.12 → 90.26). GRU 87.44 → 90.30 → 89.28 → 92.06. |
| 7 | `figures/ablation_T200.png` | 1600×640 | Boxplots + points of test MSE for (left) RNN clip 1.0 vs no clip; (right) LSTM forget bias 0 vs 1; T=200; dotted line at 0.05 | Task 5c | All four conditions at ~0.168; no point near 0.05. |

### 8.2 Figures in the project root and the nested folder (classical section, not the RNN section)

| # | File | Pixels | What it depicts | Source | Observations |
|---|---|---|---|---|---|
| 8 | `har_class_balance.png` (root) | 2669×1467 | "HAR Training Class Distribution": bar counts of the **classical 5952-window train split** | `problem1.ipynb` cell 11 (`plt.savefig("har_class_balance.png")`) | Walking ~1012, Upstairs ~870, Downstairs ~800, Sitting ~1035, Standing ~1104, Laying ~1131. |
| 9 | `problem1/figures/class_balance.png` (nested: `problem1/problem1/figures/class_balance.png`) | 2700×1500 | Same title "HAR Training Class Distribution", same 5952-split counts (classical) | `problem1.ipynb` cell 13 (`FIGURES_DIR = Path("problem1/figures")`) | **Different file from #1** (different content and dimensions; different md5). Same classical counts as #8 by visual inspection. |
| 10 | `test_confusion_matrix.png` (root) | 1523×1368 | "SVM - Test Confusion Matrix", labels 0–5, 2947 test samples | `classical_baselines.ipynb` (writes test_confusion_matrix.csv/png) | Diagonal: 442, 415, 394, 383, 473, 510. Largest off-diagonal: 99 (label 3→4 Sitting→Standing), 56 (4→3), 53 (0→2), 48 (1→2). Row totals 496, 471, 420, 491, 532, 537. |

Note on #10: its axis labels are numeric 0–5, while the CSV's activity-name version is in `problem1/artifacts/test_confusion_matrix.csv`.

### 8.3 Figures required by the brief but not stored as PNG
- Validation confusion matrix of the classical SVM (`problem1.ipynb` cell 57): shown inline only, **not saved as a file** [MISSING as a file].

### 8.4 Brief-required Problem 1 figure checklist (see §10 for rubric mapping)
- Class balance — present (#1, plus classical #8/#9).
- Confusion matrix of best HAR model — present (#2).
- Adding learning curves at T=200 — present (#4).
- MSE against T — present (#3).
- Gradient norm against time step — present (#5).
- Hidden-size sweep — present (#6).
- Task 5(c) ablation plot — present (#7).

---

## 9. Implementation details, training configuration, limitations, warnings, conflicts

### 9.1 Training configuration summary

| Item | Value | Source |
|---|---|---|
| Hidden size (main runs) | 64 | cell 1 |
| Seeds | 0, 1, 2 | cell 1 |
| Optimizer / lr | Adam, 1e-3 | cell 1 |
| Batch | 64 | cell 1 |
| Clip | 1.0 (all main runs) | cell 1 |
| HAR epochs | 30 | cell 1 |
| Adding epochs | 15 | cell 1 |
| Model selection | lowest validation loss epoch; test once | `run_model` |
| Device | CPU only (`CUDA available: False`) | cell 2 |
| Torch threads | 4 | cell 1 |
| Random seeding | `random.seed`, `np.random.seed`, `torch.manual_seed` via `set_seed` | cell 1 |
| Data loader | none; manual `randperm` mini-batches | `run_model` |
| Early stopping | none; fixed epochs, best-val checkpoint kept | `run_model` |
| Dropout | none in RNN models | cell 13 |
| Head | single `nn.Linear(h, out)` on last hidden state | cell 13 |

### 9.2 Implementation details worth reporting
- The recurrent classifiers (§3, §4) use **library** `nn.RNN/LSTM/GRU` (allowed by the brief for Task 3–5). The from-scratch cells (§2) are used only in verification and in the gradient profile (§5).
- The gradient profile uses the from-scratch `Unrolled` model with library weights copied in; the notebook asserts this model reproduces the trained library layer within 1e-5 (cell 46).
- `best_epoch` in results is 0-based.
- Results are written to `results_problem1.csv` after each run via `add_result` → `save_results()`.

### 9.3 Limitations and warnings (from files)
1. **Short training budget**: HAR 30 epochs, adding problem 15 epochs (notebook markdown cell 26 says these were chosen "to fit a CPU run"). At T ≥ 100, RNN and LSTM may need more epochs; the notebook does not test this.
2. **Hidden-size sweep uses one seed** (cell 47).
3. **Individual ablation**: 3 seeds, but one T (200) only.
4. **Gradient "before" values are not stored** — only the plot (§5.2).
5. **Total Problem 1 wall-clock is not recorded** [MISSING].
6. **Hidden size and seeds are from the notebook's stated assumptions**, not verified against a variants table (see §10).
7. **Adding-problem baseline values differ from 0.167** (§4.2).
8. **FutureWarning** printed by sklearn during classical SVM training: "The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`." (`problem1.ipynb` cell 20; `classical_baselines.ipynb` cell 39). This is classical code, not the RNN.
9. The HAR results are from **test accuracy on the official test split**, reported once per run at the best-validation epoch. Good.
10. **Classical SVM vs. recurrent** comparison: the notebook cites the classical SVM test accuracy (88.80%) beside the RNN results (cell 41). The two use **different validation splits** (§9.4), so the comparison is not like-for-like in the split.

### 9.4 Conflicts between sources (documented, not resolved)

| # | Conflict | Files / cells | Detail |
|---|---|---|---|
| C1 | **Two different class-count sets** | `problem1/figures/class_balance.png` & cell 30 (7352) vs `har_class_balance.png` & cell 62 & markdown §5.2 (5952) | The markdown §5.2 table in `problem1.ipynb` (cell 63) is labelled "Training Samples" and uses the **5952** counts. The RNN section's class-balance plot uses **7352**. |
| C2 | **Two different validation splits** | RNN section (cell 29: per-seed `default_rng(seed)`) vs classical section (cell 8 / `classical_baselines.ipynb` cell 6: `seed=42`, subjects [16,3,22,23]) | Not the same validation subjects; the classical SVM "validation" is not the RNN validation set. |
| C3 | **SVM validation precision: markdown vs computed** | `problem1.ipynb` cell 63 §5.1 says Precision **0.8820**; `validation_results.csv`, `problem1/artifacts/validation_model_comparison.csv`, and cell 55/`classical_baselines` cell 41 say **0.8861** (0.8860829…). | Markdown value does not match the stored output. Other SVM validation values match (Accuracy 0.8807, AUC 0.9862 / 0.9861707…, Recall 0.8807, F1 0.8812, MCC 0.8572). |
| C4 | **Best HAR model depends on metric** | Cell 40 selects LSTM by validation macro-F1 (0.9129 vs GRU 0.9116). Test mean accuracy: GRU 88.71 vs LSTM 88.66. Test macro-F1: LSTM 0.8867 vs GRU 0.8863. | Differences are inside one std. Report both. |
| C5 | **Artifact named "best_har_model" is not a recurrent model** | `problem1/problem1/artifacts/best_har_model.joblib`, cell 61 | It stores the classical **SVM** (`best_model` = SVM from Part C). The recurrent best model is not saved to disk. Root `best_svm_model.pkl` (27.9 MB) is also the SVM from `classical_baselines.ipynb`. |
| C6 | **Working directory conflict** | `problem1/README.md` and cell 1 markdown say run from `problem1/`; cells 13 and 61 write to `problem1/figures` and `problem1/artifacts` (paths relative to the project root). | This created `problem1/problem1/figures/` and `problem1/problem1/artifacts/`. The root `problem1/` also contains `figures/` (RNN outputs). |
| C7 | **README file list vs actual files** | `problem1/README.md` "Project Structure" lists `class_balance.png` at the top level of `problem1/`; actual `class_balance.png` is in `problem1/figures/`. The README does not list `figures/`, `har_class_balance.png`, `har_summary_table.csv` (listed), `adding_summary_table.csv`, `hidden_sweep_table.csv`, `ablation_table.csv`, `best_svm_model.pkl`, or `problem1/problem1/`. | README is out of date relative to the outputs. |
| C8 | **README results claim vs HAR table** | `problem1/README.md` "Results": "gated recurrent architectures outperform the vanilla RNN" — consistent with §3.3. "GRU performs better than vanilla RNN and LSTM under several tested sequence lengths" — consistent with §4.4 for T=50–200 vs RNN, but at T=50 LSTM is also solved (3/3) and GRU mean is lower. The README does not state the HAR LSTM vs GRU ordering. | Not a numeric contradiction; the README wording is qualitative and should be checked against the table before use. |
| C9 | **Notebook class-count label** | `problem1.ipynb` cell 63 §5.2 labels the table "Training Samples"; the same counts are "train" in `problem1/artifacts` context. | Counts are from the classical split (C1). |
| C10 | **`results_problem1.csv` mixes run types** | Rows 0–44 main; rows 45–50 sweep (seq_len=128 labelled); rows 51–59 ablation (seq_len=200, same model/seed keys as main rows). | Grouping by `model`/`seq_len` mixes ablation/sweep rows with main rows. Filter by `notes` (contains `sweep` or `ablation=`). |

### 9.5 Notebook execution and reproducibility status
- `problem1.ipynb`: 41 code cells; 39 have saved outputs. The two without outputs were **not identified in this pass** [UNCERTAIN].
- Restart-and-run-all status: **not verified** here (the notebook was not run; no kernel was used).
- Relative paths: RNN section uses `data/har/...` and `figures/` relative to `problem1/`; classical section and cells 13, 61 use `problem1/...` relative to the project root (C6).
- Random seeds are set; CPU runs. Reproducibility claim is "fixed seeds" per notebook.

### 9.6 Other files present (not part of the brief's Problem 1 deliverable list)
- `problem1/data/problem1.ipynb` and `problem1/data/problem1_continuation_clean.ipynb` (template and continuation with no outputs).
- `problem1/data/har.zip` and `problem1/data/har/__MACOSX/` (macOS archive artifacts).
- `problem1/data/har/UCI HAR Dataset.names`, `UCI HAR Dataset.zip`.
- `problem1/classical_baselines.ipynb` (classical ML; not requested by the Problem 1 brief).
- `problem1/verifyproblem1.py` (asserts classical 6×6 confusion matrix sums to 2947; not run in this pass).
- `problem1/best_svm_model.pkl` (27.9 MB) — the brief's zip excludes checkpoints; see submission note.

### 9.7 Incomplete or missing items
- **MISSING**: Problem 1 total runtime (only per-epoch seconds exist).
- **MISSING**: numeric before-training gradient values (§5.2).
- **MISSING**: saved file for the classical validation confusion matrix (§8.3).
- **MISSING**: saved HAR confusion matrix as CSV (only PNG; values in §3.4 were read from the image).
- **MISSING**: the "one-page analysis" (brief Task 5d) as text in the repo. The notebook markdown has task descriptions but no RNN analysis text (the only analysis markdown is for the classical SVM in cell 63).
- **MISSING**: wearable recommendation text (brief P1.5).
- **MISSING**: variants table (see §10).

---

## 10. Requirement-by-requirement checklist (Problem 1 rubric)

Rubric source: `DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx` ("Marking rubric (100 points)" table) and the Problem 1 task text. Points as listed in the brief. Status values: COMPLETE / MISSING / UNCERTAIN.

| Rubric item | Pts | Status | Evidence | Notes |
|---|---|---|---|---|
| **P1.1 Data pipelines** — HAR subject-wise split | 5 | COMPLETE | `problem1.ipynb` cell 29 (`make_har_data`), per-seed val subjects, asserts (no leakage) | §1.2 |
| P1.1 — per-channel mean/std from training part only | | COMPLETE | cell 29 (`X_tr_all[tr].mean(...)`) | |
| P1.1 — class-balance plot | | COMPLETE | `figures/class_balance.png` (cell 30) | Two different class-balance plots exist (C1) |
| P1.1 — adding generator + test (two markers, one per half, target = sum) | | COMPLETE | cell 6 generator; cell 14 and cell 32 `test_adding_generator` asserts; output "passed" | |
| **P1.2 Cells from scratch** — all three match library within 1e-5 | 10 | COMPLETE | cell 9 output: max diffs 2.68e-7 (RNN), 5.96e-8 (LSTM), 9.69e-8 (GRU) | Random batch, eval mode; also re-checked on trained weights in cell 46 (assert) |
| P1.2 — parameter counts derived by hand (4,736 / 18,944 / 14,208) | | COMPLETE | cell 10 markdown, cell 11 asserts | |
| P1.2 — explain PyTorch 4,800 / 19,200 / 14,400 (two bias vectors) | | COMPLETE | cell 10 markdown; cell 37 output | |
| **P1.3 HAR classification** — same budget for all three | 10 | COMPLETE | cell 1 constants; §3.1 | |
| P1.3 — 3 seeds, mean ± std | | COMPLETE | §3.3 | |
| P1.3 — test accuracy and macro-F1 | | COMPLETE | §3.2–3.3 | |
| P1.3 — parameters and seconds per epoch | | COMPLETE | §3.2 (params, sec/epoch) | |
| P1.3 — confusion matrix of the best model | | COMPLETE (with caveat) | `figures/har_confusion_best.png` (LSTM, 3 seeds summed) | Best by val macro-F1 = LSTM (C4). No CSV of values (§3.4 values read from image) |
| P1.3 — same hidden size as "student's variant" | | UNCERTAIN | Notebook uses H=64; brief's worked example uses h=64; variants table not in provided brief | §10 note below |
| **P1.4 Adding problem** — T = 50, 100, 200, 400 × 3 models × 3 seeds | 10 | COMPLETE | `results_problem1.csv` rows 9–44 (36 rows) | |
| P1.4 — MSE vs 0.167 baseline | | COMPLETE | §4.2 (per-T baselines 0.1754/0.1697/0.1679/0.1527) | Baselines differ from 0.167 (sample variation); theoretical 1/6 (§4.2) |
| P1.4 — learning curves at T = 200 | | COMPLETE | `figures/adding_learning_curves_T200.png` | |
| P1.4 — solved (MSE < 0.05) counts per model and length | | COMPLETE | §4.4; `adding_summary_table.csv` | |
| **P1.5 Gradients, ablations and analysis** — (a) gradient-norm plot before/after, T=100, log scale | 15 | COMPLETE (plot) / PARTIAL (values) | `figures/grad_norm_vs_t_T100.png`; after-training summary values in cell 46 | Before-training numbers not stored (§5.2) |
| P1.5 — (b) hidden-size sweep {16,32,64,128} for LSTM and GRU on HAR | | COMPLETE (single seed) | `hidden_sweep_table.csv`, `figures/hidden_sweep_har.png` | Single seed (seed 0) — rubric does not state seed count for sweep; flagged |
| P1.5 — (c) individual ablation (student variant) | | COMPLETE | Even ID → RNN clipping; `ablation_table.csv`, `figures/ablation_T200.png` | Student ID last digit = 2 (even). Both variants also run (§7) |
| P1.5 — (d) one-page analysis: memory, failure with T, accuracy–params–time trade-off, wearable recommendation | | **MISSING** | No analysis text for Problem 1 RNN section in any file | Notebook markdown has only task descriptions; README "Results" is qualitative only |
| P1.5 — wearable recommendation justified | | **MISSING** | none | |
| Conclusions "backed by tables" | | **MISSING** (in repo) | Analysis text not written | Will be needed for report.pdf |
| **Figure checklist** — class balance; confusion matrix; adding curves at T=200; MSE vs T; gradient norm vs t; hidden sweep; ablation | | COMPLETE | §8.1, §8.4 | |
| **Code** — notebook runs top to bottom with Restart and Run All | | UNCERTAIN | 39/41 code cells have outputs; not re-run (rule: do not rerun) | §9.5 |
| Code — unit tests are assert cells placed after the function | | COMPLETE | cell 6→14 and 32; cells 29 (asserts), 9, 11, 46 | |
| Code — relative paths only | | UNCERTAIN / partial | Mixed bases (C6) | §9.5 |
| Code — comments explain tensor shapes in LSTM and attention code | | COMPLETE (LSTM) | cell 8 comments: `(B, h)`, `(4h, d)` etc. | Attention is Problem 2 (out of scope) |
| **Results file** — `results.csv` at root with P1+P2 rows, exact columns | | MISSING (root) | Root `results.csv` does not exist; `problem1/results_problem1.csv` has the required 12 columns; `problem2/results.csv` exists (P2) | Deliverable-level merge not done (per work state) |
| `problem1/` folder structure in zip: `problem1.ipynb`, `figures/` | | COMPLETE | present | |
| `README.md` (≤1 page; environment, how to run, total run time on T4, variant, what fails) | | UNCERTAIN / PARTIAL | `problem1/README.md` exists (untracked); root `README.md` deleted in git status; no run-time record; variant not stated | §9.3, C7 |
| `requirements.txt` with exact versions | | COMPLETE | root `requirements.txt` pins torch/numpy/pandas/matplotlib/scikit-learn | |
| `report.pdf` | | MISSING | Not created (per instruction) | |
| AI-use statement | | MISSING | Not in repo | Needed in report |
| Student variant (hidden size, width, seeds) | | UNCERTAIN | Variants table absent from brief copy; notebook assumes H=64, seeds 0,1,2 | See note |

**Note on the variant (UNCERTAIN):** the brief says hidden size, width and seeds come from a variants table "under Individualisation", and the FAQ says "Look up the last digit of your student ID". That table is **not** in the docx copies available here (both copies are identical, 20,762 bytes of extracted text, no "Individualisation" section found). The Desktop PDF `2026AJ05432.pdf` is the Steel-Plate ML assignment (different course) and the Downloads "Assignment 2 Questions" PDFs are the optimization assignment, so neither supplies the variants table. The brief's own worked example uses h = 64 (parameter counts) and the hidden-size sweep set {16,32,64,128} includes 64, so H = 64 is consistent with the text, but the student-specific value must be confirmed from the instructor's variants table.

### 10.1 Summary counts (Problem 1 rubric)
- COMPLETE: data pipelines, cells from scratch, HAR, adding problem, gradient plot, sweep, ablation, figures list, requirements.txt, folder structure.
- MISSING: P1.5(d) one-page analysis and wearable recommendation (in repo), root `results.csv` (merged), `report.pdf`, AI-use statement, total Problem 1 runtime, before-training gradient numbers, variants table.
- UNCERTAIN: variant (hidden size/seeds), Restart-and-Run-All status, relative-path consistency, README accuracy, the two non-output code cells.

---

## 11. Quick reference: all Problem 1 numbers in one place

**HAR (H=64, 30 epochs, 3 seeds):**
- RNN: acc 85.04 ± 0.34 %, macro-F1 0.8495 ± 0.0009, params 5190, 0.50 s/epoch
- LSTM: acc 88.66 ± 0.13 %, macro-F1 0.8867 ± 0.0020, params 19590, 1.39 s/epoch
- GRU: acc 88.71 ± 0.66 %, macro-F1 0.8863 ± 0.0066, params 14790, 1.14 s/epoch

**Adding (H=64, 15 epochs, 3 seeds; test MSE mean ± std; solved of 3):**
- T=50: RNN 0.1748 ± 0.0001 (0); LSTM 0.0258 ± 0.0207 (3); GRU 0.0021 ± 0.0007 (3); baseline 0.1754
- T=100: RNN 0.1702 ± 0.0003 (0); LSTM 0.1697 ± 0.0003 (0); GRU 0.0020 ± 0.0001 (3); baseline 0.1697
- T=200: RNN 0.1685 ± 0.0003 (0); LSTM 0.1683 ± 0.0001 (0); GRU 0.0032 ± 0.0009 (3); baseline 0.1679
- T=400: RNN 0.1539 ± 0.0005 (0); LSTM 0.1536 ± 0.0001 (0); GRU 0.0413 ± 0.0321 (2); baseline 0.1527

**Cells:** max |diff| vs nn.* 2.68e-7 (RNN), 5.96e-8 (LSTM), 9.69e-8 (GRU) on random batch; params 4800/19200/14400 (PyTorch) vs 4736/18944/14208 (one-bias hand count).

**Classical SVM (for reference, not the recurrent comparison):** validation acc 0.8807, MCC 0.8572; test acc 0.8880, macro AUC 0.9885, weighted F1 0.8886, MCC 0.8663 (`test_results.csv`).

---

## 12. Corrections to the notebook markdown (for the report author)

- `problem1.ipynb` cell 63 §5.1: SVM validation **Precision** should read **0.8861** per stored output (not 0.8820) if the table is reused.
- `problem1.ipynb` cell 63 §5.2: the class counts are for the **5952-window classical train split**, not the full training set; label accordingly.
- `problem1/README.md` project-structure list is out of date (C7).
- `best_har_model.joblib` contains the **SVM**, not a recurrent model (C5).
