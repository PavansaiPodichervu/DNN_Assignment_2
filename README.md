# DNN Programming Assignment 2: RNN, LSTM, GRU and Transformer

Student: Pavansai Podichervu, ID 2026AJ05432

## What is here

- `problem1/`: RNN, LSTM and GRU cells from scratch, checked against PyTorch within 1e-5. Compared on UCI HAR (6 classes) and the adding problem (T = 50, 100, 200, 400). Includes gradient analysis, a hidden-size sweep and the student-specific ablation.
- `problem2/`: scaled dot-product attention, multi-head attention and a pre-LayerNorm encoder block from scratch. Used for an encoder classifier on HAR and a character-level causal LM on Tiny Shakespeare.
- `results.csv`: all 72 runs (60 Problem 1, 12 Problem 2), 12 columns as specified. Problem 2's `problem2/results.csv` is canonical for Problem 2.
- `report.pdf`: the final report (6 pages, excluding the references and AI-use statement).

## Setup

Python 3.13 with the pinned packages in `requirements.txt` (torch 2.15.0, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1). The notebook tools (nbformat, nbconvert, ipykernel) are needed only to execute notebooks. `problem2/requirements.txt` has the same pins.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- UCI HAR: `https://archive.ics.uci.edu/dataset/240` (zip `human+activity+recognition+using+smartphones.zip`). The Problem 1 notebook downloads and extracts it to `problem1/data/har/UCI HAR Dataset/`. Problem 2 reads the same folder via `../problem1/data/har/`.
- Tiny Shakespeare: downloaded by the Problem 2 notebook to `problem2/data/tinyshakespeare.txt` if absent.
- Adding problem: generated in code (`make_adding(n, T, seed)`), no download.

## Run order

1. **Problem 1:** `cd problem1`, then run `problem1.ipynb` from top to bottom. Run from `problem1/`: the notebook writes `figures/`, `artifacts/` and `results_problem1.csv` relative to that folder.
2. **Problem 2:** `cd problem2`, then run `problem2.ipynb` from top to bottom. It needs the HAR data from step 1.

Outputs used in the report: `results.csv` (root, combined), `problem1/results_problem1.csv`, `problem1/figures/`, `problem2/results.csv`, `problem2/figures/`, `problem2/generated_samples.txt`, `problem2/generated_samples_nomask.txt`. Model checkpoints `problem2/transformer_lm.pt` and `problem2/har_transformer.pt` are kept outside the submission folder.

Not used: `problem2/results_rows.csv` (an earlier timing run whose origin is not recorded; kept as evidence only), and `problem2/problem2_results.csv` (a summary written by the same notebook, duplicated in `results.csv`).

## Hardware and runtime

- CPU only (`CUDA available: False`). CPU model, core count and RAM were **not recorded**.
- Torch threads: 4 (Problem 1), 6 (Problem 2).
- Recorded seconds per epoch (mean over 3 seeds, H = 64): RNN 0.50, LSTM 1.39, GRU 1.14 (HAR). Problem 2 HAR Transformer about 7.9 to 8.3.
- Recorded wall-clock: Problem 2 LM 10.1 min (3000 steps). Total run time for each notebook was **not recorded**, and no T4 GPU run was performed.
- Peak memory was **not measured** (CPU only).

## Measured results (canonical CSVs)

| Model | Test accuracy | Source |
|---|---|---|
| RNN, H = 64, 3 seeds | 85.04 ± 0.34 % | `problem1/results_problem1.csv` (H = 64 main runs only) |
| LSTM, H = 64, 3 seeds | 88.66 ± 0.13 % | same |
| GRU, H = 64, 3 seeds | 88.71 ± 0.66 % | same |
| Transformer, no PE, seed 42 | 90.57 % | `problem2/results.csv` |
| Transformer, sinusoidal, seed 42 | 88.53 % | same |
| Transformer, learned, seed 42 | 89.21 % | same |

The sinusoidal model is the default and scores 88.53 %. The 90.57 % figure belongs to the model with **no** positional encoding. Problem 1 comparison values exclude the hidden-size sweep rows, which were contaminated earlier. Char LM: validation 1.6777 nats/char = 2.4204 bits/char.

## Notebook checks performed

Cells that were edited in this revision were executed in a scratch copy with the saved checkpoint (no training). Their outputs in the notebook come from that run. The full notebooks were **not** re-run top to bottom in one session. See `Prompt2_Verification.md` for the exact list.

## Known limitations

- Every Transformer result uses one seed (42). Differences of one or two points on HAR are not established against seed variation.
- Transformer HAR (15 epochs, AdamW, final epoch) and Problem 1 (30 epochs, Adam, best validation) are not budget-matched.
- LM ablations use 800 steps; the default uses 3000 steps.
- Memory at context lengths 64 to 512 was not measured.
- The instructor's three-token attention example is not in the project; the self-created example is used and labelled as such.
- The brief asks for 500 generated characters per sample. The samples now contain 500 generated characters, with the prompt character excluded. The no-mask sample was edited in text only (its model was not saved, so it cannot be regenerated without retraining).
- The attention figure shows the first two heads of layer 1, as the brief requires.
- `problem1/problem1/` holds outputs from an earlier run started from the project root (the path bug is fixed in the notebook). It is not part of the submission.
- Problem 1 cell 61 still shows the old printed path in its saved output until the notebook is re-run.
- The old `problem2/README.md` contained wrong positional-encoding claims. It is archived in `archive_prompt2/`.
