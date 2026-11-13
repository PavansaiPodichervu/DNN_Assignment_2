# DNN Programming Assignment-2: RNN, LSTM, GRU and Transformer

Student: Pavansai Podichervu (ID 2026AJ05432)

## Overview

- **Problem 1 (`problem1/`):** RNN, LSTM and GRU cells written from scratch and checked against PyTorch within 1e-5. Compared on UCI HAR (6-class sensor classification) and on the adding problem (T = 50, 100, 200, 400). Includes a gradient analysis, hidden-size sweep and a student-specific ablation.
- **Problem 2 (`problem2/`):** Scaled dot-product attention, multi-head attention and a pre-LayerNorm encoder block written from scratch. Used for an encoder classifier on HAR and a character-level causal language model on Tiny Shakespeare.

## Environment and setup

- Python: the notebooks were run with Python 3.13 from the project `.venv` (path shown in notebook output).
- Problem 1 pinned versions (`requirements.txt`, project root): torch 2.15.0, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-learn 1.9.1. Notebook-only: nbformat 5.11.1, nbconvert 7.17.2, ipykernel 7.4.0.
- Problem 2 (`problem2/requirements.txt`): torch>=2.0, numpy, pandas, matplotlib, scikit-learn (not pinned; the run used torch 2.15.0 as printed in notebook output).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

- **UCI HAR:** the Problem 1 notebook downloads `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip` into `problem1/data/` and extracts it to `problem1/data/har/UCI HAR Dataset/`. Problem 2 reads the same folder through `../problem1/data/har/`.
- **Tiny Shakespeare:** the Problem 2 notebook downloads `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` into `problem2/data/tinyshakespeare.txt` if it is absent.
- **Adding problem:** generated in code by `make_adding(n, T, seed)`. No download.
- No pre-trained weights are used.

## How to run

1. **Problem 1:** `cd problem1`, then open `problem1.ipynb`, restart the kernel and run all cells in order. Run from the `problem1/` folder. Two classical-baseline cells write to `problem1/figures` and `problem1/artifacts` relative to the working directory; running from the project root creates nested `problem1/problem1/` folders (see Known limitations).
2. **Problem 2:** `cd problem2`, then open `problem2.ipynb` and run all cells in order. The HAR data must be present at `../problem1/data/har/UCI HAR Dataset/`.

Outputs: Problem 1 writes `results_problem1.csv`, the summary tables and `figures/`. Problem 2 writes `results.csv`, `problem2_results.csv`, `results_rows.csv`, `generated_samples*.txt`, `figures/` and model checkpoints (`transformer_lm.pt`, `har_transformer.pt`).

## Hardware and runtime

- Hardware used: CPU only. The notebooks report `CUDA available: False`. CPU model, core count and memory were **not recorded** in the project files.
- Torch threads: 4 (Problem 1), 6 (Problem 2).
- Recorded times (seconds per epoch unless stated): Problem 1 HAR RNN 0.50, LSTM 1.39, GRU 1.14; adding problem 0.3 to 7.5 depending on model and T; Problem 2 HAR Transformer about 7.9 to 8.3; Problem 2 language model 10.1 minutes for 3000 steps; timing run 0.07 to 1.92 s per step for context 64 to 512.
- Total wall-clock time for each notebook was **not recorded**.
- Peak memory: not measured (no CUDA).

## Main experiment configuration

| Item | Problem 1 | Problem 2 |
|---|---|---|
| Hidden / model width | H = 64 | d = 128 |
| Optimizer, lr | Adam, 1e-3 | HAR: AdamW, 1e-3; LM: AdamW, 3e-4 |
| Batch | 64 | HAR: 128; LM: 64 windows of 128 |
| Epochs / steps | HAR 30; adding 15 | HAR 15; LM 3000 steps (ablations 800) |
| Gradient clipping | norm 1.0 | norm 1.0 |
| Seeds | 0, 1, 2 (sweep and ablation partly seed 0) | 42 (all runs) |
| Model selection | lowest validation loss epoch | HAR: final epoch; LM: final step |
| Layers / heads | 1 recurrent layer (library layer) | HAR: 2 layers, 4 heads; LM: 4 layers, 4 heads |

## Reproducibility notes

- Seeds are fixed in code (`random`, `numpy`, `torch`). CPU results may differ across machines and library versions.
- Problem 1 uses the library `nn.RNN`/`nn.LSTM`/`nn.GRU` only for comparisons; the from-scratch cells are verified against them.
- Problem 1 validation subjects are drawn per seed with `np.random.default_rng(seed)`. The classical baseline in Problem 1 uses a different split (seed 42).
- Results tables should be read with the notes in `results_problem1.csv` (sweep and ablation rows share model and sequence-length keys with main runs).

## Known limitations

- Single seed for all Problem 2 results and for the Problem 1 hidden-size sweep.
- Short training budgets (HAR 15 or 30 epochs, LM 3000 steps); results may change with longer training.
- Generated Tiny Shakespeare samples are 501 characters (start character plus 500 generated); the brief asks for 500.
- Context-length memory was not measured; only time was measured.
- The instructor-supplied 3-token attention example is not in the project; a self-chosen example is used.
- Some notebook cells write output paths relative to the working directory, which can create nested `problem1/problem1/` folders.
- The Problem 2 README in `problem2/README.md` contains stale claims and should not be used as a source for results (see the audit checklist).
- The root `README.md` was deleted in the working tree; this file replaces it once renamed to `README.md`.
