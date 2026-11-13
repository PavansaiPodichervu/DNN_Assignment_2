# Problem 2 Verification Notes

Companion to `DNN_Assignment2_Report_Draft.docx` (Problem 1 content preserved from `Problem1_Report.docx`; Problem 2 appended). Verified against `problem2/problem2.ipynb` (cell outputs), `problem2/results.csv`, `problem2/results_rows.csv`, `problem2/problem2_results.csv`, `problem2/generated_samples*.txt`, `problem2/figures/*.png`, `problem2/README.md`, and `problem1/results_problem1.csv`. No experiments were rerun.

Assignment source: `DNN Programming Assignment – RNN, LSTM, GRU and Transformer (1).docx`, Problem 2 section and rubric. (The Desktop PDF `2026AJ05432.pdf` is an unrelated ML assignment.)

## 1. Requirements: completed and verified

| Requirement (rubric) | Status | Evidence |
|---|---|---|
| P2.1 Scaled dot-product attention with optional mask | COMPLETE | Notebook cell 5. Signature `scaled_dot_product_attention(Q,K,V,mask,...)`; masked scores set to dtype min before softmax |
| P2.1 Row sums = 1 | COMPLETE | Cell 5 assert (atol 1e-6) |
| P2.1 Match `F.scaled_dot_product_attention` within 1e-5 | COMPLETE | Cell 5 assert (atol 1e-5); passed |
| P2.1 Causal mask gives zero weight to future positions | COMPLETE | Cell 6 assert; passed |
| P2.1 Hand-checkable 3-token example, by hand and by code | COMPLETE (self-chosen example) | Cell 8 (code vs loop implementation, atol 1e-6); cell 9 (manual). Weights and contexts agree within rounding (see §3) |
| P2.1 Instructor-supplied 3-token example | MISSING | The brief says the instructor supplies one. The notebook uses a self-chosen example; README §5 gives the same matrices without naming their source. See §4 |
| P2.2 Multi-head attention with W_Q, W_K, W_V, W_O and head reshaping | COMPLETE | Notebook cell 11 `MultiHeadAttention`; head split to (B, heads, L, d_head) |
| P2.2 Pre-LN encoder block, FFN width 4d, residuals, dropout | COMPLETE | Cell 11 `EncoderBlock`: LayerNorm before attention and FFN, GELU FFN 128→512→128, dropout 0.1 |
| P2.2 Sinusoidal positional encoding | COMPLETE | Cell 11 `SinusoidalPositionalEncoding` |
| P2.2 Weight count 4d² + 2·d·d_ff; match `sum(p.numel())` | COMPLETE | Cell 13 assert: 198,272 per block for d = 128, d_ff = 512; weights 196,608 + biases 1,152 + LayerNorm 512 |
| P2.2 Correct shapes | UNCERTAIN | Shape comments exist; no explicit shape assertions found in cell 11 |
| P2.3 Embed each time step (or patch), CLS token, classify with encoder | COMPLETE | Cell 17 `HARTransformer`: per-step linear embedding (9→128), CLS prepended, classification from CLS |
| P2.3 Reuse Problem 1 data | COMPLETE (with split caveat) | Reads `../problem1/data/har/UCI HAR Dataset`; uses classical subject split (seed 42), not the per-seed splits used in Problem 1 (see §4) |
| P2.3 Compare accuracy, parameters, training time with best Problem 1 model | COMPLETE (with caveats) | `problem2/README.md` §9.1 and notebook cell 23 contain the comparison, but the Problem 1 values in them are wrong (see §4). Correct comparison in report §4 |
| P2.3 Ablate positional info: none, sinusoidal, learned | COMPLETE | Cell 19; `results.csv` rows 1–3 |
| P2.4 Tiny Shakespeare, context 128, d = 128, 4 heads, 4 layers | COMPLETE | Cells 26–30. Context 128, d 128, heads 4, layers 4, d_ff 512, 801,664 params |
| P2.4 65 characters; 90/10 split, no shuffling | COMPLETE | Cell 26: 65 chars (asserted); train 1,003,854 tokens; validation 111,540 tokens |
| P2.4 Validation loss and bits per character | COMPLETE | Cell 30: 1.6777 nats/char, 2.4204 bits/char. bits = nats / ln 2 (checked: 1.6777 / 0.693147 = 2.4204) |
| P2.4 Learning curve | COMPLETE | `figures/lm_learning_curve.png` (cell 31) |
| P2.4 Generate 500 characters at temperatures 0.2, 0.8, 1.5 with top-k | COMPLETE (501 characters) | Cell 36, top-k = 40. `generated_samples.txt` has three labelled samples of 501 characters each (start space plus 500 generated). The brief asks for 500, so this is a one-character deviation |
| P2.4 Describe the differences between temperatures | MISSING (in repo) | No written description in the notebook or README. Qualitative description in report Table 8 is my interpretation |
| P2.5(a) Heatmaps of two heads on one sentence, with interpretation | COMPLETE (four heads) | Cell 40; `figures/attention_heads.png` shows four heads of layer 1 for "To be, or not to be". Interpretation in notebook cell 39 is generic |
| P2.5(b) Heads = 1, 4, 8 at fixed d | COMPLETE | Cell 33; table in notebook §7.2 matches cell output |
| P2.5(c) Remove causal mask: training loss and generation | COMPLETE | Cell 33 (no-mask training), cell 37 (no-mask sample); `figures/lm_ablations.png`; `generated_samples_nomask.txt` |
| P2.5(d) Time and memory at context 64, 128, 256, 512; relate to O(n²d) | PARTIAL | Time: cell 43 and `figures/context_timing.png`. Memory: **not measured** (n/a on CPU). See §4 |
| Bonus: KV cache or LSTM-vs-Transformer character model | NOT ATTEMPTED | Not in notebook or files |
| Deliverable: `results.csv` with exact 12 columns | COMPLETE | `problem2/results.csv` header: problem, model, seed, seq_len, test_acc, macro_f1, test_mse, params, sec_per_epoch, val_loss_nats, bits_per_char, notes |
| Deliverable: `generated_samples.txt` with three labelled samples | COMPLETE (length caveat) | See above |
| Deliverable: learning curves, heatmaps, heads ablation, no-mask result, time/memory vs context | COMPLETE except memory | Figures 8–12 in the draft |
| Deliverable: README at most one page with total T4 runtime and variant | UNCERTAIN / PARTIAL | `problem2/README.md` is 951 lines and is not one page. It states CPU runs only; no T4 run. Its §9 numbers are wrong (see §4) |
| Report: AI-use statement | MISSING | Not in repo |
| Report: references | MISSING | Not in repo |

## 2. Experiment results: measured values (verified against cell outputs)

**Positional-encoding ablation, HAR Transformer (one seed, 42; 15 epochs; final-epoch weights):**

| Variant | Params | Val acc | Test acc | Test macro-F1 | Sec/epoch |
|---|---|---|---|---|---|
| none | 398,982 | 0.9661 | 0.905667 | 0.903828 | 7.91 |
| sinusoidal | 398,982 | 0.9572 | 0.885307 | 0.882884 | 7.94 |
| learned | 415,494 | 0.9467 | 0.892094 | 0.889066 | 8.32 |

Source: cell 19 printed lines and `results.csv` rows 1–3. [Matches.]

**Language model (Tiny Shakespeare):** 1.677718 nats/char; 2.420436 bits/char (cell 30; `problem2_results.csv`). [Matches.]

**Heads and no-mask ablations (800 steps each, seed 42):** heads 1: 2.445098 nats; heads 4: 2.285424; heads 8: 2.274659; no mask (heads 4): 6.579494 nats (9.492203 bits). [Matches cell 33 and `results.csv` rows 5–8.]

**Timing (CPU, batch 64, 20 timed steps):** cell 43 output 0.073502 / 0.179742 / 0.483277 / 1.922594 s/step for n = 64 / 128 / 256 / 512. `results.csv` rows 9–12 match these to 4 dp. [Matches.]

## 3. Hand example: verification details

The printed weights and contexts (cell 8) are:
- Row 2 weights [0.3302, 0.6698, 0]; context [2.3395, 3.3395]. Hand calculation in cell 9 and report: [2.3396, 3.3396].
- Row 3 weights [0.2483, 0.2483, 0.5035]; context [3.5105, 4.5105]. Hand calculation: [3.5107, 4.5107].

The 0.0001–0.0002 differences come from the notebook printing weights to 4 dp and recomputing from rounded values. The code's own check is against an independent loop at atol 1e-6, so the code result is the reference. [INTERPRETATION]

## 4. Conflicts, errors and missing evidence

### 4.1 Conflicts between sources

1. **Problem 2 README §9 misattributes 0.9057.** It states "The sinusoidal Transformer achieved … Test accuracy: 0.9057". The CSV and notebook show 0.9057 is the **no-positional-encoding** model. Sinusoidal is **0.8853**. The report uses the CSV values.
2. **Problem 1 comparison values in the notebook (cell 23) and README §9.1 are contaminated.** The code takes `results_problem1.csv` rows with `seq_len == 128` and model in {RNN, LSTM, GRU}, which includes the hidden-size sweep rows (seed 0, H = 16, 32, 128). Results:
   - Notebook/README LSTM 0.8824 (params "≈ 23,041"), GRU 0.8932 (params 17,374) are wrong.
   - Correct 3-seed H = 64 means: LSTM 88.66% (params 19,590), GRU 88.71% (params 14,790), RNN 85.04% (params 5,190).
   - I reproduced the notebook's contaminated numbers from the CSV: GRU mean over 6 rows = 0.893225, which matches cell 23's output exactly.
3. **Claimed "+1.25 point" Transformer advantage is not supported.** It was computed as 0.9057 (no-PE model) minus the contaminated GRU value. Against the correct GRU mean (88.71%), the no-PE model is +1.86 points and the default sinusoidal model is −0.18 points. Both comparisons rest on one Transformer seed.
4. **Timing values disagree.** `results_rows.csv` and notebook §7.4 markdown give 0.0770 / 0.1848 / 0.4961 / 1.7814 s/step. The live cell 43 output and `results.csv` give 0.0735 / 0.1797 / 0.4833 / 1.9226. The report uses the cell-43 output. The source of the other set is not recorded. [NEEDS VERIFICATION]
5. **Sample length.** The brief asks for 500 characters. The files contain 501 (start space + 500 generated). The notebook label says "500 characters".
6. **Ablation budget vs default.** The default LM is 3000 steps (1.6777 nats). The head-count and no-mask ablations are 800 steps (heads 4 = 2.2854 nats). The notebook says so, but `results.csv` lists both under the same model family, so the table can mislead.
7. **Problem 1 vs Problem 2 HAR protocol.** Problem 1: 30 epochs, Adam, best-validation-loss checkpoint, per-seed validation subjects, 3 seeds. Problem 2: 15 epochs, AdamW lr 1e-3, final-epoch weights, classical split (seed 42), 1 seed. The comparison in the report flags this.

### 4.2 Missing or unverified evidence

- **Instructor's hand example:** not in the project. [MISSING]
- **Memory measurement** for context lengths 64–512: not measured (CPU, no CUDA). [MISSING]
- **Written description of temperature differences** (P2.4): none in the repo. [MISSING]
- **Written heatmap interpretation** (P2.5a): notebook cell 39 is generic; the interpretation in the report is mine, based on the figure. [MISSING in repo; INTERPRETATION in report]
- **Multiple seeds** for any Problem 2 experiment: all single seed 42. [LIMITATION]
- **Notebook run-top-to-bottom:** not verified in this session. Cell execution order matters (the HAR split uses the Problem 1 folder; the LM uses `data/tinyshakespeare.txt`, downloaded by the notebook). [UNCERTAIN]
- **`results.csv` vs `results_rows.csv` vs `problem2_results.csv`:** three result files, two of which disagree on timing. The notebook writes `results_rows.csv` and `problem2_results.csv`; `results.csv` matches the live output. [NEEDS VERIFICATION which is canonical]
- **HAR Transformer per-seed runs:** not performed. The comparison with Problem 1's 3-seed means is therefore one seed against three. [LIMITATION]

## 5. Figures inserted in the draft (Problem 2)

| Figure | File | Shows | Source cell |
|---|---|---|---|
| 8 | `problem2/figures/har_learning_curve.png` | Train and validation cross-entropy per epoch, default (sinusoidal) HAR Transformer, 15 epochs | 21 |
| 9 | `problem2/figures/lm_learning_curve.png` | Train (window mean) and validation (fixed windows) nats/char per step, 3000 steps | 31 |
| 10 | `problem2/figures/attention_heads.png` | Four heads of layer 1, "To be, or not to be" | 40 |
| 11 | `problem2/figures/lm_ablations.png` | Bits/char for heads 1, 4, 8 and no-mask; train loss causal vs no-mask | 34 |
| 12 | `problem2/figures/context_timing.png` | Sec/step vs context length (log2 x-axis), CPU | 43 |

Tables inserted: Problem 2 Tables 6–11 (positional ablation; HAR confusion matrix; temperature samples; heads ablation; mask ablation; timing). Problem 1 Tables 1–5 are unchanged.

Figures not inserted: `problem2/figures/*` none omitted. Model checkpoints (`*.pt`), `problem2/data/tinyshakespeare.txt`, and the `generated_samples*.txt` text files are not figures.

## 6. Problem 1 content

Problem 1 content in the draft is unchanged from `Problem1_Report.docx` (rebuilt from `Problem1_Report.md` with a figure-caption fix only: each figure now has one caption instead of two). Problem 1's own verification items are in `Problem1_Report_Verification.md`.

## 7. Assumptions requiring your confirmation

1. The **sinusoidal** model is the default Problem 2 HAR result in the report, because the notebook uses it for the confusion matrix, learning curve and saved checkpoint. The no-PE model is reported alongside it because it scored higher on test and validation.
2. The **timing values from cell 43** (0.0735 … 1.9226) are used in the report, not the `results_rows.csv`/§7.4 values.
3. Problem 1 comparison values use the **3-seed H = 64** means, not the contaminated notebook values.
4. The **501-character** samples are reported as found, with the brief's 500-character requirement flagged.
5. The page budget is not verified. The combined report exceeds 6 pages and will be condensed later, as you asked.
