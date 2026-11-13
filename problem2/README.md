# Problem 2: Transformer from scratch (folder note)

The full setup, data, run order, measured results and limitations are in the root `README.md`. This file only lists the contents of this folder.

- `problem2.ipynb`: final notebook (Tasks 1 and 2, HAR encoder classifier, character LM, ablations, timing, discussion).
- `results.csv`: canonical Problem 2 results (12 rows, written by notebook cell 45).
- `problem2_results.csv`: summary written by the same cell (duplicates part of `results.csv`).
- `results_rows.csv`: an earlier timing run whose origin is not recorded. Not used in the report.
- `generated_samples.txt`: three samples at temperatures 0.2, 0.8 and 1.5, top-k 40, 500 generated characters each (prompt character excluded).
- `generated_samples_nomask.txt`: no-causal-mask sample at temperature 0.8. The text was edited to exclude the prompt character; its model was not saved, so it cannot be regenerated.
- `figures/`: attention heatmaps (two heads), HAR and LM learning curves, ablations, timing.
- `transformer_lm.pt`, `har_transformer.pt`: checkpoints (not part of the submission).
- `requirements.txt`: pinned versions, same as the root file.
- `data/`: Tiny Shakespeare text, downloaded on first run.

The HAR data are read from `../problem1/data/har/UCI HAR Dataset/`.
