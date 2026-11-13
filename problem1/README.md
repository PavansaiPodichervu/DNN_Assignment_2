# Problem 1 — Recurrent Neural Networks from Scratch

## Overview

This problem implements and evaluates recurrent neural network architectures for sequence modeling and Human Activity Recognition (HAR).

The implementation includes vanilla RNN, LSTM, and GRU cells built from scratch using PyTorch tensor operations. The recurrent implementations are verified against PyTorch reference implementations and evaluated on sequence classification and synthetic sequence-learning tasks.

## Objectives

- Implement vanilla RNN, LSTM, and GRU recurrent cells from scratch.
- Verify the custom recurrent implementations against PyTorch reference implementations.
- Evaluate recurrent architectures on the Human Activity Recognition dataset.
- Compare model performance using accuracy, macro-F1, parameter counts, and training time.
- Investigate recurrent model behavior on the adding problem.
- Analyze sequence length, hidden size, gradient behavior, and model ablations.

## Dataset

### UCI Human Activity Recognition (HAR)

The project uses the UCI Human Activity Recognition Using Smartphones dataset.

The dataset contains smartphone sensor measurements collected from participants performing six activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

The official training and test partitions are retained for evaluation.

### Adding Problem

The adding problem is a synthetic sequence-learning task used to investigate how recurrent architectures learn dependencies across time.

## Implemented Models

- Vanilla RNN
- LSTM
- GRU

All three recurrent architectures are implemented from scratch and evaluated under controlled experimental settings.

## Experiments

The notebook includes:

- Recurrent cell verification against PyTorch reference implementations.
- HAR classification using RNN, LSTM, and GRU.
- Multiple-seed experiments.
- Sequence-length experiments.
- Hidden-size experiments.
- Gradient analysis.
- Adding-problem experiments.
- Model ablations.
- Model comparison using evaluation metrics and computational cost.

## Evaluation Metrics

The experiments use:

- Accuracy
- Macro-F1
- Parameter count
- Training time

Macro-F1 is included to evaluate class-wise performance rather than relying only on overall accuracy.

## Results

The HAR experiments show that gated recurrent architectures outperform the vanilla RNN in the evaluated configuration.

The adding-problem experiments demonstrate differences in learning long-range dependencies across recurrent architectures. GRU performs better than vanilla RNN and LSTM under several tested sequence lengths and training configurations.

Detailed experimental results, comparisons, and visualizations are available in `problem1.ipynb` and the exported result files.

## Project Structure

```text
problem1/
├── problem1.ipynb
├── classical_baselines.ipynb
├── data/
│   └── har/
│       └── UCI HAR Dataset/
├── results_problem1.csv
├── test_results.csv
├── validation_results.csv
├── test_confusion_matrix.csv
├── test_confusion_matrix.png
├── figures/class_balance.png   (other figures are also in figures/)
├── har_summary_table.csv
└── README.md
```

Additional model checkpoints or experiment outputs may be present depending on execution.

## Reproducibility

1. Install the dependencies listed in the repository's `requirements.txt`.
2. Ensure the UCI HAR dataset is available in the expected `data/har/` directory.
3. Open `problem1.ipynb`.
4. Restart the kernel and execute the notebook from top to bottom.
5. Review the generated metrics, tables, and figures.

## Notes

- Experiments use fixed random seeds where applicable.
- The official HAR test split is retained for final evaluation.
- Results depend on the experimental configuration and computational environment.
- Detailed implementation and experimental analysis are documented in the notebook.
