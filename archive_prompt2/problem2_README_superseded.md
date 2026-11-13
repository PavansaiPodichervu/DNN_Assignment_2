
# Problem 2: Transformer Implementation and Applications

## Overview

This problem focuses on understanding and implementing the Transformer architecture from scratch using PyTorch.

The implementation demonstrates how self-attention, causal masking, positional encoding, and feed-forward networks work together to process sequential data.

Two applications are explored:

1. Human Activity Recognition (HAR) using Transformer Encoder models.
2. Tiny Shakespeare character-level language modeling using a causal Transformer.

The assignment also includes attention verification, visualization of attention patterns, context-length timing experiments, model evaluation, and artifact generation.

---

# 1. Objectives

The main objectives of Problem 2 are:

- Understand the mathematical foundations of scaled dot-product attention.
- Implement attention mechanisms using PyTorch.
- Understand causal attention masking.
- Verify attention calculations using a hand-checkable example.
- Implement Transformer components from scratch.
- Apply Transformers to Human Activity Recognition.
- Apply Transformers to character-level language modeling.
- Compare Transformer performance with recurrent models from Problem 1.
- Analyze attention patterns using attention maps.
- Investigate the computational cost of increasing context length.
- Save trained models, evaluation results, and visualization artifacts.

---

# 2. Transformer Architecture

## 2.1 Introduction

A Transformer is a neural network architecture that processes sequences using attention mechanisms rather than relying entirely on recurrent computation.

Unlike traditional RNNs, Transformers can process multiple positions in a sequence simultaneously during training.

The Transformer architecture consists of several important components:

- Input embeddings
- Positional encoding
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Output projection

These components work together to learn relationships between different positions in a sequence.

## 2.2 Why Transformers Are Used

Traditional recurrent neural networks process sequential information step by step.

This can make training difficult for long sequences because information must pass through multiple recurrent steps.

Transformers use attention to directly connect different positions in a sequence.

This allows the model to learn relationships between distant positions more effectively.

---

# 3. Scaled Dot-Product Attention

## 3.1 Definition

Scaled dot-product attention determines how much attention each query should give to different keys.

The attention weights are then used to calculate a weighted combination of value vectors.

The attention calculation consists of:

1. Computing similarity scores between queries and keys.
2. Scaling the scores.
3. Applying an attention mask when required.
4. Applying softmax to obtain attention weights.
5. Multiplying the attention weights by the value matrix.

## 3.2 Attention Formula

Attention(Q, K, V) = Softmax((Q × Kᵀ) / sqrt(d_k) + M) × V

Where:

- Q: Query matrix
- K: Key matrix
- V: Value matrix
- Kᵀ: Transpose of the key matrix
- d_k: Dimension of each key vector
- M: Attention mask
- Softmax: Converts attention scores into probabilities.

The scaling factor is 1 / sqrt(d_k).

It prevents attention scores from becoming excessively large as the key dimension increases.

## 3.3 Query, Key, and Value

Query:
Represents what a token is looking for.

Key:
Represents information used to determine whether a token is relevant to a query.

Value:
Contains the information that is combined using attention weights.

The query-key similarity determines the attention weights, while the values determine the resulting context representation.

---

# 4. Causal Attention Masking

## 4.1 Definition

Causal masking prevents a token from attending to future tokens.

This is important in autoregressive language modeling, where the model predicts the next token using only previously available information.

For example:

- Token 1 can attend only to Token 1.
- Token 2 can attend to Tokens 1 and 2.
- Token 3 can attend to Tokens 1, 2, and 3.

## 4.2 Causal Mask

A three-token causal mask is:

    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]

The value 1 indicates that attention is allowed.

The value 0 indicates that attention is blocked.

In the implementation, a lower-triangular Boolean mask is created using:

    torch.tril(torch.ones(L, L, dtype=torch.bool))

Masked attention positions receive an extremely negative score before softmax.

This makes their attention probabilities approximately zero.

## 4.3 Why Causal Masking Is Necessary

Without causal masking, a language model could access future tokens during training.

This would create information leakage and make next-token prediction invalid.

Causal masking ensures that the model learns to predict the next token using only the preceding context.

---

# 5. Hand-Checkable Attention Example

## 5.1 Purpose

A small three-token example is used to verify the scaled dot-product attention implementation.

The example compares the output of the implemented attention function against an independent loop-based calculation.

## 5.2 Input Matrices

Query and Key matrices:

    Q = K =

    [[1, 0],
     [0, 1],
     [1, 1]]

Value matrix:

    V =

    [[1, 2],
     [3, 4],
     [5, 6]]

Key dimension:

    d_k = 2

Scaling factor:

    1 / sqrt(2) = 0.7071

## 5.3 Token 1

Token 1 can attend only to itself.

Attention score:

    [0]

Softmax weight:

    [1]

Context output:

    [1, 2]

## 5.4 Token 2

Token 2 can attend to Tokens 1 and 2.

Attention scores:

    [0, 0.7071]

Softmax weights:

    [0.3302, 0.6698]

Context output:

    0.3302 × [1, 2] + 0.6698 × [3, 4]

    = [2.3396, 3.3396]

## 5.5 Token 3

Token 3 can attend to all three tokens.

Attention scores:

    [0.7071, 0.7071, 1.4142]

Softmax weights:

    [0.2483, 0.2483, 0.5035]

Context output:

    0.2483 × [1, 2]
    + 0.2483 × [3, 4]
    + 0.5035 × [5, 6]

    = [3.5107, 4.5107]

## 5.6 Verification

The implementation is independently checked using a loop-based calculation.

The loop:

- Processes one query at a time.
- Uses only the allowed causal prefix.
- Computes attention scores independently.
- Applies softmax.
- Calculates the context vector.

The implementation is verified using torch.allclose with a numerical tolerance of 1e-6.

The verification checks both:

- Attention weights
- Context outputs

Expected verification message:

    Hand-example code check passed.

---

# 6. Multi-Head Attention

## 6.1 Definition

Multi-head attention allows a Transformer to learn several attention patterns simultaneously.

Instead of calculating attention using a single set of query, key, and value vectors, the model divides the representation into multiple heads.

Each head performs its own attention calculation.

The outputs are then combined through a final projection.

## 6.2 Why Multiple Heads Are Used

Different heads can learn different relationships between tokens.

For example:

- One head may focus on nearby tokens.
- Another may focus on earlier tokens.
- Another may distribute attention across several positions.

These are possible learned patterns, not guaranteed semantic roles.

## 6.3 Implementation

The implementation includes:

- Query projection
- Key projection
- Value projection
- Head splitting
- Scaled dot-product attention
- Attention masking
- Head concatenation
- Output projection

The attention heads operate in parallel.

---

# 7. Positional Encoding

## 7.1 Why Positional Information Is Needed

Self-attention alone does not inherently encode the order of tokens.

Without positional information, the model has difficulty distinguishing sequences that contain the same tokens in different positions.

Positional encoding introduces information about token positions.

## 7.2 Sinusoidal Positional Encoding

The implementation uses sinusoidal positional encoding.

It creates position-dependent values using sine and cosine functions at different frequencies.

These positional values are added to the token embeddings.

The resulting representation contains both:

- Token information
- Positional information

## 7.3 Advantages

Sinusoidal positional encoding:

- Does not require learned positional parameters.
- Provides deterministic position representations.
- Can be calculated directly from position indices.
- Introduces positional information into the Transformer input.

---

# 8. Transformer Encoder for Human Activity Recognition

## 8.1 Dataset

The Human Activity Recognition experiment uses the UCI HAR Dataset.

The dataset contains smartphone sensor measurements collected during human activities.

The sensor data includes:

- Accelerometer measurements
- Gyroscope measurements

The input sequences contain:

- 128 time steps
- 9 sensor features per time step

Original dataset shapes:

    Training: (7352, 128, 9)
    Test:    (2947, 128, 9)

## 8.2 Data Preparation

The dataset is loaded and normalized.

The original training subjects are separated into training and validation subjects.

The subject-wise split is:

Training subjects:

    [1, 5, 6, 7, 8, 11, 14, 15,
     17, 19, 21, 25, 26, 27, 28, 29, 30]

Validation subjects:

    [3, 16, 22, 23]

Test subjects:

    [2, 4, 9, 10, 12, 13, 18, 20, 24]

The validation and test subjects are excluded from training.

The subject separation helps evaluate generalization to unseen participants.

## 8.3 Input Representation

Each HAR sample is represented as a sequence of sensor measurements.

Input shape:

    (batch_size, 128, 9)

The Transformer processes the sequence using sensor-feature projection and positional information.

The final representation is used for activity classification.

## 8.4 Model Components

The HAR Transformer includes:

- Input projection
- Sinusoidal positional encoding
- Transformer encoder blocks
- Multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Classification head

The classification head predicts one of the human activity classes.

## 8.5 Evaluation Metrics

The HAR Transformer is evaluated using:

- Test accuracy
- Macro-F1 score

Accuracy measures the fraction of correctly classified samples.

Macro-F1 calculates the F1 score independently for each class and averages the scores equally.

Macro-F1 is useful when class performance needs to be considered independently of class frequency.

---

# 9. HAR Transformer Results

The sinusoidal Transformer achieved the following test accuracy:

    Test accuracy: 0.9057

This corresponds to approximately 90.57% test accuracy.

The macro-F1 score should be taken from the saved evaluation results.

## 9.1 Comparison with Problem 1

The recurrent HAR models from Problem 1 were compared with the Transformer using sequence length 128.

Results are averaged over seeds.

| Model | Test Accuracy | Macro-F1 | Parameters |
|---|---:|---:|---:|
| RNN | 0.8504 | 0.8495 | 5,190 |
| LSTM | 0.8824 | 0.8821 | Approximately 23,041 |
| GRU | 0.8932 | 0.8922 | 17,374 |
| Transformer | 0.9057 | Check saved results | Check saved results |

The GRU was the best recurrent model.

The Transformer achieved higher test accuracy than the best recurrent model.

Accuracy difference:

    0.9057 - 0.8932 = 0.0125

The Transformer achieved approximately 1.25 percentage points higher test accuracy.

This comparison indicates that the Transformer performed better than the best recurrent model in this experiment.

The result should be interpreted within the specific dataset, architecture, training configuration, and evaluation protocol.

---

# 10. Tiny Shakespeare Character-Level Language Modeling

## 10.1 Objective

The Tiny Shakespeare experiment demonstrates autoregressive language modeling using a Transformer.

The model learns character-level patterns from Shakespeare text.

Instead of predicting complete words, the model predicts the next character.

## 10.2 Dataset

The dataset is stored in:

    data/tinyshakespeare.txt

The text is converted into a character-level vocabulary.

Each character is assigned an integer ID.

The model learns to predict the next character based on the preceding context.

## 10.3 Character-Level Encoding

The vocabulary contains unique characters from the training text.

Encoding converts characters into integers.

Decoding converts predicted integer IDs back into characters.

This allows the model to generate readable text character by character.

## 10.4 Training Objective

The model performs next-character prediction.

Given a context of characters, the model predicts the next character.

The training objective is cross-entropy loss.

The model is trained to assign higher probability to the correct next character.

## 10.5 Model Architecture

The language model includes:

- Token embeddings
- Positional encoding
- Transformer blocks
- Causal self-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Language-model output projection

The output projection produces logits for every character in the vocabulary.

## 10.6 Autoregressive Generation

Generation begins with an initial context.

The model predicts the next character.

The predicted character is appended to the context.

The updated context is used to predict the following character.

This process continues until the required number of characters has been generated.

The model generates text using learned statistical patterns rather than retrieving original Shakespeare passages.

---

# 11. Tiny Shakespeare Results

The experiment evaluates the language model using validation loss.

The validation loss is converted into two metrics:

Validation Nats per Character:

    validation_nats_per_character = validation_loss

Validation Bits per Character:

    validation_bits_per_character =
        validation_loss / ln(2)

These metrics describe the model's predictive performance.

Lower values indicate better predictive performance.

Generated text is saved for qualitative inspection.

The generated samples should be interpreted as outputs of a learned character-level language model.

They may contain grammatical patterns, recognizable words, and Shakespeare-like structures, but they are not guaranteed to reproduce coherent original passages.

---

# 12. Attention Maps

## 12.1 Objective

Attention maps visualize how each attention head distributes attention across token positions.

They help inspect the attention patterns learned by the Transformer.

## 12.2 Interpretation

The horizontal axis represents key positions.

The vertical axis represents query positions.

Each cell represents the attention weight assigned to a key position by a query position.

Because causal masking is used, a query at position t can attend only to positions 0 through t.

Therefore, the region above the main diagonal should contain zero attention weights.

## 12.3 Different Attention Heads

Different heads may learn different attention patterns.

Some may emphasize nearby tokens.

Others may distribute attention across earlier positions.

The heatmaps provide qualitative evidence of these patterns.

The exact function of an attention head cannot be established from its heatmap alone.

Attention maps should be interpreted together with model performance and other experiments.

## 12.4 Saved Figure

    figures/attention_heads.png

---

# 13. Context-Length Timing

## 13.1 Objective

The context-length experiment investigates how increasing the sequence length affects Transformer training time.

The tested context lengths are:

    64
    128
    256
    512

The experiment measures average training-step time.

## 13.2 Experimental Setup

The timing procedure includes:

- Batch size: 64
- One warm-up training step
- 20 timed training steps
- AdamW optimizer
- Learning rate: 3e-4

The reported training time is averaged across the timed steps.

## 13.3 Timing Metric

Seconds per training step:

    sec_per_step = elapsed_time / number_of_steps

The experiment records:

- Context length
- Seconds per training step
- Parameter count
- Peak CUDA memory, when available

The machine used for this experiment runs on CPU.

Therefore, CUDA peak memory is reported as unavailable.

## 13.4 Computational Complexity

The attention score matrix has dimensions:

    context_length × context_length

For each attention head, the attention score computation grows approximately quadratically with context length.

Attention score computation:

    O(n² × d)

Where:

- n = context length
- d = embedding dimension

The feed-forward and projection operations grow approximately linearly with context length for a fixed model configuration.

Their complexity is approximately:

    O(n × d²)

The overall computational cost increases as context length increases.

## 13.5 Interpretation

Increasing context length allows the model to process more preceding tokens.

However, it also increases computational requirements.

The attention score matrix becomes larger as context length increases.

This leads to increased computation and memory requirements.

The timing results provide an empirical view of this behavior on the CPU.

## 13.6 Saved Figure

    figures/context_timing.png

---

# 14. Model Ablation Experiments

The language-model experiment includes ablation comparisons.

Ablation experiments investigate how changing one component affects model performance.

The comparisons include:

- Transformer configuration changes
- Positional encoding comparisons
- Model configuration variations

The results are saved for analysis.

## 14.1 Ablation Figure

    figures/lm_ablations.png

The ablation results should be interpreted according to the exact configurations and metrics recorded in the experiment.

---

# 15. Learning Curves

Learning curves visualize the change in training and validation performance during training.

They help identify:

- Whether training is progressing.
- Whether validation performance improves.
- Whether overfitting occurs.
- Whether additional training provides meaningful improvement.

## 15.1 HAR Learning Curve

    figures/har_learning_curve.png

## 15.2 Language Model Learning Curve

    figures/lm_learning_curve.png

These figures support the analysis of training behavior.

---

# 16. Saved Artifacts

The Problem 2 directory contains the following important files.

## 16.1 Model Checkpoints

    transformer_lm.pt
    har_transformer.pt

The language-model checkpoint stores:

- Model state dictionary
- Vocabulary
- Model configuration

The HAR checkpoint stores the trained HAR Transformer model parameters.

## 16.2 Results

    problem2_results.csv
    results_rows.csv
    results.csv

These files contain experiment results and metric records.

The exact columns should be inspected from the saved CSV files.

## 16.3 Generated Samples

    generated_samples.txt
    generated_samples_nomask.txt

These files contain generated language-model samples.

The masked and unmasked generation results can be compared qualitatively.

## 16.4 Figures

    figures/attention_heads.png
    figures/context_timing.png
    figures/har_learning_curve.png
    figures/lm_ablations.png
    figures/lm_learning_curve.png

These figures support the analysis and interpretation of the experiments.

---

# 17. Project Structure

    problem2/
    |
    |-- data/
    |   |-- tinyshakespeare.txt
    |
    |-- figures/
    |   |-- attention_heads.png
    |   |-- context_timing.png
    |   |-- har_learning_curve.png
    |   |-- lm_ablations.png
    |   |-- lm_learning_curve.png
    |
    |-- generated_samples.txt
    |-- generated_samples_nomask.txt
    |-- har_transformer.pt
    |-- lm_ablations.png
    |-- lm_learning_curve.png
    |-- problem2.ipynb
    |-- problem2_results.csv
    |-- README.md
    |-- requirements.txt
    |-- results.csv
    |-- results_rows.csv
    |-- transformer_lm.pt

---

# 18. Reproducibility

The experiment uses fixed random seeds where specified.

The training configuration includes:

- Model architecture
- Training steps
- Learning rate
- Context length
- Random seed

Reproducibility depends on the complete configuration and computational environment.

Different hardware, software versions, and execution environments may produce small differences in numerical results.

The notebook should be executed in the intended order so that model definitions, datasets, training functions, and evaluation functions are available before their use.

---

# 19. Limitations

The experiments have several limitations.

## HAR

- The evaluation is based on a specific dataset.
- Performance depends on the train-validation-test subject split.
- A higher test accuracy does not automatically establish universal superiority over recurrent models.
- Results depend on architecture and training configuration.

## Tiny Shakespeare

- Character-level generation does not directly model complete words.
- Generated text may contain spelling errors and grammatical inconsistencies.
- A low validation loss does not guarantee coherent generated text.
- The dataset is relatively small compared with modern language-model training corpora.

## Attention Maps

- Attention weights are useful for visualization.
- They do not independently prove the semantic function of a head.
- Attention patterns should be interpreted with other experimental evidence.

## Timing

- CPU timing depends on the hardware and execution environment.
- Timing results are not directly comparable across machines.
- CUDA memory measurements are unavailable in the CPU experiment.

---

# 20. Key Takeaways

- Transformers use attention to model relationships between sequence positions.
- Scaled dot-product attention combines queries, keys, and values.
- Causal masking prevents access to future tokens.
- Multi-head attention allows multiple attention patterns to be learned.
- Positional encoding introduces sequence-order information.
- Transformers can be applied to sensor-based activity classification.
- Transformers can perform autoregressive character-level language modeling.
- Attention maps help visualize learned attention patterns.
- Increasing context length increases computational requirements.
- Model performance must be evaluated using appropriate metrics.
- Saved checkpoints and results support reproducibility and further analysis.

---

# 21. Quick Revision

## Short Definition

A Transformer is a neural network architecture that uses attention mechanisms to process sequences and learn relationships between different positions.

## Brief Recap

Scaled dot-product attention calculates query-key similarity, applies scaling and masking, converts scores into probabilities using softmax, and combines value vectors.

Causal masking ensures that each token can access only the current and preceding tokens.

The assignment applies Transformers to HAR classification and Tiny Shakespeare character-level language modeling.

## Revision Questions

### Basic

Q: What is the purpose of positional encoding?

A: It introduces information about the position of tokens in a sequence.

### Technical

Q: Why is scaled dot-product attention divided by the square root of the key dimension?

A: Scaling helps prevent attention scores from becoming excessively large, which can make softmax distributions overly concentrated.

### Practical

Q: Why is causal masking required for autoregressive language modeling?

A: It prevents the model from accessing future tokens during training and generation.

---

# 22. Related Concepts

- [[Neural Networks]]
- [[Deep Learning]]
- [[Recurrent Neural Networks]]
- [[RNN]]
- [[LSTM]]
- [[GRU]]
- [[Attention Mechanism]]
- [[Scaled Dot-Product Attention]]
- [[Multi-Head Attention]]
- [[Causal Masking]]
- [[Positional Encoding]]
- [[Transformer Architecture]]
- [[Language Modeling]]
- [[Human Activity Recognition]]
- [[Cross-Entropy Loss]]
- [[Model Evaluation]]
- [[Ablation Study]]
- [[Autoregressive Generation]]
