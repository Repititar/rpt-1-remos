# RPT-1 Remos — Architecture

**Project:** Repititar  
**Model:** RPT-1 Remos  
**Architecture:** Decoder-only Transformer  
**Framework:** PyTorch  
**Tokenizer:** UTF-8 byte-level tokenizer  
**Status:** Research and development

---

## 1. Overview

RPT-1 Remos is an independently developed language model research
project under Repititar.

The project explores the implementation, training, evaluation, and
deployment of a language model built from randomly initialized
parameters rather than relying on pretrained model weights.

The initial implementation focuses on a compact decoder-only
Transformer that can be developed and tested using standard Python
and PyTorch tooling.

The primary objectives are:

- Implement a readable and maintainable Transformer architecture.
- Train model parameters from scratch using text data.
- Explore causal language modeling and autoregressive generation.
- Evaluate model performance using reproducible metrics.
- Support CPU-based development and compatible accelerators.
- Publish source code and, when available, model weights.
- Establish a foundation for future architecture and training research.

RPT-1 Remos is a research project. Its capabilities must be measured
through training experiments and evaluation rather than assumed from
its architecture.

## 2. System Architecture

The system is organized into five primary components:

1. Configuration
2. Tokenization
3. Model architecture
4. Training and checkpoint management
5. Inference and evaluation

The general data flow is:

```text
                Raw UTF-8 Text
                      |
                      v
              Byte-Level Tokenizer
                      |
                      v
                 Token IDs
                      |
                      v
              Input/Target Batches
                      |
                      v
              Token Embeddings
                      |
                      v
          Decoder Transformer Blocks
          +-------------------------+
          | RMSNorm                 |
          | Causal Self-Attention   |
          | Rotary Position Encoding|
          | Residual Connection     |
          | RMSNorm                 |
          | Feed-Forward Network    |
          | Residual Connection     |
          +-------------------------+
                      |
                      v
                  Final RMSNorm
                      |
                      v
              Language Model Head
                      |
                      v
                Vocabulary Logits
                      |
             +--------+--------+
             |                 |
             v                 v
          Training          Generation
             |                 |
             v                 v
        Loss and Update     Next Token
                               |
                               v
                         Repeated Sampling
```

Training and inference share the same underlying model architecture.
Training updates parameters, while inference uses the learned
parameters to predict subsequent tokens.

## 3. Configuration

Model settings are managed by `ModelConfig` in:

`src/repititar/config.py`

The configuration controls the dimensions and numerical settings
used to construct the model.

The initial default configuration is:

| Parameter | Default | Description |
|---|---:|---|
| `vocab_size` | 256 | Number of possible byte-token IDs |
| `d_model` | 256 | Hidden representation dimension |
| `n_layers` | 6 | Number of Transformer blocks |
| `n_heads` | 8 | Number of attention heads |
| `max_seq_len` | 512 | Maximum input context length |
| `dropout` | 0.1 | Dropout probability |
| `rope_theta` | 10000.0 | RoPE frequency base |
| `norm_eps` | 1e-5 | Numerical stability constant for RMSNorm |

These values are starting points for research, not claims of optimal
performance.

The configuration is stored in:

`configs/default.json`

Configuration validation should reject incompatible dimensions and
invalid numerical parameters before model construction.

For standard multi-head attention, `d_model` must be divisible by
`n_heads`.

The attention head dimension is:

\[
d_{\text{head}} = \frac{d_{\text{model}}}{n_{\text{heads}}}
\]

## 4. Tokenization

Implementation:

`src/repititar/tokenizer.py`

RPT-1 Remos uses a byte-level tokenizer.

Text is encoded as UTF-8, and each byte becomes an integer token
between 0 and 255.

The vocabulary size is therefore:

\[
V = 256
\]

For example, an ASCII character is represented by one UTF-8 byte,
while many non-ASCII characters require multiple bytes.

### Advantages

- No vocabulary training is required.
- Every valid UTF-8 input can be represented as bytes.
- The tokenizer is deterministic.
- Tokenization and detokenization are straightforward.
- The model can operate without an external tokenizer dependency.

### Limitations

Byte-level tokenization can require substantially more tokens than
subword tokenization for the same text.

This increases the sequence length required to represent a document
and may reduce the amount of semantic context available within a
fixed context window.

The tokenizer does not currently define dedicated beginning-of-
sequence or end-of-sequence tokens.

Generated bytes must be decoded as UTF-8. If generation ends in the
middle of a multibyte character, the decoded output may contain
replacement characters.

## 5. Model Architecture

Implementation:

`src/repititar/model.py`

RPT-1 Remos uses a decoder-only Transformer.

The model predicts the next token from the tokens that precede it.
It does not use a separate encoder or an encoder-decoder architecture.

### 5.1 Token Embeddings

The embedding layer maps each token ID to a learned vector:

\[
E \in \mathbb{R}^{V \times d_{\text{model}}}
\]

Each input token is represented by its corresponding row in the
embedding matrix.

### 5.2 Rotary Positional Embeddings

Rotary Positional Embeddings (RoPE) encode relative positional
information by rotating query and key representations according to
their positions.

RoPE is applied within self-attention.

Its frequency base is controlled by `rope_theta`.

The implementation must use compatible frequency, sine, cosine,
and rotation conventions. Mathematical tests should verify that
rotations preserve vector norms and that positional encoding behaves
consistently across supported sequence lengths.

### 5.3 Causal Self-Attention

Each Transformer block contains multi-head causal self-attention.

The query, key, and value representations are derived from the
input hidden states.

Attention computes relationships between token positions while
preventing a position from attending to future positions.

The scaled dot-product attention operation is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_{\text{head}}}}
+
M
\right)V
\]

Here, \(M\) represents the causal mask.

The mask prevents information from future tokens from influencing
the prediction at an earlier position.

The implementation uses PyTorch's scaled dot-product attention
operation where supported by the current code.

### 5.4 RMSNorm

RMSNorm normalizes hidden representations using their root mean
square.

For an input vector \(x\), a simplified form is:

\[
\operatorname{RMSNorm}(x)
=
\frac{x}
{\sqrt{\operatorname{mean}(x^2)+\epsilon}}
\odot \gamma
\]

Here:

- \(\epsilon\) is a small numerical stability constant.
- \(\gamma\) is a learned scale vector.
- \(\odot\) denotes element-wise multiplication.

RMSNorm is used before attention and the feed-forward network,
followed by residual connections.

### 5.5 Feed-Forward Network

Each block contains a position-wise feed-forward network.

The initial implementation expands the hidden dimension by a factor
of four, applies GELU, and projects the result back to the model
dimension.

\[
d_{\text{ff}} = 4d_{\text{model}}
\]

The feed-forward network provides additional nonlinear
transformation capacity independently at each sequence position.

### 5.6 Residual Connections

Residual connections combine each sublayer's output with its input.

The block follows this general pattern:

\[
h' = h + \operatorname{Attention}(\operatorname{RMSNorm}(h))
\]

\[
h_{\text{next}}
=
h' +
\operatorname{FFN}(\operatorname{RMSNorm}(h'))
\]

Residual connections help preserve information across the network
and support gradient propagation during training.

### 5.7 Language Model Head

The final hidden states are normalized and projected into
vocabulary logits.

For each input position, the model produces a vector of length
`vocab_size`.

These logits represent the unnormalized scores used to calculate
the next-token probability distribution.

The initial design ties the language model output weights to the
input token embedding weights.

This means the input embedding and output projection share the
same parameter matrix.

Weight tying reduces the number of independent parameters.

## 6. Training

Implementation:

`src/repititar/train.py`

RPT-1 Remos is intended to learn from text using causal language
modeling.

The model is initialized with randomly generated parameters and
optimized using training data.

No pretrained model weights are required by the architecture.

### 6.1 Training Examples

A sequence of tokens is divided into input and target sequences.

For a token sequence:

```text
[A, B, C, D, E]
```

The training input and target are:

```text
Input:  [A, B, C, D]
Target: [B, C, D, E]
```

The model learns to predict the next token at every valid position.

### 6.2 Objective Function

The primary objective is next-token cross-entropy loss.

For a target sequence containing \(N\) valid prediction positions:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P(x_t \mid x_{<t})
\]

The model minimizes this loss by updating its trainable parameters.

### 6.3 Optimizer

The initial training design uses AdamW.

AdamW combines adaptive gradient updates with decoupled weight decay.

The learning rate, weight decay, batch size, training duration,
and other optimization parameters should be recorded for each
experiment.

Their values should be selected through controlled experiments
rather than assumed to be optimal.

### 6.4 Training Data

Training data must be provided explicitly.

Dataset provenance, licensing, preprocessing, deduplication,
and train-validation-test separation should be documented.

Evaluation data must remain separate from the training data used
to optimize model parameters.

Training results are only meaningful when the dataset and training
configuration are reproducible.

### 6.5 Checkpoints

Training checkpoints should preserve enough information to resume
training when supported by the training implementation.

An inference checkpoint should contain the model parameters and
any configuration information required to reconstruct the model.

Checkpoint formats should be documented, and untrusted checkpoint
files must not be loaded using unsafe deserialization settings.

A checkpoint is not evidence of successful training unless its
origin and training procedure can be established.

## 7. Text Generation

Implementation:

`src/repititar/generate.py`

Generation is autoregressive.

The model receives an initial sequence, predicts the next token,
appends the selected token, and repeats the process.

The generation loop is:

```text
Input Prompt
     |
     v
Model Forward Pass
     |
     v
Next-Token Logits
     |
     v
Temperature Scaling
     |
     v
Optional Top-k Filtering
     |
     v
Probability Distribution
     |
     v
Sample Next Token
     |
     v
Append Token to Sequence
     |
     +------ Repeat
```

### Temperature

Temperature controls the sharpness of the sampling distribution.

For logits \(z\) and positive temperature \(T\):

\[
P(x_i)
=
\operatorname{softmax}\left(\frac{z_i}{T}\right)
\]

Lower temperatures generally concentrate probability on more
likely tokens. Higher temperatures generally produce a flatter
distribution.

### Top-k Sampling

When top-k sampling is enabled, the candidate distribution is
restricted to the highest-scoring \(k\) tokens before sampling.

The selected value must be valid for the model vocabulary.

### Context Window

The model supports inputs up to `max_seq_len`.

When generation extends beyond this limit, the implementation
must use an appropriate context-window policy.

The initial approach retains the most recent tokens within the
maximum context length.

This approach does not implement a persistent key-value attention
cache.

### Generation Limitations

Generation quality depends on the model's training, data quality,
parameter count, and optimization.

A correctly implemented generation loop does not guarantee
coherent, factual, or useful text.

## 8. Evaluation

Implementation:

`src/repititar/evaluate.py`

Evaluation measures the model's next-token prediction performance
on held-out text.

### 8.1 Cross-Entropy Loss

Lower cross-entropy indicates that the model assigns greater
probability to the observed target tokens, averaged over the
evaluated prediction positions.

### 8.2 Perplexity

Perplexity is derived from mean cross-entropy loss:

\[
\operatorname{PPL} = e^{\mathcal{L}}
\]

Lower perplexity generally indicates better predictive performance
on the same evaluation dataset and under the same evaluation
methodology.

Perplexity comparisons require consistent tokenization, data
preprocessing, and evaluation procedures.

### 8.3 Bits per Byte

Because the tokenizer represents text as UTF-8 bytes, the
evaluation code can report cross-entropy in bits per byte:

\[
\operatorname{BPB}
=
\frac{\mathcal{L}}{\ln 2}
\]

This measures the average cross-entropy in bits for each evaluated
byte-level prediction.

### 8.4 Evaluation Integrity

Evaluation should use held-out data that was not used to optimize
the model.

Reports should record:

- Checkpoint identifier or path
- Model configuration
- Evaluation dataset
- Device
- Number of evaluated prediction positions
- Mean loss
- Perplexity
- Bits per byte
- Evaluation method

Metrics must be compared carefully when evaluation windowing,
context length, or dataset composition differs.

## 9. Software Structure

The principal project files are organized as follows:

```text
rpt-1-remos/
├── configs/
│   └── default.json
├── docs/
│   └── ARCHITECTURE.md
├── src/
│   └── repititar/
│       ├── __init__.py
│       ├── config.py
│       ├── tokenizer.py
│       ├── model.py
│       ├── train.py
│       ├── generate.py
│       ├── evaluate.py
│       └── api.py
├── tests/
│   ├── test_model.py
│   └── ...
├── .github/
│   └── workflows/
│       └── test.yml
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── NOTICE
└── README.md
```

This tree describes the intended project organization. Individual
files and optional modules may evolve during development.

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Model configuration and validation |
| `tokenizer.py` | UTF-8 byte encoding and decoding |
| `model.py` | Transformer layers, RoPE, forward pass, generation |
| `train.py` | Training loop, optimization, checkpoint creation |
| `generate.py` | Command-line text generation |
| `evaluate.py` | Held-out evaluation and metric reporting |
| `api.py` | Optional model-serving API |
| `tests/` | Automated correctness and regression tests |
| `configs/` | Reusable model configurations |

## 10. Testing and Continuous Integration

Automated tests are used to validate the model and prevent regressions.

Important test categories include:

- Model output dimensions
- Gradient propagation
- Weight tying
- Causal attention behavior
- RoPE mathematical properties
- Invalid input handling
- Empty-batch validation
- Generation argument validation
- Evaluation correctness
- Configuration validation
- Checkpoint compatibility

The GitHub Actions workflow should install the project and its
development dependencies, run Ruff, and execute the test suite.

A passing test suite demonstrates that the tested behavior satisfies
the implemented checks. It does not establish language-model
quality, factual accuracy, or successful large-scale training.

## 11. Hardware and Performance

The project is designed to support development on systems capable
of running PyTorch.

CPU execution is useful for unit tests, debugging, and small
experiments.

Compatible CUDA or Apple MPS devices may accelerate training and
inference when supported by the installed PyTorch version and
implementation.

Hardware availability constrains feasible model size, batch size,
sequence length, and training duration.

The default configuration is intended as an initial research
configuration. It should not be interpreted as a demonstrated
high-performance model.

Large-scale training requires appropriate compute resources,
sufficient training data, and systematic performance measurement.

## 12. Reproducibility and Research Standards

Experiments should record:

- Source revision
- Model configuration
- Dataset source and version
- Dataset preprocessing
- Random seeds
- Optimizer settings
- Training steps
- Training duration
- Hardware and software environment
- Validation and evaluation metrics
- Checkpoint provenance

Random seeds improve reproducibility but do not guarantee identical
results across different hardware, software versions, or
nondeterministic operations.

Published performance claims should be accompanied by reproducible
evaluation procedures and relevant experimental details.

## 13. Security and Reliability

The project should follow these practices:

- Validate user-provided input shapes and token IDs.
- Reject invalid model configurations.
- Avoid unsafe deserialization of untrusted checkpoints.
- Keep secrets out of source control.
- Validate API inputs and apply appropriate request limits.
- Avoid claiming that generated text is necessarily factual.
- Record the provenance of released weights and datasets.

If an inference API is exposed publicly, authentication, resource
limits, error handling, and deployment security must be considered.

## 14. Open Weights and Licensing

Repititar aims to make its model implementation and research
artifacts accessible.

The source-code license and the terms applicable to trained model
weights and training datasets must be clearly documented.

Open-source code does not automatically make every dataset or
trained checkpoint redistributable.

Before releasing model weights, document their provenance, the
training data used, the applicable license, and any known
limitations.

## 15. Development Roadmap

The following items are research directions, not claims that they
have already been implemented.

### Phase 1 — Correctness

- Validate the Transformer implementation.
- Verify RoPE mathematical properties.
- Strengthen automated tests.
- Validate configuration and checkpoint loading.
- Establish reproducible training and evaluation procedures.

### Phase 2 — Baseline Training

- Prepare a documented training dataset.
- Run small-scale training experiments.
- Track training and validation loss.
- Save and reload checkpoints.
- Establish a reproducible baseline.

### Phase 3 — Evaluation

- Compare checkpoints on held-out datasets.
- Track loss, perplexity, and bits per byte.
- Evaluate generation behavior.
- Document limitations and failure cases.

### Phase 4 — Optimization

- Profile training and inference performance.
- Evaluate mixed precision where supported.
- Investigate more efficient data loading.
- Measure the effects of architecture and hyperparameter changes.

### Phase 5 — Research Extensions

Potential areas for investigation include:

- Alternative positional encoding methods
- Different normalization strategies
- Learning-rate schedules
- Improved data mixtures
- Larger model configurations
- More efficient attention implementations
- Better evaluation and benchmarking procedures

Any extension should be supported by implementation tests and
measured experiments.

## 16. Design Principles

RPT-1 Remos follows these principles:

1. **Independent development:** Build and understand the model's
   implementation rather than depending on pretrained weights.
2. **Reproducibility:** Record the information needed to understand
   and repeat experiments.
3. **Correctness first:** Validate mathematical properties and
   software behavior before scaling training.
4. **Transparency:** Document actual capabilities and limitations.
5. **Maintainability:** Prefer clear modules and readable code.
6. **Measured progress:** Use experiments and evaluation to guide
   architectural decisions.
7. **Responsible release:** Document code, data, and checkpoint
   licensing and provenance.

---

**Repititar — RPT-1 Remos**

Independent model development. Open research. Measured progress.