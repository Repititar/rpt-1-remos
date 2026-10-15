<div align="center">
  <img src="./rpt.png" alt="Repititar logo" width="160">

  # Repititar

  ### RPT-1 Remos

  An independently developed, open-weight language model research project.

  [Architecture](./docs/ARCHITECTURE.md) · [Training](./src/repititar/train.py) · [Configuration](./configs/default.json)
</div>

---

## About

Repititar is an independent language model research project focused on developing, training, evaluating, and releasing language models with their own learned weights.

**RPT-1 Remos** is the first model in the Repititar series.

The project begins with a decoder-only Transformer language model trained from randomly initialized weights. Its initial goal is to establish a reproducible training pipeline, validate the architecture, and build the infrastructure needed for systematic language model research.

Repititar aims to make its source code, configurations, training procedures, and evaluation results accessible to the open-source community.

> Model quality must be demonstrated through reproducible experiments and evaluation, not architectural descriptions alone.

## Goals

- Develop and document the Repititar architecture.
- Train model weights from random initialization.
- Build a reproducible training and evaluation pipeline.
- Support checkpoint saving and training resumption.
- Provide an autoregressive text-generation interface.
- Automate testing with GitHub Actions.
- Publish model weights and documentation when available.
- Investigate architectural improvements through controlled experiments.

## Model Information

| Property | Details |
| --- | --- |
| Project | Repititar |
| Model | RPT-1 Remos |
| Repository | `Repititar/rpt-1-remos` |
| Architecture | Decoder-only Transformer |
| Framework | PyTorch |
| Language | Python |
| Tokenizer | UTF-8 byte-level tokenizer |
| Attention | Causal self-attention |
| Positional encoding | Rotary positional embeddings (RoPE) |
| Normalization | RMSNorm |
| Activation | GELU |
| Optimizer | AdamW |
| Weight initialization | Random initialization |
| Pretrained weights | None used by the initial training pipeline |
| Vocabulary size | 256 byte values |
| Hidden dimension | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Maximum context length | 512 tokens |

These values describe the initial configuration, not a claim of competitive performance against existing language models.

## Architecture

The initial RPT-1 Remos implementation consists of the following components:

- **Byte-level tokenizer:** Converts UTF-8 text into byte IDs without requiring a pretrained tokenizer.
- **Token embeddings:** Maps byte IDs to hidden representations.
- **Causal self-attention:** Allows each position to attend only to preceding positions and itself.
- **Rotary positional embeddings:** Encodes positional information in attention queries and keys.
- **RMSNorm:** Normalizes hidden representations.
- **Feed-forward networks:** Transform representations inside each Transformer block.
- **Language modeling head:** Predicts the next byte in a sequence.
- **Autoregressive generation:** Generates text one token at a time.

The initial architecture serves as a baseline for experimentation. Proposed improvements must be evaluated against measurable baselines before being considered successful.

See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for the architecture specification.

## Repository Structure

```text
rpt-1-remos/
├── .github/
│   └── workflows/
├── configs/
│   └── default.json
├── data/
│   └── README.md
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
│   ├── test_tokenizer.py
│   └── test_model.py
├── scripts/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── rpt.png
```

This structure describes the intended project layout. Files that have not yet been created or implemented should be treated as planned components.

## Requirements

- Python 3.11 or later
- PyTorch 2.4 or later
- Git
- A compatible CPU or CUDA-enabled GPU

A GPU is recommended for faster training. Small experiments can run on a CPU, but training may be significantly slower.

## Installation

Clone the repository:

```bash
git clone https://github.com/Repititar/rpt-1-remos.git
cd rpt-1-remos
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the project:

```bash
python -m pip install --upgrade pip
pip install -e .
pip install pytest
```

Alternatively, install the listed dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The default configuration is stored in `configs/default.json`.

```json
{
  "vocab_size": 256,
  "d_model": 256,
  "n_layers": 6,
  "n_heads": 8,
  "max_seq_len": 512,
  "dropout": 0.1,
  "rope_theta": 10000.0,
  "norm_eps": 1e-5
}
```

| Parameter | Description |
| --- | --- |
| `vocab_size` | Number of supported byte IDs |
| `d_model` | Hidden representation dimension |
| `n_layers` | Number of Transformer blocks |
| `n_heads` | Number of attention heads |
| `max_seq_len` | Maximum context length |
| `dropout` | Dropout probability during training |
| `rope_theta` | Rotary positional embedding base |
| `norm_eps` | Numerical stability constant for RMSNorm |

The hidden dimension must be divisible by the number of attention heads. The attention head dimension must also be even for the current RoPE implementation.

Increasing the model dimensions or context length increases computational and memory requirements.

## Dataset Preparation

The initial training pipeline uses a byte-level tokenizer and does not require a pretrained tokenizer vocabulary.

Prepare separate training and validation text files:

```text
data/
├── README.md
├── train.txt
└── validation.txt
```

These files must be created locally unless they are already present in the repository.

Before training:

- Verify that the dataset can legally be used for model training.
- Document data sources and licenses.
- Keep validation data separate from training data.
- Avoid training-validation contamination.
- Document preprocessing steps.
- Exclude sensitive personal information where appropriate.
- Avoid committing large or restricted datasets to Git.

See `data/README.md` for dataset preparation instructions.

## Training

RPT-1 Remos is trained using autoregressive next-token prediction.

The initial training pipeline includes:

- Random weight initialization
- Cross-entropy loss
- AdamW optimization
- Learning-rate warmup and cosine decay
- Gradient clipping
- Periodic validation
- Checkpoint saving
- Training resumption

Inspect the available training arguments:

```bash
python -m repititar.train --help
```

Training requires a prepared dataset and sufficient computational resources. Check the script's supported arguments before starting a training run.

A successful training run does not automatically mean the model generates useful or coherent text. Training loss, validation loss, and other evaluation results must be examined.

## Text Generation

After training, inspect the generation options:

```bash
python -m repititar.generate --help
```

Example:

```bash
python -m repititar.generate \
  --checkpoint checkpoints/best.pt \
  --config configs/default.json \
  --prompt "Repititar is"
```

The checkpoint must exist and contain weights compatible with the selected configuration.

The initial tokenizer operates on UTF-8 bytes rather than words or subword tokens. Consequently, its token counts are not directly comparable to those of models using conventional subword tokenizers.

## Testing

Run the test suite:

```bash
pytest
```

The initial tests cover:

- Tokenizer round trips
- Unicode text handling
- Configuration validation
- Model output shapes
- Forward and backward passes
- Weight tying
- Basic generation behavior
- Context-length validation

Passing these tests verifies selected implementation behaviors. It does not establish language quality, reasoning capability, safety, or generalization.

## Evaluation

Model quality must be assessed through reproducible experiments.

Relevant measurements include:

- Training loss
- Validation loss
- Perplexity
- Training throughput
- Peak memory usage
- Checkpoint size
- Generation latency
- Performance on held-out tasks

Comparisons should document the datasets, model configurations, evaluation procedures, and computational resources used.

Reasoning ability must not be claimed solely because the model uses a Transformer architecture or generates text that resembles a chain of thought.

## GitHub Actions

GitHub Actions can automate:

- Tests on pull requests
- Package installation checks
- Configuration validation
- Code-quality checks
- Small CPU-based smoke tests

Standard GitHub-hosted runners are not a replacement for dedicated large-scale GPU training infrastructure. Larger training runs require appropriate compute, storage, time, and a documented training plan.

Continuous integration tests and actual model training should be treated as separate workflows.

## Checkpoints and Model Weights

Training checkpoints may contain model weights, optimizer state, random-number-generator state, and training progress.

Checkpoints support training resumption and experiment reproducibility.

A model release should include:

- Model configuration
- Trained weights
- Dataset documentation
- Training procedure
- Evaluation results
- Hardware and software details
- Known limitations
- Applicable license information

Only publish weights that have actually been produced and tested. Randomly initialized weights must not be presented as a trained model release.

Load checkpoint files only from trusted sources.

## Project Status

RPT-1 Remos is an evolving research and development project.

The initial implementation establishes the model and training infrastructure. Further work is required to validate training stability, evaluate language modeling performance, determine appropriate datasets and compute requirements, and establish reproducible release procedures.

Progress should be measured through working code, verified training runs, documented evaluation results, and reproducible experiments.

No benchmark ranking, reasoning performance, or superiority over existing models is claimed without supporting evidence.

## Roadmap

- [x] Establish the initial repository structure
- [x] Implement model configuration
- [x] Implement byte-level tokenization
- [x] Implement the initial Transformer model
- [x] Add core model and tokenizer tests
- [ ] Finalize the architecture specification
- [ ] Validate the complete training pipeline
- [ ] Implement deterministic evaluation and reporting
- [ ] Configure continuous integration
- [ ] Run documented training experiments
- [ ] Publish reproducible evaluation results
- [ ] Prepare the first verified model-weight release
- [ ] Investigate architectural improvements through controlled experiments

Update these checkboxes as work is completed and verified.

## Contributing

Contributions, issue reports, architecture discussions, and reproducibility improvements are welcome.

When contributing:

1. Explain the problem or proposed improvement.
2. Keep changes focused and maintainable.
3. Add tests for new behavior where appropriate.
4. Document changes to the architecture or training procedure.
5. Include experimental evidence for performance claims.

Experimental ideas should be compared against a clear baseline rather than presented as improvements without measurements.

## Licensing

Source code, trained weights, and datasets may have different licensing requirements.

Each component must have its own clearly documented license or usage terms where necessary. The license for one component does not automatically apply to the others.

Until the relevant license files and release documentation are published, do not assume that all project components are available under the same terms.

## Acknowledgements

Repititar builds on established machine learning research and software, including PyTorch and the Transformer architecture.

The initial implementation uses established techniques such as causal self-attention, RMSNorm, rotary positional embeddings, and AdamW. Their inclusion does not imply that these techniques originated with Repititar.

Relevant research references and implementation details should be documented in the architecture specification.

---

<div align="center">

**Repititar — RPT-1 Remos**

Research. Build. Train. Evaluate.

</div>