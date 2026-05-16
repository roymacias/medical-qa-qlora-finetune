<p align="center">
  <img src="docs/project_banner.png" alt="Medical QA QLoRA Fine-Tune" width="100%">
</p>

# Medical QA QLoRA Fine-Tune

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A study of parameter-efficient domain adaptation for medical multiple-choice question answering. The project applies QLoRA fine-tuning to `gemma-3-4b-it` on MedMCQA and benchmarks the resulting adapter against the unmodified base (floor) and `medgemma-4b-it` (industrial ceiling) under an identical evaluation protocol, on both an in-distribution test set (MedMCQA) and an out-of-distribution one (MedQA-USMLE).

The central question is how much of the gap between a general-purpose instruction-tuned LLM and a domain-adapted industrial sibling can be recovered with a single SFT pass on a consumer GPU.

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Quick Start](#quick-start)
3. [Repository Structure](#repository-structure)
4. [Data Pipeline](#data-pipeline)
5. [Model & Training](#model--training)
6. [Evaluation & Results](#evaluation--results)
7. [Discussion & Future Work](#discussion--future-work)
8. [Acknowledgements](#acknowledgements)
9. [License](#license)

## Setup & Installation

### Prerequisites

- Python 3.12+
- A CUDA-capable GPU with at least 24 GB VRAM for training and inference of the 4B model in 4-bit (the experiments were run on a single L4).
- A HuggingFace account with the license accepted for `gemma-3-4b-it` and `medgemma-4b-it`. Both models are gated.

### Local installation

The project uses `uv` for dependency and environment management. A frozen `uv.lock` pins the exact versions used to produce the environment.

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install dependencies (creates .venv automatically)
git clone https://github.com/roymacias/medical-qa-qlora-finetune.git
cd medical-qa-qlora-finetune
make setup
```

### Pre-commit code quality hooks

A multi-stage git hook configuration is enforced to guarantee compliance with the style rules declared in `.pre-commit-config.yaml` and `pyproject.toml`. Once installed, every commit is automatically checked and reformatted; non-conformant changes block the commit until they are addressed.

Install the active git hooks locally (one-off, after `make setup`):

```bash
uv run pre-commit install              # register the hooks in .git/hooks/
uv run pre-commit run --all-files      # optional: run the full hook suite on the current tree
```

### Docker

A two-stage `Dockerfile` is provided for reproducible runs on a CUDA host. It targets `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, installs Python 3.12 inside the image, and ships only `src/` and `configs/` into the runtime layer.

```bash
docker build -t medqa-qlora:$VERSION .
docker run --gpus all --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/reports:/app/reports \
  -e HF_TOKEN=$HF_TOKEN \
  medqa-qlora src.pipelines.training_pipeline
```

The container entrypoint is `python -m`, so any argument passed after the image name is interpreted as a Python module path. If no module is provided, the default command is `src.data.build`.

## Quick Start

The full experiment is driven by a small `Makefile`. Each target wraps a Python module under `src/pipelines/` or `src/data/` and forwards optional flags as environment-style variables.

```bash
make help                              # list all targets and flags

# 1. Download the raw datasets (MedMCQA + MedQA-USMLE) into data/raw/
uv run python -m src.data.extract      # one-off; cached locally afterwards

# 2. Build cleaned splits from the raw datasets
make data                              # uses configs/data.yaml

# 3. Train the QLoRA adapter on MedMCQA
make train                             # uses configs/training.yaml
make train MAX_STEPS=50                # short smoke run
make train RESUME=latest               # resume from the latest checkpoint

# 4. Run inference + metrics for each of the three compared models
make eval MODEL=gemma-3-4b-it
make eval MODEL=gemma-3-4b-it-qlora-finetune
make eval MODEL=medgemma-4b-it
make eval MODEL=medgemma-4b-it SPLITS="test_ood" FORCE=1   # only one split, force re-run
```

`src.data.extract` is the only pipeline step without a dedicated `make` target. It is designed to run only once, keeping `make data` fully offline and preventing repeated dataset downloads. If the raw datasets are not already present, the build fails fast and `src.data.extract` must be executed first.

Direct module invocations (equivalent and useful when calling from a notebook or another tool):

```bash
uv run python -m src.data.extract                    # download raw datasets
uv run python -m src.data.build                      # produce data/processed/splits/
uv run python -m src.pipelines.training_pipeline
uv run python -m src.pipelines.evaluation_pipeline --model gemma-3-4b-it
uv run python -m src.eval.qualitative                # cross-model qualitative markdown
uv run python scripts/aggregate_metrics.py           # cross-model comparison JSON
```

Linting and cleanup:

```bash
make lint                              # ruff check + ruff format on src/, notebooks/, scripts/
make clean                             # remove __pycache__, .ruff_cache
```

## Repository Structure

```
.
├── configs/                       # YAML configs driving every pipeline stage
│   ├── data.yaml                  # train/val/test target sizes and seeds
│   ├── training.yaml              # QLoRA + SFT hyperparameters
│   ├── evaluation.yaml            # generation kwargs, splits, qualitative samples
│   └── models/                    # per-model loading specs
├── src/
│   ├── data/                      # extract → filter → dedup → leakage → splits → build
│   ├── models/
│   │   ├── prompt.py              # shared chat-template rendering
│   │   └── qlora_finetune/        # model + data + curves modules for the finetune
│   ├── eval/                      # parsing, metrics, inference, qualitative
│   └── pipelines/
│       ├── training_pipeline.py   # end-to-end QLoRA SFT loop
│       └── evaluation_pipeline.py # end-to-end inference + metrics per model
├── scripts/
│   └── aggregate_metrics.py       # combines per-model metrics.json into one report
├── notebooks/
│   └── 01_data_exploration.ipynb  # initial EDA and figure generation
├── data/                          # data/raw, data/interim and data/processed
├── artifacts/                     # one folder per evaluated model
│   └── gemma-3-4b-it-qlora-finetune/
│       ├── adapter/               # final LoRA weights
│       ├── checkpoints/           # intermediate SFT checkpoints during training
│       ├── predictions/           # per-split prediction JSONL
│       └── metrics.json           # final accuracy / parse-rate report
├── reports/                       # figures and reports outputs
│   ├── eda report/                # metadata.json
│   ├── training report/           # full trainer_state.json
│   ├── evaluation comparison/     # cross-model comparison.json
│   ├── qualitative/               # per-(model, split) qualitative markdown
│   └── figures/                   # all PNG figures
├── Dockerfile
├── Makefile
├── pyproject.toml                 # project metadata + runtime dependencies
├── uv.lock                        # exact pinned versions used
```

## Data Pipeline

Two HuggingFace datasets:

| Dataset | Split used | Role |
|---|---|---|
| [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | `train` + `validation` | Training and in-distribution evaluation (`test_id`) |
| [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | `test` only | Out-of-distribution evaluation (`test_ood`) |

`make data` runs the deterministic preprocessing pipeline (`src/data/build.py`): quality filtering, MD5-based deduplication, cross-dataset leakage check, and stratified sampling by `subject_name`. Effective experiment splits (after all filters) can be found in `configs/data.yaml`.

## Model & Training

The fine-tune sits on top of `google/gemma-3-4b-it` with QLoRA — 4-bit NF4-quantized base, low-rank adapters in bf16 (`configs/training.yaml`).

| Hyperparameter | Value |
|---|---|
| Base model | `google/gemma-3-4b-it` (frozen, NF4 + double quant) |
| LoRA rank `r` / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | `paged_adamw_32bit` |
| Learning rate | `2e-4` with cosine schedule, 3 % warmup |
| Effective batch size | 16 (`per_device=2` × `grad_accum=8`) |
| Epochs / max length | 1 / 1024 tokens |
| Loss | Assistant-only cross-entropy (`assistant_only_loss=True` in TRL) |

`make train` produces:

- `artifacts/gemma-3-4b-it-qlora-finetune/adapter/` — final LoRA weights
- `reports/training report/trainer_state.json` — full HF Trainer log history
- `reports/figures/training/loss_curves.png` — training and validation loss

## Evaluation & Results

All three models are evaluated under an identical protocol (`configs/evaluation.yaml` and `configs/models/<model>.yaml`). Answers are extracted from the generated text via a four-pattern regex cascade (`src/eval/parsing.py`) and parse failures count as wrong.

| Model | `test_id` accuracy | `test_id` parse rate | `test_ood` accuracy | `test_ood` parse rate |
|---|---|---|---|---|
| `gemma-3-4b-it` (base) | 46.15 % | 99.75 % | 53.34 % | 99.92 % |
| `gemma-3-4b-it-qlora-finetune` (this work) | 44.95 % | 96.95 % | 49.80 % | 99.69 % |
| `medgemma-4b-it` (industrial ceiling) | 56.64 % | 99.81 % | 66.22 % | 99.92 % |

The fraction of the base → MedGemma gap recovered by the fine-tune is −11.51 % on `test_id` and −27.44 % on `test_ood`: the adapter does not close the gap; it falls below the base by a small margin in-distribution and by a larger one out-of-distribution. The negative result is real and informative, see discussion in next section.

`make eval MODEL=<name>` runs inference + metrics for one model and writes:

- `artifacts/<name>/predictions/{test_id,test_ood}.jsonl` — per-question outputs
- `artifacts/<name>/metrics.json` — versioned summary

Cross-model and qualitative outputs:

- `reports/evaluation comparison/comparison.json` — produced by `scripts/aggregate_metrics.py`
- `reports/qualitative/<model>/<split>.md` — three side-by-side cases per the samples selection, produced by `src/eval/qualitative.py`

## Discussion

## Discussion & Future Work

The qualitative analysis (`reports/qualitative/gemma-3-4b-it-qlora-finetune/`) shows the dominant failure mode of the fine-tune: it adopts the surface style of MedMCQA's explanation field and breaks the answer-format discipline the base model inherited from its own instruction tuning. Three factors are consistent with the evidence:

- **Data scale insufficient**. For a stable adaptation at this regime, training subset is generous for an academic project but small compared to industrial fine-tuning corpora, and a single epoch leaves the gradient dominated by the surface patterns of the corpus rather than by clinical reasoning.

- **4-bit quantization precision cost**. Keeping the base model in NF4 with double quantization is what makes the run feasible on consumer hardware. However, the additional quantization noise can degrade adaptation quality compared to applying LoRA over a bf16 base model with sufficient VRAM.

- **Missing domain pretraining + post-SFT alignment**. Industrial pipelines like MedGemma's include a continued-pretraining phase on medical corpora followed by a preference-alignment stage to preserve instruction-following behavior, coherence, and formatting quality. The pipeline presented here includes neither stage and performs only the intermediate supervised fine-tuning step.

Concrete directions for follow-up work (no order):

- **Continued pretraining on medical sources before SFT.** A lightweight continued-pretraining stage on sources such as PubMed abstracts, clinical guidelines, or open medical references would better expose the model to the terminology and structure of the medical domain before SFT. This would also help separate how much improvement comes from domain adaptation itself versus the SFT stage alone.
- **Post-SFT alignment (DPO / RLHF).** Applying a small DPO or RLHF step using the qualitative failures identified during evaluation could help recover instruction-following behavior, formatting consistency, and answer discipline without repeating the entire fine-tuning process. More broadly, this would test whether effective domain adaptation requires not just SFT alone, but SFT followed by a lightweight alignment stage.
- **Wider hyperparameter sweep.** Higher LoRA rank, alternative target-module compositions, multiple epochs, larger training subset, learning-configuration changes, among other factors that may influence adaptation quality.
- **Plain LoRA over a bf16 base**. On hardware that allows it, to quantify the precision cost of QLoRA in this specific setting.

It is also noteworthy that even MedGemma did not achieve a 100% parse rate. Manual inspection showed that the reference model itself occasionally deviated from the requested answer format, suggesting that strict format adherence remains a non-trivial behavior even for industrial domain-adapted models.

## Acknowledgements

Special thanks to everyone who keeps trying, learning, building, and making this world a more valuable, beautiful, and enjoyable place to live in.

Additional thanks to the maintainers and organizations behind the open models, datasets, research, and software libraries that made this project possible.

## License

Released under the [MIT License](LICENSE). Base models, reference models, and external datasets remain subject to their respective owners' license terms.
