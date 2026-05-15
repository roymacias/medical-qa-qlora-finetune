# ============================================================================
# Variables & Environment
# ============================================================================

# Force Python into UTF-8 mode to prevent Windows CP1252 decoding crashes in third-party libs
export PYTHONUTF8=1

# Use 'uv run' to execute within the managed virtual environment without activation
PYTHON := uv run python

# Default model name for evaluation and training
MODEL ?= gemma-3-4b-it-qlora-finetune

# Generic catch-all variable for extra CLI flags
ARGS ?=

# ============================================================================
# Flag Processing Logic
# ============================================================================

# --- Data Build Flags ---
DATA_FLAGS :=
ifdef CONFIG
	# Use abspath to prevent pathlib.relative_to() crashes in Python
	DATA_FLAGS += --config $(abspath $(CONFIG))
endif
ifeq ($(QUIET), 1)
	DATA_FLAGS += --quiet
endif

# --- Training Flags ---
TRAIN_FLAGS :=
ifdef CONFIG
	TRAIN_FLAGS += --config $(abspath $(CONFIG))
endif
ifdef RESUME
	TRAIN_FLAGS += --resume-from-checkpoint $(RESUME)
endif
ifdef MAX_STEPS
	TRAIN_FLAGS += --max-steps $(MAX_STEPS)
endif

# --- Evaluation Flags ---
EVAL_FLAGS :=
ifdef EVAL_CONFIG
	EVAL_FLAGS += --eval-config $(abspath $(EVAL_CONFIG))
endif
ifdef MODEL_CONFIG
	EVAL_FLAGS += --model-config $(abspath $(MODEL_CONFIG))
endif
ifdef SPLITS
	EVAL_FLAGS += --splits $(SPLITS)
endif
ifeq ($(FORCE), 1)
	EVAL_FLAGS += --force
endif

# ============================================================================
# Primary Targets
# ============================================================================

.PHONY: help setup lint clean data train eval

help:
	@echo "Medical QA Pipeline Management"
	@echo "------------------------------"
	@echo "setup   : Install dependencies and lock environment using uv."
	@echo "lint    : Format and check code quality (ruff)."
	@echo "clean   : Remove temporary build artifacts and caches."
	@echo ""
	@echo "Machine Learning Targets:"
	@echo "data    : Process raw datasets and create experiment splits."
	@echo "          Flags: [CONFIG=path/to/data.yaml] [QUIET=1]"
	@echo "train   : Execute the QLoRA fine-tuning pipeline."
	@echo "          Flags: [CONFIG=path/to/train.yaml] [RESUME=latest] [MAX_STEPS=5]"
	@echo "eval    : Run inference and compute metrics."
	@echo "          Flags: [MODEL=name] [EVAL_CONFIG=path] [MODEL_CONFIG=path] [SPLITS=\"s1 s2\"] [FORCE=1]"
	@echo ""
	@echo "Examples:"
	@echo "  make data CONFIG=configs/data.yaml"
	@echo "  make train MAX_STEPS=100"
	@echo "  make eval MODEL=medgemma-4b-it SPLITS=\"test_ood\" FORCE=1"

setup:
	uv sync --all-extras

lint:
	uv run ruff check src/ notebooks/ scripts/
	uv run ruff format src/ notebooks/ scripts/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/

# ----------------------------------------------------------------------------
# ML Pipelines
# ----------------------------------------------------------------------------

data:
	$(PYTHON) -m src.data.build $(DATA_FLAGS) $(ARGS)

train:
	$(PYTHON) -m src.pipelines.training_pipeline $(TRAIN_FLAGS) $(ARGS)

eval:
	# Evaluation pipeline requires --model.
	# It defaults to configs/models/{MODEL}.yaml unless MODEL_CONFIG is provided.
	$(PYTHON) -m src.pipelines.evaluation_pipeline --model $(MODEL) $(EVAL_FLAGS) $(ARGS)
