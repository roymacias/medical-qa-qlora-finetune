"""End-to-end training pipeline for the QLoRA fine-tune.

Loads the 4-bit Gemma 3 base, attaches a LoRA adapter per ``configs/training.yaml``,
formats the train/eval splits as chat ``messages`` and runs SFT with TRL's
``SFTTrainer``. The completion-only loss masking is applied
by setting ``assistant_only_loss=True`` in the ``SFTConfig``: TRL uses the
tokenizer's chat template to identify the assistant turn and masks every
other token to ``-100`` at collation time.

Outputs
-------
Under ``artifacts/gemma-3-4b-it-qlora-finetune/`` (gitignored except for the
allowlisted ``adapter`` and ``metrics.json``):

- ``checkpoints/`` — intermediate checkpoints. ``save_total_limit=1`` keeps
  only the most recent for crash recovery; the best by ``eval_loss`` is
  preserved separately by the trainer when ``load_best_model_at_end=true``.
- ``adapter/`` — the final selected LoRA weights (``adapter_config.json`` +
  ``adapter_model.safetensors``).

Under ``reports/``:

- ``reports/training report/trainer_state.json`` — the full HF Trainer log
  history at end of run (every train log + every eval block).
- ``reports/figures/training/loss_curves.png`` — Built by
  ``src.models.qlora_finetune.curves`` from the trainer state above.

Usage
-----
::

    python -m src.pipelines.training_pipeline
    python -m src.pipelines.training_pipeline --config configs/training.yaml
    python -m src.pipelines.training_pipeline --resume-from-checkpoint latest
    python -m src.pipelines.training_pipeline --max-steps 5    # smoke test
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from src.models.qlora_finetune.curves import plot_loss_curves
from src.models.qlora_finetune.data import load_train_eval_datasets
from src.models.qlora_finetune.model import build_model_and_tokenizer


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

# Where the run-level training report lives.
TRAINING_REPORT_DIR = PROJECT_ROOT / "reports" / "training report"
LOSS_CURVES_PATH = PROJECT_ROOT / "reports" / "figures" / "training" / "loss_curves.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(p, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path from the config: absolute kept; relative joined to root."""
    path = Path(p)
    return path if path.is_absolute() else root / path


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Trainer / config wiring
# ---------------------------------------------------------------------------


def _build_sft_config(
    train_cfg: dict,
    output_root: Path,
    max_length: int,
    max_steps_override: int | None,
) -> SFTConfig:
    kwargs: dict = {
        "output_dir": str(output_root / "checkpoints"),

        # Duration / batch
        "num_train_epochs": int(train_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),

        # Optimizer / schedule
        "optim": train_cfg.get("optim", "paged_adamw_32bit"),
        "learning_rate": float(train_cfg["learning_rate"]),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),

        # Precision
        "bf16": bool(train_cfg.get("bf16", True)),

        # Memory
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},

        # Eval / logging cadence
        "eval_strategy": train_cfg.get("eval_strategy", "steps"),
        "eval_steps": int(train_cfg.get("eval_steps", 200)),
        "logging_strategy": train_cfg.get("logging_strategy", "steps"),
        "logging_steps": int(train_cfg.get("logging_steps", 25)),

        # Checkpointing
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "save_steps": int(train_cfg.get("save_steps", 200)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 1)),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(train_cfg.get("greater_is_better", False)),

        # External reporters (wandb/tensorboard) — opt-in via config.
        "report_to": train_cfg.get("report_to", "none"),

        # SFT-specific
        # The dataset has a `messages` column; SFTTrainer auto-detects it,
        # applies the chat template internally and (with the flag below)
        # masks everything outside the assistant turn to -100.
        "assistant_only_loss": True,
        "max_length": int(max_length),
        "packing": False,
        "remove_unused_columns": False,
    }
    if max_steps_override is not None:
        kwargs["max_steps"] = int(max_steps_override)
    return SFTConfig(**kwargs)


def _save_final_adapter(trainer: SFTTrainer, tokenizer, output_root: Path) -> None:
    """Persist the best adapter (loaded into memory by Trainer) to artifacts.

    Replaces any prior adapter at the destination so reruns produce a clean
    final artifact.
    """
    final_dir = output_root / "adapter"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Final adapter persisted to %s", final_dir)


def _save_training_report(trainer: SFTTrainer) -> None:
    """Persist the full trainer state to ``reports/training report/``.

    The trainer state contains the full ``log_history`` accumulated during the
    run (every train log + every eval block), which is what the curves module
    needs. Saving it under ``reports/`` makes the figure reproducible from a
    versioned source even after the intermediate checkpoints are pruned.
    """
    TRAINING_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = TRAINING_REPORT_DIR / "trainer_state.json"
    trainer.state.save_to_json(str(state_path))
    log.info("Trainer state saved to %s", state_path)

    plot_loss_curves(state_path=state_path, output_path=LOSS_CURVES_PATH)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(
    cfg: dict,
    resume_from_checkpoint: str | None = None,
    max_steps_override: int | None = None,
) -> None:
    set_seed(int(cfg.get("seed", 42)))

    # 1. Model + tokenizer (4-bit base + fresh LoRA adapter)
    model, tokenizer = build_model_and_tokenizer(cfg["base_model"], cfg["lora"])

    # 2. Datasets formatted as chat messages (no chat-template applied here)
    splits_dir = _resolve(cfg["data"]["splits_dir"])
    train_ds, eval_ds = load_train_eval_datasets(
        splits_dir=splits_dir,
        train_split=cfg["data"]["train_split"],
        eval_split=cfg["data"]["eval_split"],
    )

    # 3. Trainer config
    train_cfg = cfg["training"]
    output_root = _resolve(train_cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    sft_args = _build_sft_config(
        train_cfg=train_cfg,
        output_root=output_root,
        max_length=cfg["data"].get("max_length", 1024),
        max_steps_override=max_steps_override,
    )

    # 4. Trainer. With assistant_only_loss=True there is no need for a custom
    # collator; SFTTrainer applies the chat template and produces label masks
    # internally. The tokenizer is passed under both API names for
    # forward-compat across TRL versions that renamed `tokenizer` to
    # `processing_class`.
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # 5. Train. With load_best_model_at_end=True the trainer reloads the best
    # checkpoint into memory after the loop completes, so what is saved next
    # is already the selected adapter.
    log.info("Starting training")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 6. Persist the final adapter where eval pipeline expects it.
    _save_final_adapter(trainer, tokenizer, output_root)

    # 7. Persist the run-level training report.
    _save_training_report(trainer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA SFT training pipeline.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Training config YAML (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help="Checkpoint path or 'latest' to resume from a prior run.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override num_train_epochs by capping total optimizer steps. "
             "Useful for smoke-testing the pipeline end-to-end on a few steps.",
    )
    args = parser.parse_args(argv)

    setup_logging()
    cfg = _load_yaml(args.config)
    run(
        cfg=cfg,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_steps_override=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
