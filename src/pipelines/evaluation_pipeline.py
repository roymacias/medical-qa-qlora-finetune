"""End-to-end evaluation pipeline for a single model.

Runs inference on the requested splits, persists predictions JSONL per
(model, split), and writes one ``metrics.json`` summarizing accuracy and
parse-success across splits and (for MedMCQA-derived splits) per subject.

Qualitative analysis is a SEPARATE step (``python -m src.eval.qualitative``)
because the cross-model 5-row pattern requires predictions from all 3 models;
running the qualitative step early would just fail.

Outputs
-------
- ``artifacts/{model}/predictions/{split}.jsonl`` — verbose per-question, NOT
  versioned. Streamed line-by-line so a crash mid-run leaves a valid partial.
- ``artifacts/{model}/metrics.json`` — versioned.

Usage examples
-----
::

    python -m src.pipelines.evaluation_pipeline --model gemma-3-4b-it
    python -m src.pipelines.evaluation_pipeline --model gemma-3-4b-it-qlora-finetune
    python -m src.pipelines.evaluation_pipeline --model medgemma-4b-it
    python -m src.pipelines.evaluation_pipeline --model gemma-3-4b-it --splits test_ood
    python -m src.pipelines.evaluation_pipeline --model gemma-3-4b-it --force

CLI flags
---------
``--model``         Model name. Resolved to ``configs/models/{name}.yaml`` unless
                    ``--model-config`` is given.
``--eval-config``   Path to the horizontal eval config (default
                    ``configs/evaluation.yaml``).
``--model-config``  Override path to the per-model YAML.
``--splits``        Subset of splits to run (default: as declared in eval config).
``--force``         Re-run inference even if predictions JSONL already exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from datasets import DatasetDict

from src.eval.inference import (
    GenerationConfig,
    load_model_and_tokenizer,
    run_inference,
)
from src.eval.metrics import compute_metrics, load_predictions, save_metrics

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"
MODELS_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(p: str | Path) -> Path:
    """Resolve a path from config: absolute kept as-is, relative joined to root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(
    model_name: str,
    eval_cfg: dict,
    model_cfg: dict,
    splits_override: list[str] | None = None,
    force: bool = False,
) -> None:
    splits_dir = _resolve(eval_cfg.get("data_dir", "data/processed/splits"))
    output_root = _resolve(eval_cfg.get("output_root", "artifacts"))
    splits = splits_override or eval_cfg.get("splits", ["test_id", "test_ood"])
    gen_cfg = GenerationConfig(**eval_cfg.get("generation", {}))
    log.info("Generation config: %s", gen_cfg)

    log.info("Loading splits from %s", splits_dir)
    dd = DatasetDict.load_from_disk(str(splits_dir))
    missing = [s for s in splits if s not in dd]
    if missing:
        raise KeyError(f"requested splits not in DatasetDict: {missing} (have {sorted(dd)})")

    out_dir = output_root / model_name
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Pick which splits actually need a fresh inference pass.
    pending: list[str] = []
    for split in splits:
        path = pred_dir / f"{split}.jsonl"
        if path.exists() and not force:
            log.info("Skipping inference (predictions exist): %s", path)
        else:
            pending.append(split)

    # Only load the model if at least one split needs to be regenerated.
    if pending:
        log.info("Loading model %s for splits %s", model_name, pending)
        model, tokenizer = load_model_and_tokenizer(model_cfg)
        try:
            for split in pending:
                output_path = pred_dir / f"{split}.jsonl"
                run_inference(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dd[split],
                    split=split,
                    gen_cfg=gen_cfg,
                    output_path=output_path,
                )
        finally:
            # Free GPU memory before downstream steps.
            del model
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - cuda may be absent
                pass

    # Compute metrics across ALL requested splits (re-reads predictions JSONL).
    records_by_split = {split: load_predictions(pred_dir / f"{split}.jsonl") for split in splits}
    metrics = compute_metrics(records_by_split)
    metrics_path = out_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    log.info("Metrics saved to %s", metrics_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluation (inference + metrics) for a single model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name; resolves to configs/models/{name}.yaml unless --model-config is given.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG,
        help=f"Horizontal eval config (default: {DEFAULT_EVAL_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Override path to the per-model YAML.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        help="Override splits from eval config (e.g. test_id test_ood).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if predictions JSONL exists.",
    )
    args = parser.parse_args(argv)

    setup_logging()

    eval_cfg = _load_yaml(args.eval_config)
    model_cfg_path = args.model_config or (MODELS_CONFIG_DIR / f"{args.model}.yaml")
    model_cfg = _load_yaml(model_cfg_path)
    if model_cfg.get("name") not in (None, args.model):
        log.warning(
            "model name in config (%r) differs from --model (%r); using --model",
            model_cfg.get("name"),
            args.model,
        )

    run(
        model_name=args.model,
        eval_cfg=eval_cfg,
        model_cfg=model_cfg,
        splits_override=args.splits,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
