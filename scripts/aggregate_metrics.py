"""Aggregate per-model ``metrics.json`` files into a single comparison JSON.

Reads ``artifacts/{model}/metrics.json`` for each model declared in the
``qualitative`` block of the evaluation config and writes a combined report at
``reports/evaluation comparison/comparison.json``.

Output structure
----------------
::

    {
      "models":        [<base>, <finetune>, ...],
      "order":         {"base": ..., "finetune": ..., ...},
      "metrics":       {<model>: <contents of metrics.json>, ...},
      "gap_recovered": {"test_id": <float|null>, "test_ood": <float|null>}
    }

``gap_recovered`` per split is computed as
``(accuracy_finetune - accuracy_base) / (accuracy_medgemma - accuracy_base)``
when the denominator is positive and the three accuracies are present;
``null`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
COMPARISON_DIR = PROJECT_ROOT / "reports" / "evaluation comparison"

log = logging.getLogger(__name__)


def _model_order(eval_cfg: dict) -> tuple[str | None, str | None, str | None]:
    q = eval_cfg.get("qualitative", {})
    return (q.get("base_model"), q.get("finetune_model"), q.get("medgemma_model"))


def _gap_recovered(
    base: float | None,
    finetune: float | None,
    medgemma: float | None,
) -> float | None:
    if None in (base, finetune, medgemma):
        return None
    denom = medgemma - base
    if denom <= 0:
        return None
    return (finetune - base) / denom


def aggregate(eval_cfg: dict, artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    base, finetune, medgemma = _model_order(eval_cfg)
    if not all((base, finetune, medgemma)):
        raise ValueError("eval config missing one of qualitative.model_name")
    models = [base, finetune, medgemma]

    per_model: dict[str, dict] = {}
    for name in models:
        path = artifacts_dir / name / "metrics.json"
        if not path.exists():
            log.warning("metrics.json missing for %s; skipping", name)
            continue
        with open(path, encoding="utf-8") as f:
            per_model[name] = json.load(f)

    gap_by_split: dict[str, float | None] = {}
    for split in ("test_id", "test_ood"):
        gap_by_split[split] = _gap_recovered(
            per_model.get(base, {}).get(split, {}).get("accuracy"),
            per_model.get(finetune, {}).get(split, {}).get("accuracy"),
            per_model.get(medgemma, {}).get(split, {}).get("accuracy"),
        )

    return {
        "models": models,
        "order": {"base": base, "finetune": finetune, "medgemma": medgemma},
        "metrics": per_model,
        "gap_recovered": gap_by_split,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-model metrics.json into a single comparison JSON.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG,
        help=f"Eval config YAML (default: {DEFAULT_EVAL_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.eval_config, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    agg = aggregate(eval_cfg)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPARISON_DIR / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
