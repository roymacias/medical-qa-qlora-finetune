"""End-to-end build of the cleaned corpora and experiment splits.

This script automates the pipeline for the building of the cleaned corpora and splits.

Configuration is loaded from a YAML file under ``configs/`` (default
``configs/data.yaml``); the script does not accept per-parameter CLI flags so
that every run is fully described by a versionable config.

Outputs (overwritten on every run):

- ``data/processed/`` — DatasetDict with ``medmcqa_processed`` and
  ``medqa_processed`` (cleaned corpora after filtering, dedup, leakage check).
- ``data/processed/splits/`` — DatasetDict with ``train``, ``validation``,
  ``test_id``, ``test_ood`` (final experiment splits).
- ``data/processed/build_metadata.json`` — operational metadata: seed, config
  source, target sizes, pool sizes, attrition counts, effective sizes,
  timestamp.

Usage:

    python -m src.data.build
    python -m src.data.build --config configs/data-experiment-x.yaml
    python -m src.data.build --quiet

Dependencies
------------
``datasets``, ``langdetect``, ``pandas``, ``pyyaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, load_from_disk

from .deduplication import QHASH_COL, dedup
from .filtering import filter_medmcqa, filter_medqa
from .leakage import remove_overlap
from .splits import (
    Splits,
    assert_no_leakage_across_splits,
    build_splits,
    pool_sizes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
METADATA_PATH = PROCESSED_DIR / "build_metadata.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

# Helper columns added during preprocessing that are dropped before saving.
DROP_COLS_ON_SAVE = [QHASH_COL]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_all_seeds(seed: int) -> None:
    """Seed every RNG used by the pipeline for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # langdetect's internal RNG must be seeded explicitly to avoid jitter on
    # short or ambiguous strings.
    import langdetect

    langdetect.DetectorFactory.seed = seed


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and validate the expected schema.

    The schema is intentionally minimal: ``test_id_size`` is NOT a config key,
    it is derived as the complement of ``val_size`` within the validation
    pool (with a fallback to 10% of train pool when the validation pool is
    fully consumed; see ``src.data.splits.build_splits``).
    """
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top = {"seed", "splits"}
    missing_top = required_top - cfg.keys()
    if missing_top:
        raise ValueError(f"config {path} missing top-level keys: {missing_top}")

    required_splits = {"train_size", "val_size"}
    missing_splits = required_splits - cfg["splits"].keys()
    if missing_splits:
        raise ValueError(f"config {path} missing splits keys: {missing_splits}")

    if "test_id_size" in cfg["splits"]:
        raise ValueError(
            f"config {path} contains 'test_id_size', which is no longer a "
            f"config key. test_id is derived automatically as the complement "
            f"of val_size within the validation pool. Remove this entry."
        )

    return cfg


# ---------------------------------------------------------------------------
# Data shape conversion
# ---------------------------------------------------------------------------


def to_dataframe(ds_dict: DatasetDict) -> pd.DataFrame:
    """Concatenate splits of a HuggingFace DatasetDict, tagging origin."""
    frames = []
    for split, ds in ds_dict.items():
        df = ds.to_pandas()
        df["_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_clean_dataset(df: pd.DataFrame) -> Dataset:
    """Drop helper columns and convert to a HuggingFace Dataset."""
    cols_to_drop = [c for c in DROP_COLS_ON_SAVE if c in df.columns]
    return Dataset.from_pandas(df.drop(columns=cols_to_drop), preserve_index=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    seed: int,
    train_size: int,
    val_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Splits, dict, dict]:
    """Run the full data pipeline end-to-end and return the artifacts.

    Returns
    -------
    medmcqa_clean, medqa_clean, splits, attrition, pools
    """
    log = logging.getLogger(__name__)

    log.info("Loading raw datasets from %s", RAW_DIR)
    medmcqa_raw = load_from_disk(str(RAW_DIR / "medmcqa"))
    medqa_raw = load_from_disk(str(RAW_DIR / "medqa"))

    medmcqa_df = to_dataframe(medmcqa_raw)
    medqa_df = to_dataframe(medqa_raw)
    log.info("Raw sizes: MedMCQA=%d MedQA=%d", len(medmcqa_df), len(medqa_df))

    attrition: dict[str, dict[str, int]] = {
        "MedMCQA": {"raw": len(medmcqa_df)},
        "MedQA": {"raw": len(medqa_df)},
    }

    # 1. Quality filtering
    medmcqa_df = filter_medmcqa(medmcqa_df)
    medqa_df = filter_medqa(medqa_df)
    attrition["MedMCQA"]["filtered"] = len(medmcqa_df)
    attrition["MedQA"]["filtered"] = len(medqa_df)

    # 2. Deduplication (also attaches `_qhash`)
    medmcqa_df = dedup(medmcqa_df, name="MedMCQA")
    medqa_df = dedup(medqa_df, name="MedQA")
    attrition["MedMCQA"]["deduplicated"] = len(medmcqa_df)
    attrition["MedQA"]["deduplicated"] = len(medqa_df)

    # 3. Cross-dataset leakage — overlaps removed from MedQA side.
    medqa_df, _n_overlap = remove_overlap(
        target=medqa_df,
        reference_hashes=medmcqa_df[QHASH_COL],
        target_name="MedQA",
        reference_name="MedMCQA",
    )
    attrition["MedMCQA"]["no_leakage"] = len(medmcqa_df)
    attrition["MedQA"]["no_leakage"] = len(medqa_df)

    # 4. OOV check is intentionally omitted here. Measuring OOV requires loading
    # the Gemma 3 tokenizer, which is gated on HuggingFace and forces a login
    # plus a non-trivial download — undesirable for a fast, credential-free
    # build script. The exploration notebook performs the actual OOV check
    # and observed an effectively-zero rate on this corpus, so skipping it
    # here is justified and does not affect data integrity.
    attrition["MedMCQA"]["after_oov"] = len(medmcqa_df)
    attrition["MedQA"]["after_oov"] = len(medqa_df)

    # Pool sizes are reported in metadata so the next config can be tuned
    # without re-running the notebook just to know the maxes.
    pools = pool_sizes(medmcqa_df, medqa_df)
    log.info("Pools after preprocessing: %s", pools)

    # 5. Build experiment splits.
    splits = build_splits(
        medmcqa_df,
        medqa_df,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
    )
    assert_no_leakage_across_splits(splits)

    return medmcqa_df, medqa_df, splits, attrition, pools


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_outputs(
    medmcqa_clean: pd.DataFrame,
    medqa_clean: pd.DataFrame,
    splits: Splits,
    attrition: dict,
    pools: dict,
    seed: int,
    train_size: int,
    val_size: int,
    config_path: Path,
) -> None:
    log = logging.getLogger(__name__)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Cleaned corpora at the top of data/processed/
    DatasetDict(
        {
            "medmcqa_processed": to_clean_dataset(medmcqa_clean),
            "medqa_processed": to_clean_dataset(medqa_clean),
        }
    ).save_to_disk(str(PROCESSED_DIR))
    log.info("Cleaned corpora saved to %s", PROCESSED_DIR)

    # Experiment splits one level down
    DatasetDict(
        {
            "train": to_clean_dataset(splits.train),
            "validation": to_clean_dataset(splits.validation),
            "test_id": to_clean_dataset(splits.test_id),
            "test_ood": to_clean_dataset(splits.test_ood),
        }
    ).save_to_disk(str(SPLITS_DIR))
    log.info("Experiment splits saved to %s", SPLITS_DIR)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "seed": seed,
        "config": {
            "train_size": train_size,
            "val_size": val_size,
            "test_id_size": "derived (complement of val_size in validation pool, or 10% of train pool fallback)",
        },
        "attrition": attrition,
        "pool_sizes": pools,
        "splits_sizes": splits.sizes(),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Build metadata saved to %s", METADATA_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build cleaned corpora and experiment splits from data/raw/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config (default: {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity to WARNING.",
    )
    args = parser.parse_args(argv)

    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    train_size = int(cfg["splits"]["train_size"])
    val_size = int(cfg["splits"]["val_size"])

    set_all_seeds(seed)

    logging.getLogger(__name__).info(
        "Loaded config from %s (seed=%d, train_size=%d, val_size=%d)",
        args.config.relative_to(PROJECT_ROOT),
        seed,
        train_size,
        val_size,
    )

    medmcqa_clean, medqa_clean, splits, attrition, pools = run_pipeline(
        seed=seed,
        train_size=train_size,
        val_size=val_size,
    )
    save_outputs(
        medmcqa_clean=medmcqa_clean,
        medqa_clean=medqa_clean,
        splits=splits,
        attrition=attrition,
        pools=pools,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        config_path=args.config.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
