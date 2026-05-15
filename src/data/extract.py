"""Extract raw datasets from HuggingFace Hub and save to ``data/raw/``.

This script downloads the two datasets used in the project and persists them
locally so the rest of the pipeline (preprocessing notebook, training, eval)
can read from disk without hitting the network.

Datasets:

- **MedMCQA** (Pal et al., CHIL 2022) — ``train`` and ``validation`` splits.
  Used for fine-tuning, intermediate validation during training, and final
  in-distribution evaluation. The original ``test`` split is not retained
  because its labels are not publicly distributed.

- **MedQA-USMLE-4-options** (Jin et al., 2020) — ``test`` split only.
  Used exclusively for out-of-distribution evaluation.

Datasets are saved in HuggingFace Arrow format via ``DatasetDict.save_to_disk``,
so they can be reloaded with ``datasets.load_from_disk``.

The script is idempotent: if the target directory already exists, the download
is skipped unless ``--force`` is passed.

Usage
-----
From the repository root::

    python -m src.data.extract            # download if missing
    python -m src.data.extract --force    # re-download

Dependencies
------------
``datasets`` (HuggingFace) — install with ``pip install datasets``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset

# HuggingFace Hub identifiers. Adjust here if the canonical mirror moves.
MEDMCQA_HF_ID = "openlifescienceai/medmcqa"
MEDQA_HF_ID = "GBaker/MedQA-USMLE-4-options"

# Paths resolved relative to the project root (two levels up from this file:
# src/data/extract.py -> src/data -> src -> <project root>).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MEDMCQA_DIR = RAW_DIR / "medmcqa"
MEDQA_DIR = RAW_DIR / "medqa"

# Splits retained from each dataset. Anything else is dropped before saving.
MEDMCQA_KEEP_SPLITS: tuple[str, ...] = ("train", "validation")
MEDQA_KEEP_SPLITS: tuple[str, ...] = ("test",)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _download_and_save(
    hf_id: str,
    target_dir: Path,
    keep_splits: tuple[str, ...],
    force: bool,
) -> None:
    if target_dir.exists() and not force:
        logging.info(
            "%s already present at %s — skipping (use --force to re-download)",
            hf_id,
            target_dir,
        )
        return

    logging.info("Downloading %s from HuggingFace Hub", hf_id)
    full = load_dataset(hf_id)

    missing = [name for name in keep_splits if name not in full]
    if missing:
        raise RuntimeError(f"Dataset {hf_id!r} is missing expected splits: {missing}. Got splits: {list(full)}")

    subset = DatasetDict({name: full[name] for name in keep_splits})
    for name, split in subset.items():
        logging.info("  %s: %d examples", name, len(split))

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(target_dir))
    logging.info("Saved %s to %s", hf_id, target_dir)


def extract_medmcqa(force: bool = False) -> None:
    """Download MedMCQA train and validation splits."""
    _download_and_save(MEDMCQA_HF_ID, MEDMCQA_DIR, MEDMCQA_KEEP_SPLITS, force)


def extract_medqa(force: bool = False) -> None:
    """Download MedQA-USMLE-4-options test split."""
    _download_and_save(MEDQA_HF_ID, MEDQA_DIR, MEDQA_KEEP_SPLITS, force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MedMCQA and MedQA-USMLE-4-options to data/raw/.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if already present on disk.",
    )
    args = parser.parse_args()

    setup_logging()
    extract_medmcqa(force=args.force)
    extract_medqa(force=args.force)
    logging.info("Done.")


if __name__ == "__main__":
    main()
