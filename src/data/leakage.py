"""Cross-dataset leakage detection.

The reference (training) corpus is treated as authoritative; any overlap is
removed from the *target* (evaluation) corpus to preserve the integrity of the
test set.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from .deduplication import QHASH_COL


logger = logging.getLogger(__name__)


def remove_overlap(
    target: pd.DataFrame,
    reference_hashes: Iterable[str],
    target_name: str = "target",
    reference_name: str = "reference",
) -> tuple[pd.DataFrame, int]:
    """Drop rows from ``target`` whose ``_qhash`` appears in ``reference_hashes``.

    Both DataFrames must already carry the ``_qhash`` column produced by
    ``src.data.deduplication.dedup``.

    Returns the cleaned target DataFrame and the number of overlaps removed.
    """
    if QHASH_COL not in target.columns:
        raise ValueError(
            f"target DataFrame is missing the {QHASH_COL!r} column; run "
            f"src.data.deduplication.dedup first."
        )

    reference_set = set(reference_hashes)
    overlap_mask = target[QHASH_COL].isin(reference_set)
    n_overlap = int(overlap_mask.sum())

    logger.info(
        "leakage[%s vs %s]: overlap=%d",
        reference_name, target_name, n_overlap,
    )

    cleaned = target[~overlap_mask].reset_index(drop=True)
    return cleaned, n_overlap
