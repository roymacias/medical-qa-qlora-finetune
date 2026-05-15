"""Deduplication utilities.

Each unique question gets an MD5 hash over its normalized form (lowercased,
punctuation stripped, whitespace collapsed). Duplicates are dropped keeping the
first occurrence. The hash column ``_qhash`` is left attached to the returned
DataFrame because it is also reused by the cross-dataset leakage check
(``src.data.leakage``).
"""

from __future__ import annotations

import hashlib
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

QHASH_COL = "_qhash"


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = str(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def hash_question(text: str) -> str:
    """Return the MD5 hexdigest of the normalized text."""
    return hashlib.md5(normalize(text).encode("utf-8")).hexdigest()


def dedup(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Drop exact duplicates by MD5 hash of the normalized ``question`` field.

    Adds a ``_qhash`` column to the returned DataFrame. The first occurrence of
    each unique question is kept; subsequent duplicates are dropped.
    """
    df = df.copy()
    df[QHASH_COL] = df["question"].astype(str).map(hash_question)

    n_before = len(df)
    df = df.drop_duplicates(subset=QHASH_COL, keep="first").reset_index(drop=True)
    n_dups = n_before - len(df)

    logger.info(
        "dedup[%s]: %d -> %d (duplicates removed: %d)",
        name,
        n_before,
        len(df),
        n_dups,
    )
    return df
