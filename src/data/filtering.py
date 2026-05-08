"""Quality filters applied to MedMCQA and MedQA.

The two ``filter_*`` functions are pure transformations on a DataFrame and are
safe to call independently.
"""
from __future__ import annotations

import logging

import langdetect
import pandas as pd


logger = logging.getLogger(__name__)


def _detect_lang_safe(text: str) -> str:
    """Wrap ``langdetect.detect`` to never raise on degenerate inputs."""
    try:
        return langdetect.detect(text)
    except langdetect.LangDetectException:
        return "unknown"


def _word_count(text: str) -> int:
    return len(str(text).split())


# Public --------------------------------------------------------------------


def filter_medmcqa(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the quality filters to MedMCQA.

    Filters (in order):

    1. Required text fields are non-null and non-empty:
       ``question, opa, opb, opc, opd, exp, subject_name``.
    2. ``cop`` is in ``{0, 1, 2, 3}``.
    3. Question has at least 5 whitespace-separated tokens.
    4. Question is detected as English by ``langdetect``.

    Returns a new DataFrame with the index reset.
    """
    n_prev = len(df)
    logger.info("filter_medmcqa: initial=%d", n_prev)

    required_text = [
        "question", "opa", "opb", "opc", "opd", "exp", "subject_name",
    ]
    mask = df["cop"].notna()
    for col in required_text:
        mask &= df[col].notna() & (df[col].astype(str).str.strip().str.len() > 0)
    df = df[mask]
    logger.info(
        "filter_medmcqa: after non-empty fields=%d (dropped %d)",
        len(df), n_prev - len(df),
    )
    n_prev = len(df)

    df = df[df["cop"].astype(int).isin([0, 1, 2, 3])]
    logger.info(
        "filter_medmcqa: after cop in range=%d (dropped %d)",
        len(df), n_prev - len(df),
    )
    n_prev = len(df)

    df = df[df["question"].astype(str).map(_word_count) >= 5]
    logger.info(
        "filter_medmcqa: after question >= 5 toks=%d (dropped %d)",
        len(df), n_prev - len(df),
    )
    n_prev = len(df)

    logger.info(
        "filter_medmcqa: running language detection on %d examples "
        "(this can take several minutes)...",
        len(df),
    )
    df = df.copy()
    df["_lang"] = df["question"].astype(str).map(_detect_lang_safe)
    df = df[df["_lang"] == "en"].drop(columns=["_lang"])
    logger.info(
        "filter_medmcqa: after language == en=%d (dropped %d)",
        len(df), n_prev - len(df),
    )

    return df.reset_index(drop=True)


def filter_medqa(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to MedQA.

    The ``exp`` filter is intentionally NOT applied: MedQA has no explanations
    and is used only for evaluation, where the chain of reasoning is generated
    by the model in inference rather than consumed from the dataset.

    Filters (in order):

    1. ``question`` non-null/non-empty and ``options`` non-null.
    2. ``answer_idx`` in ``{A, B, C, D}``.
    3. Question has at least 5 whitespace-separated tokens.
    4. Question is detected as English.
    """
    n_prev = len(df)
    logger.info("filter_medqa: initial=%d", n_prev)

    mask = df["question"].notna() & (
        df["question"].astype(str).str.strip().str.len() > 0
    )
    mask &= df["options"].notna()
    df = df[mask]
    logger.info(
        "filter_medqa: after non-empty fields=%d (dropped %d)",
        len(df), n_prev - len(df),
    )
    n_prev = len(df)

    df = df[df["answer_idx"].astype(str).str.upper().isin(["A", "B", "C", "D"])]
    logger.info(
        "filter_medqa: after answer_idx in set=%d (dropped %d)",
        len(df), n_prev - len(df),
    )
    n_prev = len(df)

    df = df[df["question"].astype(str).map(_word_count) >= 5]
    logger.info(
        "filter_medqa: after question >= 5 toks=%d (dropped %d)",
        len(df), n_prev - len(df),
    )
    n_prev = len(df)

    logger.info(
        "filter_medmcqa: running language detection on %d examples "
        "(this can take several minutes)...",
        len(df),
    )
    df = df.copy()
    df["_lang"] = df["question"].astype(str).map(_detect_lang_safe)
    df = df[df["_lang"] == "en"].drop(columns=["_lang"])
    logger.info(
        "filter_medqa: after language == en=%d (dropped %d)",
        len(df), n_prev - len(df),
    )

    return df.reset_index(drop=True)
