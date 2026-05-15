"""Stratified subsampling to generate experiment splits.

Builds the experiment splits (``train``, ``validation``, ``test_id``,
``test_ood``) from the cleaned MedMCQA and MedQA corpora. Stratification on
MedMCQA-derived splits is performed by ``subject_name`` to preserve the
natural specialty distribution of the corpus.

The size of ``test_id`` is **derived**, not configured: it is the complement
of ``val_size`` within the validation pool. This avoids inconsistent configs
where the user requests ``val_size + test_id_size > pool_val_size``. When
``val_size`` consumes the entire validation pool, the build falls back to
sourcing ``test_id`` from the train pool (10%, stratified, excluded from the
train sample) and logs a WARNING.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


# Defaults used when callers (CLI, tests) don't pass explicit values.
DEFAULT_TRAIN_SIZE = 50_000
DEFAULT_VAL_SIZE = 500

# Fraction of the train pool used as test_id when the validation pool is
# fully consumed by val_size (edge-case fallback).
TEST_ID_FALLBACK_FRACTION = 0.10
# Lower bound to keep the fallback test_id statistically meaningful even on
# unusually small train pools.
TEST_ID_FALLBACK_MIN = 100

STRATIFY_COL = "subject_name"


@dataclass(frozen=True)
class Splits:
    """Container for the four experiment splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test_id: pd.DataFrame
    test_ood: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test_id": len(self.test_id),
            "test_ood": len(self.test_ood),
        }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    group_col: str,
    seed: int,
) -> pd.DataFrame:
    """Return up to ``n`` rows, stratified by ``group_col``.

    If ``len(df) <= n``, returns the whole DataFrame shuffled. Otherwise
    samples proportionally per group, then trims to exactly ``n`` rows with a
    final random sample to absorb rounding.
    """
    if len(df) <= n:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    weights = df[group_col].value_counts(normalize=True)
    chunks = []
    for group, w in weights.items():
        gdf = df[df[group_col] == group]
        size = max(1, int(round(n * w)))
        chunks.append(gdf.sample(n=min(size, len(gdf)), random_state=seed))
    out = pd.concat(chunks, ignore_index=True)

    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out.reset_index(drop=True)


def _warn_if_target_exceeds_pool(target: int, pool_size: int, name: str) -> None:
    """Emit a WARNING (not error) when a config target cannot be fully honored."""
    if target > pool_size:
        logger.warning(
            "%s=%d exceeds available pool of %d after preprocessing; falling back to all %d available examples.",
            name,
            target,
            pool_size,
            pool_size,
        )


def pool_sizes(
    medmcqa_clean: pd.DataFrame,
    medqa_clean: pd.DataFrame,
) -> dict[str, int]:
    """Return the per-source pool sizes after preprocessing."""
    return {
        "medmcqa_train": int((medmcqa_clean["_split"] == "train").sum()),
        "medmcqa_validation": int((medmcqa_clean["_split"] == "validation").sum()),
        "medqa_test": int(len(medqa_clean)),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def build_splits(
    medmcqa_clean: pd.DataFrame,
    medqa_clean: pd.DataFrame,
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 42,
) -> Splits:
    """Build the four experiment splits from cleaned corpora.

    Parameters
    ----------
    medmcqa_clean
        MedMCQA after filtering, deduplication and leakage removal. Must
        carry the ``_split`` column tagging the original partition (train,
        validation) and the ``_qhash`` column from deduplication.
    medqa_clean
        MedQA after preprocessing. Becomes ``test_ood`` in full.
    train_size
        Target size for the training subset, stratified by specialty from the
        MedMCQA train pool.
    val_size
        Target size for the intermediate validation subset, drawn stratified
        from the MedMCQA validation pool. ``test_id`` is derived as the
        complement (or as a 10% slice of the train pool if val_size consumes
        the entire validation pool — see below).
    seed
        Random seed propagated to all sampling operations.

    Returns
    -------
    Splits with DataFrames: ``train``, ``validation``, ``test_id``,
    ``test_ood``. All carry ``_qhash`` for downstream leakage verification.
    """
    from .deduplication import QHASH_COL

    pool_train = medmcqa_clean[medmcqa_clean["_split"] == "train"].reset_index(drop=True)
    pool_val = medmcqa_clean[medmcqa_clean["_split"] == "validation"].reset_index(drop=True)
    pool_test_ood = medqa_clean.reset_index(drop=True)

    logger.info(
        "build_splits: pools train=%d val=%d test_ood=%d",
        len(pool_train),
        len(pool_val),
        len(pool_test_ood),
    )

    # Per-pool overflow warnings.
    _warn_if_target_exceeds_pool(train_size, len(pool_train), "train_size")
    _warn_if_target_exceeds_pool(val_size, len(pool_val), "val_size")

    # ---------------- validation + test_id derivation ---------------------
    if val_size >= len(pool_val):
        # Edge case: the validation pool is fully consumed by val_size, leaving
        # zero examples for test_id from the validation side. Fall back to
        # taking ~10% of the train pool as test_id (stratified by specialty)
        # and exclude those examples from the train sample to preserve
        # no-leakage between train and test_id.
        test_id_target = max(
            TEST_ID_FALLBACK_MIN,
            int(round(len(pool_train) * TEST_ID_FALLBACK_FRACTION)),
        )
        logger.warning(
            "val_size=%d consumes the entire validation pool (%d) — no "
            "examples remain for test_id from the validation side. Falling "
            "back to taking %d examples (~%.0f%% of train pool) as test_id, "
            "stratified by specialty and excluded from the train sample. "
            "Reduce val_size in the config to keep test_id sourced from the "
            "validation pool.",
            val_size,
            len(pool_val),
            test_id_target,
            TEST_ID_FALLBACK_FRACTION * 100,
        )
        validation = pool_val.sample(frac=1, random_state=seed).reset_index(drop=True)
        test_id = stratified_sample(
            pool_train,
            test_id_target,
            STRATIFY_COL,
            seed=seed,
        )
        train_pool_excl = pool_train[~pool_train[QHASH_COL].isin(set(test_id[QHASH_COL]))].reset_index(drop=True)
        train = stratified_sample(
            train_pool_excl,
            train_size,
            STRATIFY_COL,
            seed=seed,
        )
    else:
        # Standard path: stratified val from val pool; test_id is the
        # complement (also approximately stratified).
        validation = stratified_sample(
            pool_val,
            val_size,
            STRATIFY_COL,
            seed=seed,
        )
        val_hashes = set(validation[QHASH_COL])
        test_id = pool_val[~pool_val[QHASH_COL].isin(val_hashes)].reset_index(drop=True)
        train = stratified_sample(
            pool_train,
            train_size,
            STRATIFY_COL,
            seed=seed,
        )

    test_ood = pool_test_ood.copy()

    splits = Splits(
        train=train,
        validation=validation,
        test_id=test_id,
        test_ood=test_ood,
    )

    logger.info(
        "build_splits: produced train=%d validation=%d test_id=%d test_ood=%d",
        len(splits.train),
        len(splits.validation),
        len(splits.test_id),
        len(splits.test_ood),
    )
    return splits


def assert_no_leakage_across_splits(splits: Splits) -> None:
    """Raise if any pair of splits shares a question hash."""
    from .deduplication import QHASH_COL

    if any(
        QHASH_COL not in df.columns
        for df in (
            splits.train,
            splits.validation,
            splits.test_id,
            splits.test_ood,
        )
    ):
        raise ValueError(f"all splits must carry the {QHASH_COL!r} column for leakage check.")

    pairs = [
        ("train", "validation", splits.train, splits.validation),
        ("train", "test_id", splits.train, splits.test_id),
        ("train", "test_ood", splits.train, splits.test_ood),
        ("validation", "test_id", splits.validation, splits.test_id),
        ("validation", "test_ood", splits.validation, splits.test_ood),
        ("test_id", "test_ood", splits.test_id, splits.test_ood),
    ]
    for name_a, name_b, a, b in pairs:
        overlap = set(a[QHASH_COL]) & set(b[QHASH_COL])
        if overlap:
            raise RuntimeError(
                f"leakage detected between splits {name_a!r} and {name_b!r}: {len(overlap)} shared hashes"
            )
        logger.info("split-leakage[%s vs %s]: 0", name_a, name_b)
