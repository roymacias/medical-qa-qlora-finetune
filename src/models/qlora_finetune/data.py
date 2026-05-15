"""Dataset preparation for SFT training of the QLoRA fine-tune.

Loads the experiment splits saved by the data pipeline and formats each
example as a list of chat ``messages`` (one user turn + one assistant turn).
The SFT trainer applies the tokenizer's chat template internally; with
``assistant_only_loss=True`` in the ``SFTConfig`` every token outside the
assistant turn is masked to ``-100`` at collation time, so cross-entropy is
computed only on the assistant response.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from src.models.prompt import (
    render_medmcqa_assistant_content,
    render_medmcqa_user_content,
)

log = logging.getLogger(__name__)

MESSAGES_COLUMN = "messages"


def _add_messages_column(dataset: Dataset) -> Dataset:
    """Map each MedMCQA row to a ``messages`` column with user + assistant turns.

    The resulting structure is what ``SFTTrainer`` expects when
    ``assistant_only_loss=True``: a list of ``{"role", "content"}`` dicts that
    the trainer passes through ``apply_chat_template`` internally.
    """

    def _row_to_messages(row: dict) -> dict:
        return {
            MESSAGES_COLUMN: [
                {"role": "user", "content": render_medmcqa_user_content(row)},
                {"role": "assistant", "content": render_medmcqa_assistant_content(row)},
            ],
        }

    return dataset.map(_row_to_messages, desc="Building chat messages")


def load_train_eval_datasets(
    splits_dir: Path,
    train_split: str,
    eval_split: str,
) -> tuple[Dataset, Dataset]:
    """Load and pre-format the train and eval splits.

    Both splits must share the MedMCQA schema; passing a MedQA-derived split
    here would fail when the renderer accesses MedMCQA-only fields.
    """
    dd = DatasetDict.load_from_disk(str(splits_dir))
    missing = [s for s in (train_split, eval_split) if s not in dd]
    if missing:
        raise KeyError(f"requested splits not in DatasetDict: {missing} (have {sorted(dd)})")

    train_ds = _add_messages_column(dd[train_split])
    eval_ds = _add_messages_column(dd[eval_split])
    log.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))
    return train_ds, eval_ds
