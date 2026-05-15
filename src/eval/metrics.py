"""Compute evaluation metrics from prediction JSONL files.

Inputs
------
A list of prediction records (one per question). Required fields:

- ``id`` (str)
- ``ground_truth_letter`` (str: "A".."D")
- ``predicted_letter`` (str | None — None when parse failed)
- ``parse_success`` (bool)
- ``subject_name`` (str | None — None for MedQA-derived splits)

Outputs (single dict, written to ``artifacts/{model}/metrics.json``)
--------------------------------------------------------------------
Per split:

- ``accuracy`` — fraction of questions where ``predicted_letter ==
  ground_truth_letter``. Parse failures count as WRONG in this denominator;
  hiding them would inflate the headline number.
- ``parse_success_rate`` — fraction of questions where the parser produced a
  letter. Reported as a model-health indicator alongside accuracy: a model
  with low parse rate has its accuracy under-reported, so the two need to be
  read together.
- ``n``, ``n_correct``, ``n_parsed`` — raw counts for sanity-checking.
- ``by_subject`` (only present for splits with non-null ``subject_name``) — per-
  subject accuracy. Each subject bucket has just ``accuracy``, ``n``, and
  ``n_correct``
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

UNKNOWN_SUBJECT = "Unknown"


def _is_correct(record: dict) -> bool:
    return bool(record.get("parse_success")) and (record.get("predicted_letter") == record.get("ground_truth_letter"))


def _split_summary(records: list[dict]) -> dict:
    """Top-level per-split summary: accuracy + parse_success_rate + counts."""
    n = len(records)
    if n == 0:
        return {"accuracy": None, "parse_success_rate": None, "n": 0, "n_correct": 0, "n_parsed": 0}
    n_correct = sum(1 for r in records if _is_correct(r))
    n_parsed = sum(1 for r in records if r.get("parse_success"))
    return {
        "accuracy": n_correct / n,
        "parse_success_rate": n_parsed / n,
        "n": n,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
    }


def _subject_summary(records: list[dict]) -> dict:
    """Per-subject summary: accuracy + counts only (no parse_success_rate)."""
    n = len(records)
    if n == 0:
        return {"accuracy": None, "n": 0, "n_correct": 0}
    n_correct = sum(1 for r in records if _is_correct(r))
    return {"accuracy": n_correct / n, "n": n, "n_correct": n_correct}


def _by_subject(records: list[dict]) -> dict[str, dict]:
    """Per-subject summary. Records without ``subject_name`` go to 'Unknown'."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        subj = r.get("subject_name") or UNKNOWN_SUBJECT
        buckets[subj].append(r)
    # Sort by descending sample size for readability.
    return {subj: _subject_summary(rs) for subj, rs in sorted(buckets.items(), key=lambda kv: -len(kv[1]))}


def compute_metrics(records_by_split: dict[str, list[dict]]) -> dict:
    """Compute the full metrics blob for a single model.

    ``records_by_split`` maps split name (``test_id``, ``test_ood``) to its
    list of records. ``by_subject`` is added only for splits where any record
    has a non-null ``subject_name`` (i.e. MedMCQA-derived splits).
    """
    out: dict = {}
    for split, records in records_by_split.items():
        split_metrics = _split_summary(records)
        if any(r.get("subject_name") for r in records):
            split_metrics["by_subject"] = _by_subject(records)
        out[split] = split_metrics
    return out


def load_predictions(path: Path) -> list[dict]:
    """Read a predictions JSONL file (one record per line)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_metrics(metrics: dict, path: Path) -> None:
    """Write the metrics blob as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
