"""Render qualitative analysis markdown for the configured cases.

Each entry of ``CASE_IDS`` pins a single (split, id) tuple. The position of
the entry in the list defines its case number (1-indexed). The module looks
up each id in the prediction JSONL of every model declared in the
evaluation config and writes one markdown file per (model, split) under
``reports/qualitative/{model}/{split}.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "qualitative"


# ---------------------------------------------------------------------------
# USER-EDITABLE: list of (split, id) tuples. The index defines the case
# number. Each id must exist in the corresponding split's predictions JSONL
# for every model declared in the evaluation config.
# ---------------------------------------------------------------------------
CASE_IDS: list[tuple[str, str]] = [
    ("test_id", "de09d388-bd4e-42a9-ac6b-ee2d95f822e2"),
    ("test_id", "d2398cd6-b205-4fb3-a4c4-9e575662b0bf"),
    ("test_ood", "test_ood-00207"),
]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _predictions_path(artifacts_dir: Path, model: str, split: str) -> Path:
    return artifacts_dir / model / "predictions" / f"{split}.jsonl"


def _records_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _summary_cell(rec: dict | None) -> str:
    """Compact cell for the summary table: predicted letter and a glyph
    indicating whether it matched the gold letter. ``_n/a_`` when the
    record could not be located."""
    if rec is None:
        return "_n/a_"
    predicted = rec.get("predicted_letter") or "—"
    correct = bool(rec.get("parse_success")) and rec.get("predicted_letter") == rec.get("ground_truth_letter")
    return f"`{predicted}` {'✓' if correct else '✗'}"


def render_markdown(
    model_name: str,
    split: str,
    cases_for_split: list[tuple[int, str]],
    model_records: dict[str, dict[str, dict]],
    all_models: list[str],
) -> str:
    """Markdown for ONE (model, split). The summary table covers all three
    models for the cases in this split; each per-case section shows the
    generation of the model addressed by ``model_name``."""
    out: list[str] = []
    out.append(f"# Qualitative analysis — {model_name} / {split}")
    out.append("")
    out.append("Case identifiers are configured in `src/eval/qualitative.py` (`CASE_IDS`).")
    out.append("")

    header_cells = ["Case", "id", *[f"`{m}`" for m in all_models]]
    out.append("| " + " | ".join(header_cells) + " |")
    out.append("|" + "---|" * len(header_cells))
    for case_num, sid in cases_for_split:
        cells = [str(case_num), f"`{sid}`"]
        for m in all_models:
            cells.append(_summary_cell(model_records.get(m, {}).get(sid)))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("---")
    out.append("")

    own_records = model_records.get(model_name, {})
    for case_num, sid in cases_for_split:
        out.append(f"## Case {case_num}")
        out.append("")
        rec = own_records.get(sid)
        if rec is None:
            out.append(
                f"_(id `{sid}` not found in `{model_name}/{split}.jsonl`. Verify it is set correctly in `CASE_IDS`.)_"
            )
            out.append("")
            continue
        out.append(f"- **id**: `{sid}`")
        out.append(f"- **gold**: `{rec.get('ground_truth_letter')}`")
        out.append(f"- **predicted**: `{rec.get('predicted_letter')}` (parse_success={rec.get('parse_success')})")
        out.append("")
        out.append("**Model generation:**")
        out.append("")
        out.append("```text")
        out.append((rec.get("generated_text") or "").rstrip())
        out.append("```")
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _warn_on_placeholders() -> None:
    for split, sid in CASE_IDS:
        if sid == "REPLACE_ME":
            log.warning(
                "an entry for split=%s still holds the placeholder id; edit CASE_IDS in src/eval/qualitative.py",
                split,
            )


def build_qualitative(
    base_model: str,
    finetune_model: str,
    medgemma_model: str,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> None:
    _warn_on_placeholders()
    all_models = [base_model, finetune_model, medgemma_model]

    # Group cases by split, preserving the absolute case number (1-indexed
    # over the full CASE_IDS list, not reset per split).
    cases_by_split: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for case_num, (split, sid) in enumerate(CASE_IDS, start=1):
        cases_by_split[split].append((case_num, sid))

    # Load every (model, split) prediction file once, indexed by id.
    records_cache: dict[tuple[str, str], dict[str, dict]] = {}
    for split in cases_by_split:
        for model in all_models:
            records_cache[(model, split)] = _records_by_id(_load_jsonl(_predictions_path(artifacts_dir, model, split)))

    for split, cases_for_split in cases_by_split.items():
        model_records = {m: records_cache[(m, split)] for m in all_models}
        for model_name in all_models:
            md = render_markdown(
                model_name=model_name,
                split=split,
                cases_for_split=cases_for_split,
                model_records=model_records,
                all_models=all_models,
            )
            out_path = reports_dir / model_name / f"{split}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md, encoding="utf-8")
            log.info("Wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render qualitative markdown for the configured cases.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qcfg = cfg["qualitative"]

    build_qualitative(
        base_model=qcfg["base_model"],
        finetune_model=qcfg["finetune_model"],
        medgemma_model=qcfg["medgemma_model"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
