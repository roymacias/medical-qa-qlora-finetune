"""Build qualitative analysis markdown for the cross-model 5-row pattern.

Reads predictions for the three models on a given split, finds 5 question IDs
that satisfy the cross-model correctness pattern (anchored on the fine-tune),
and emits a markdown file per model with the selected questions and that
model's full ``generated_text``.

Pattern (5 rows, totals: base 1✓/4✗, fine-tune 3✓/2✗, medgemma 4✓/1✗)
---------------------------------------------------------------------

| # | base  | fine-tune | medgemma | story
|---|-------|-----------|----------|-----------------------------------------------
| 1 | wrong | right     | right    | fine-tuning closes the gap from the base model
| 2 | wrong | right     | right    | fine-tuning closes the gap from the base model
| 3 | wrong | wrong     | right    | residual gap compared to the industrial model
| 4 | wrong | wrong     | right    | residual gap compared to the industrial model
| 5 | right | right     | wrong    | even the industrial model fails sometimes

Selection algorithm
-------------------
1. Build per-id correctness across the 3 models on the intersection of IDs.
2. For each row, sample one matching ID without replacement (deterministic
   under ``seed`` via sorted candidates + ``random.Random.choice``).
3. If a row has zero candidates, log a warning and emit a placeholder cell;
   the markdown still renders, just with one empty section.

This step is run separately from ``evaluation_pipeline.py`` because it needs
predictions from all 3 models.

Usage:

    python -m src.eval.qualitative
    python -m src.eval.qualitative --splits test_ood
    python -m src.eval.qualitative --config configs/evaluation.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "qualitative"


@dataclass(frozen=True)
class PatternRow:
    base: bool       # True = correct
    finetune: bool
    medgemma: bool
    story: str


# The 5 rows of the cross-model pattern (anchored on the fine-tune model).
PATTERN: tuple[PatternRow, ...] = (
    PatternRow(False, True,  True,  "fine-tuning closes the gap that the base model cannot"),
    PatternRow(False, True,  True,  "fine-tuning closes the gap that the base model cannot"),
    PatternRow(False, False, True,  "residual gap compared to the industrial ceiling"),
    PatternRow(False, False, True,  "residual gap compared to the industrial ceiling"),
    PatternRow(True,  True,  False, "even the industrial model fails sometimes"),
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_correct(record: dict) -> bool:
    return bool(record.get("parse_success")) and (
        record.get("predicted_letter") == record.get("ground_truth_letter")
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_ids(
    base_records: list[dict],
    finetune_records: list[dict],
    medgemma_records: list[dict],
    seed: int = 42,
) -> list[tuple[PatternRow, str | None]]:
    """Pick one ID per pattern row. Returns list aligned to ``PATTERN``.

    Each entry is ``(row, id)`` where ``id is None`` if no candidate satisfied
    the row's constraints (logged as a warning).
    """
    correct_base = {r["id"]: _is_correct(r) for r in base_records}
    correct_ft   = {r["id"]: _is_correct(r) for r in finetune_records}
    correct_med  = {r["id"]: _is_correct(r) for r in medgemma_records}

    common_ids = set(correct_base) & set(correct_ft) & set(correct_med)
    if not common_ids:
        raise RuntimeError("no overlapping IDs across the 3 prediction files")
    log.info("Selecting from %d common IDs", len(common_ids))

    rng = random.Random(seed)
    selected: list[tuple[PatternRow, str | None]] = []
    used: set[str] = set()
    for row in PATTERN:
        candidates = sorted(
            i for i in common_ids
            if i not in used
            and correct_base[i] == row.base
            and correct_ft[i]   == row.finetune
            and correct_med[i]  == row.medgemma
        )
        if not candidates:
            log.warning(
                "no candidate for row (base=%s, ft=%s, med=%s); leaving empty",
                row.base, row.finetune, row.medgemma,
            )
            selected.append((row, None))
            continue
        chosen = rng.choice(candidates)
        used.add(chosen)
        selected.append((row, chosen))
    return selected


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _bool_glyph(b: bool) -> str:
    return "✓" if b else "✗"


def _records_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records}


def render_markdown(
    model_name: str,
    split: str,
    selected: list[tuple[PatternRow, str | None]],
    records: list[dict],
) -> str:
    """Markdown for ONE model — same 5 IDs across the 3 model files."""
    by_id = _records_by_id(records)
    out: list[str] = []
    out.append(f"# Análisis cualitativo — {model_name} / {split}")
    out.append("")
    out.append(
        "Selection anchored to the fine-tune with a cross-pattern of 5 examples. "
        "The same 5 IDs are used across all 3 models to enable side-by-side comparison."
    )
    out.append("")
    out.append("| # | base | fine-tune | medgemma | historia | id |")
    out.append("|---|------|-----------|----------|----------|----|")
    for i, (row, sid) in enumerate(selected, 1):
        out.append(
            f"| {i} | {_bool_glyph(row.base)} | {_bool_glyph(row.finetune)} | "
            f"{_bool_glyph(row.medgemma)} | {row.story} | "
            f"{f'`{sid}`' if sid else '_N/A_'} |"
        )
    out.append("")
    out.append("---")
    out.append("")

    for i, (row, sid) in enumerate(selected, 1):
        out.append(f"## Example {i}")
        out.append(
            f"_Pattern: base={_bool_glyph(row.base)}, "
            f"fine-tune={_bool_glyph(row.finetune)}, "
            f"medgemma={_bool_glyph(row.medgemma)} — {row.story}_"
        )
        out.append("")
        if sid is None or sid not in by_id:
            out.append("_(no candidate matching the pattern in this split)_")
            out.append("")
            continue
        rec = by_id[sid]
        out.append(f"- **id**: `{sid}`")
        out.append(f"- **subject**: {rec.get('subject_name') or 'N/A'}")
        out.append(f"- **gold**: `{rec.get('ground_truth_letter')}`")
        out.append(
            f"- **predicted**: `{rec.get('predicted_letter')}` "
            f"(parse_success={rec.get('parse_success')})"
        )
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


def build_qualitative_for_split(
    split: str,
    base_model: str,
    finetune_model: str,
    medgemma_model: str,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    seed: int = 42,
) -> None:
    """Read 3 prediction files, select 5 IDs, write 3 md files for this split."""
    paths = {
        base_model:     artifacts_dir / base_model     / "predictions" / f"{split}.jsonl",
        finetune_model: artifacts_dir / finetune_model / "predictions" / f"{split}.jsonl",
        medgemma_model: artifacts_dir / medgemma_model / "predictions" / f"{split}.jsonl",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing predictions for {name}: {p}")

    base_records     = _load_jsonl(paths[base_model])
    finetune_records = _load_jsonl(paths[finetune_model])
    medgemma_records = _load_jsonl(paths[medgemma_model])

    selected = select_ids(base_records, finetune_records, medgemma_records, seed=seed)

    for model_name, records in (
        (base_model, base_records),
        (finetune_model, finetune_records),
        (medgemma_model, medgemma_records),
    ):
        md = render_markdown(model_name, split, selected, records)
        out_path = reports_dir / model_name / f"{split}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        log.info("Wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build cross-model qualitative markdown.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument(
        "--splits", nargs="*",
        help="Override splits from config (e.g. test_id test_ood).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qcfg = cfg["qualitative"]
    splits = args.splits or qcfg.get("splits", ["test_id", "test_ood"])
    seed = int(cfg.get("seed", 42))

    for split in splits:
        log.info("=== %s ===", split)
        build_qualitative_for_split(
            split=split,
            base_model=qcfg["base_model"],
            finetune_model=qcfg["finetune_model"],
            medgemma_model=qcfg["medgemma_model"],
            seed=seed,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
