"""Run inference for a single model on a single split.

Loads the model per a model config, iterates the dataset split, generates a
response, parses the predicted letter, and streams one
prediction record per line into a JSONL file.

Streaming is line-at-a-time + ``flush()`` so a crash mid-run leaves a valid
partial file (the orchestrator can resume by skipping already-completed
splits).

Model loading
-------------
Two model "types" are supported via the model config (``configs/models/*.yaml``):

- ``hf_pretrained`` — load weights directly via
  ``AutoModelForCausalLM.from_pretrained``. Used for ``gemma-3-4b-it``
  (the base) and ``medgemma-4b-it`` (the industrial reference).

- ``peft_adapter`` — load the base model and a LoRA adapter on top. Used for
  ``gemma-3-4b-it-qlora-finetune``. The base is loaded under the SAME 4-bit
  quantization as during QLoRA training so eval matches train.

Output schema (one record per question)
---------------------------------------
::

    {
        "id": str,
        "model": str,                  # model name from CLI / model config
        "split": str,                  # "test_id" or "test_ood"
        "ground_truth_letter": "A".."D",
        "predicted_letter": "A".."D" | None,
        "parse_success": bool,
        "generated_text": str,         # raw text, no special tokens
        "subject_name": str | None,    # null for MedQA (no subject in source)
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.eval.parsing import parse_letter
from src.models.prompt import (
    cop_to_letter,
    render_medmcqa_user,
    render_medqa_user,
)

log = logging.getLogger(__name__)


# Splits that share MedMCQA's schema (``cop``, ``subject_name``, ``id``).
MEDMCQA_SPLITS = {"train", "validation", "test_id"}
# Splits that share MedQA's schema (``answer_idx``, ``options``; no native id).
MEDQA_SPLITS = {"test_ood"}

LOG_EVERY = 40  # examples


@dataclass
class GenerationConfig:
    """Generation kwargs forwarded to ``model.generate``.

    Defaults match the project decisions:
    - greedy decoding (``do_sample=False``);
    - ``max_new_tokens=512``;
    - ``temperature=1.0`` and ``repetition_penalty=1.0`` are inert under greedy
      but kept explicit to avoid version-specific crashes on edge values.
    """

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    repetition_penalty: float = 1.0


def _gold_letter(row: dict, split: str) -> str:
    if split in MEDMCQA_SPLITS:
        return cop_to_letter(int(row["cop"]))
    if split in MEDQA_SPLITS:
        return str(row["answer_idx"]).upper()
    raise ValueError(f"unknown split: {split!r}")


def _render_prompt(row: dict, split: str, tokenizer) -> str:
    if split in MEDMCQA_SPLITS:
        return render_medmcqa_user(row, tokenizer)
    if split in MEDQA_SPLITS:
        return render_medqa_user(row, tokenizer)
    raise ValueError(f"unknown split: {split!r}")


def _example_id(row: dict, split: str, idx: int) -> str:
    """MedMCQA carries an ``id`` column; MedQA does not, so one is synthesize
    from split + position. Position is stable because splits are saved on disk
    with deterministic order."""
    raw_id = row.get("id") if isinstance(row, dict) else None
    if raw_id:
        return str(raw_id)
    return f"{split}-{idx:05d}"


def _subject_name(row: dict, split: str) -> str | None:
    if split in MEDMCQA_SPLITS:
        return row.get("subject_name")
    return None


def _build_quant_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_and_tokenizer(model_cfg: dict) -> tuple[Any, Any]:
    """Load the model and tokenizer per the model config.

    See module docstring for the supported config types.
    """
    model_type = model_cfg["type"]
    tokenizer_id = model_cfg.get("tokenizer_id") or model_cfg.get("model_id") or model_cfg.get("base_model_id")
    if tokenizer_id is None:
        raise ValueError("model config must specify tokenizer_id, model_id, or base_model_id")

    cache_dir = model_cfg.get("cache_dir")
    dtype_name = model_cfg.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    quant_cfg = _build_quant_config(load_in_4bit)

    log.info("Loading tokenizer from %s", tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_type == "hf_pretrained":
        model_id = model_cfg["model_id"]
        log.info("Loading hf_pretrained model %s (4bit=%s)", model_id, load_in_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
            device_map="auto",
            cache_dir=cache_dir,
            attn_implementation="sdpa",
        )
    elif model_type == "peft_adapter":
        from peft import PeftModel  # imported lazily so non-finetune runs don't need peft

        base_id = model_cfg["base_model_id"]
        adapter_path = model_cfg["adapter_path"]
        log.info("Loading base %s with adapter %s (4bit=%s)", base_id, adapter_path, load_in_4bit)
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
            device_map="auto",
            cache_dir=cache_dir,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        raise ValueError(f"unsupported model type: {model_type!r}")

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def run_inference(
    model_name: str,
    model,
    tokenizer,
    dataset: Dataset,
    split: str,
    gen_cfg: GenerationConfig,
    output_path: Path,
    batch_size: int = 8,
) -> None:
    """Iterate the dataset, generate, parse, and stream predictions to JSONL.

    Streaming + per-line ``flush()`` so a crash mid-run leaves a valid partial
    file. The orchestrator skips already-complete (model, split) combinations
    so re-running is cheap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.padding_side = "left"

    gen_kwargs = {
        "max_new_tokens": gen_cfg.max_new_tokens,
        "do_sample": gen_cfg.do_sample,
        "temperature": gen_cfg.temperature,
        "repetition_penalty": gen_cfg.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    n = len(dataset)
    log.info("Inference: model=%s split=%s n=%d (batch_size=%d)", model_name, split, n, batch_size)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, n, batch_size):
            batch_rows = [dataset[idx] for idx in range(i, min(i + batch_size, n))]

            prompts = [_render_prompt(row, split, tokenizer) for row in batch_rows]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            output_ids = model.generate(**inputs, **gen_kwargs)

            for j, row in enumerate(batch_rows):
                input_len = inputs["input_ids"][j].shape[0]
                new_tokens = output_ids[j, input_len:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

                predicted = parse_letter(generated)
                gold = _gold_letter(row, split)

                original_idx = i + j

                record = {
                    "id": _example_id(row, split, original_idx),
                    "model": model_name,
                    "split": split,
                    "ground_truth_letter": gold,
                    "predicted_letter": predicted,
                    "parse_success": predicted is not None,
                    "generated_text": generated,
                    "subject_name": _subject_name(row, split),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()

            # Progress logging
            current_processed = min(i + batch_size, n)

            if current_processed % LOG_EVERY == 0 or current_processed == n:
                log.info("  %s/%s: %d/%d", model_name, split, current_processed, n)
