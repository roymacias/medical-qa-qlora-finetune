"""Build the QLoRA-trainable model: 4-bit base + fresh LoRA adapter.

The base is loaded under NF4 + double quantization with bf16 compute dtype. The adapter is created from a ``LoraConfig`` and attached via
``peft.get_peft_model``. Before attachment, the base is run through
``prepare_model_for_kbit_training`` which:

- enables input gradients on the embedding layer so gradient checkpointing
  can backprop through to the adapters,
- casts layer norms and the LM head to fp32 for numerical stability under
  4-bit weights.

The trainable parameter count is logged on construction.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


log = logging.getLogger(__name__)


def _build_quant_config(base_cfg: dict) -> BitsAndBytesConfig:
    compute_dtype_name = base_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=base_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(base_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype_name),
    )


def _build_lora_config(lora_cfg: dict) -> LoraConfig:
    task_type_name = lora_cfg.get("task_type", "CAUSAL_LM")
    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=getattr(TaskType, task_type_name),
        target_modules=list(lora_cfg["target_modules"]),
    )


def _log_trainable_params(model) -> None:
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    log.info("Trainable params: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", pct)


def build_model_and_tokenizer(
    base_cfg: dict,
    lora_cfg: dict,
) -> tuple[Any, Any]:
    """Load the 4-bit base model and attach a fresh LoRA adapter.

    Parameters
    ----------
    base_cfg
        ``base_model`` block of ``configs/training.yaml``.
    lora_cfg
        ``lora`` block of ``configs/training.yaml``.

    Returns
    -------
    (model, tokenizer)
        The PEFT-wrapped model ready for training and the tokenizer with
        ``pad_token_id`` set if it was missing.
    """
    tokenizer_id = base_cfg.get("tokenizer_id") or base_cfg["model_id"]
    cache_dir = base_cfg.get("cache_dir")
    dtype = getattr(torch, base_cfg.get("torch_dtype", "bfloat16"))

    log.info("Loading tokenizer: %s", tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info("Loading 4-bit base: %s", base_cfg["model_id"])
    quant_cfg = _build_quant_config(base_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        base_cfg["model_id"],
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map="auto",
        cache_dir=cache_dir,
    )

    # Required pre-step before attaching LoRA on a k-bit-quantized base.
    model = prepare_model_for_kbit_training(model)

    log.info(
        "Attaching LoRA: r=%s alpha=%s targets=%s",
        lora_cfg["r"], lora_cfg["lora_alpha"], lora_cfg["target_modules"],
    )
    peft_cfg = _build_lora_config(lora_cfg)
    model = get_peft_model(model, peft_cfg)

    _log_trainable_params(model)

    return model, tokenizer
