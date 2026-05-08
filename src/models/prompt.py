"""Prompt templates and rendering helpers used by training and evaluation.

The user-side instruction and the assistant-side response are formatted. 
Rendering is delegated to the model's chat template via
``tokenizer.apply_chat_template`` so the special turn tokens stay in sync with
whatever Gemma version the tokenizer was trained against.
"""
from __future__ import annotations

from typing import Any, Mapping


USER_TEMPLATE = (
    "You are a medical expert. Answer the following multiple-choice question "
    "by reasoning step by step, then giving your final answer as a single letter.\n\n"
    "Question: {question}\n\n"
    "A) {opa}\n"
    "B) {opb}\n"
    "C) {opc}\n"
    "D) {opd}\n\n"
    "Reason step by step, then provide your answer in the format "
    "\"Answer: <letter>\"."
)

ASSISTANT_TEMPLATE = "Reasoning: {exp}\n\nAnswer: {letter}"

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


def cop_to_letter(cop: int) -> str:
    """Map MedMCQA's integer ``cop`` field (0..3) to the letter A..D."""
    return LETTERS[int(cop)]


# ------------------------------------------------------------------------------------------
# MedMCQA — full chat-formatted training example (user + assistant) and for inference.
# ------------------------------------------------------------------------------------------


def render_medmcqa_user_content(row: Mapping[str, Any]) -> str:
    """Render the user-turn content (instruction + question + options)."""
    return USER_TEMPLATE.format(
        question=row["question"],
        opa=row["opa"], opb=row["opb"], opc=row["opc"], opd=row["opd"],
    )


def render_medmcqa_assistant_content(row: Mapping[str, Any]) -> str:
    """Render the assistant-turn content (reasoning + final letter)."""
    return ASSISTANT_TEMPLATE.format(
        exp=row["exp"],
        letter=cop_to_letter(row["cop"]),
    )


def render_medmcqa_full(row: Mapping[str, Any], tokenizer) -> str:
    """Apply the chat template to produce the complete training example."""
    msgs = [
        {"role": "user",      "content": render_medmcqa_user_content(row)},
        {"role": "assistant", "content": render_medmcqa_assistant_content(row)},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )


def render_medmcqa_user(row: Mapping[str, Any], tokenizer) -> str:
    """Apply the chat template with ``add_generation_prompt=True`` for inference.

    Used for the validation monitoring during training (split ``validation``)
    and for the in-distribution evaluation (split ``test_id``). The assistant
    turn is left unfilled for the model to generate.
    """
    msgs = [{"role": "user", "content": render_medmcqa_user_content(row)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# MedQA — user-only prompt for inference (assistant content is generated).
# ---------------------------------------------------------------------------


def render_medqa_user_content(row: Mapping[str, Any]) -> str:
    """Render the user-turn content for a MedQA row.

    MedQA stores the four options inside a dict under the ``options`` key,
    keyed by ``A``/``B``/``C``/``D``.
    """
    opts = row["options"]
    return USER_TEMPLATE.format(
        question=row["question"],
        opa=opts["A"], opb=opts["B"], opc=opts["C"], opd=opts["D"],
    )


def render_medqa_user(row: Mapping[str, Any], tokenizer) -> str:
    """Apply the chat template with ``add_generation_prompt=True`` for eval."""
    msgs = [{"role": "user", "content": render_medqa_user_content(row)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
