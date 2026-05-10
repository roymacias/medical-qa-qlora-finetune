"""Plot training and validation loss curves from a Trainer state JSON.

Reads ``trainer_state.json`` (the HF Trainer log history saved at the end of a
training run) and emits a single matplotlib figure with both train and eval
loss as functions of optimizer step.

Train loss is logged every ``logging_steps`` (per-batch); eval loss is logged
every ``eval_steps``. Both series are plotted on the same axes so the gap
between them is directly visible.

``matplotlib`` is treated as a soft dependency: if it is not available, the
function logs a warning and returns without writing the figure (the
``trainer_state.json`` itself is still persisted by the pipeline, so the plot
can be regenerated later by a separate environment).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path


log = logging.getLogger(__name__)


def _load_log_history(state_path: Path) -> list[dict]:
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    return state.get("log_history", [])


def _extract_curves(
    log_history: list[dict],
) -> tuple[list[int], list[float], list[int], list[float]]:
    """Split the log history into (train_step, train_loss, eval_step, eval_loss).

    HF Trainer log entries either contain a ``loss`` (train batch log,
    cadenced by ``logging_steps``) or an ``eval_loss`` (eval block log,
    cadenced by ``eval_steps``); a few summary entries near the end may
    contain neither and are skipped.
    """
    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])
    return train_steps, train_losses, eval_steps, eval_losses


def plot_loss_curves(state_path: Path, output_path: Path) -> None:
    """Read ``trainer_state.json`` and write the loss-curves PNG.

    Parameters
    ----------
    state_path
        Path to ``trainer_state.json`` produced by the training pipeline.
    output_path
        Destination PNG (typically
        ``reports/figures/training/loss_curves.png``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(
            "matplotlib not installed; skipping figure generation. "
            "trainer_state.json was still persisted at %s.",
            state_path,
        )
        return

    log_history = _load_log_history(state_path)
    train_steps, train_losses, eval_steps, eval_losses = _extract_curves(log_history)

    if not train_steps and not eval_steps:
        log.warning("trainer_state.json contains no loss entries; skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_steps:
        ax.plot(
            train_steps, train_losses,
            label=f"train ({len(train_steps)} pts)",
            color="steelblue", alpha=0.7,
        )
    if eval_steps:
        ax.plot(
            eval_steps, eval_losses,
            label=f"eval ({len(eval_steps)} pts)",
            color="darkorange", marker="o", linewidth=2,
        )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("QLoRA fine-tune — training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Loss-curves figure written to %s", output_path)
