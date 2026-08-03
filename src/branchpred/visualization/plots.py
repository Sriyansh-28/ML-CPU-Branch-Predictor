"""Matplotlib figures for the predictor comparison.

All plots use the non-interactive ``Agg`` backend so they render headlessly (in
CI, over SSH, etc.) and are written straight to disk. Each function returns the
path it wrote, and takes the evaluator's :class:`EvalResult` objects (or a training
history) as input, keeping plotting decoupled from computation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from ..evaluation.evaluator import EvalResult  # noqa: E402

# A consistent, colour-blind-friendly ordering of bar colours.
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_accuracy_comparison(results: dict[str, EvalResult], path: str | Path) -> Path:
    """Bar chart of prediction accuracy per predictor."""
    path = _ensure_parent(path)
    names = list(results)
    accs = [results[n].metrics.accuracy for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, accs, color=_COLORS[: len(names)])
    ax.set_ylabel("Accuracy")
    ax.set_title("Branch Prediction Accuracy")
    ax.set_ylim(0, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_mpki_comparison(results: dict[str, EvalResult], path: str | Path) -> Path:
    """Bar chart of MPKI per predictor (lower is better)."""
    path = _ensure_parent(path)
    names = list(results)
    vals = [results[n].metrics.mpki for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals, color=_COLORS[: len(names)])
    ax.set_ylabel("MPKI (mispredictions / 1000 instructions)")
    ax.set_title("MPKI — lower is better")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cumulative_mispredictions(results: dict[str, EvalResult], path: str | Path) -> Path:
    """Line plot of cumulative mispredictions over the scored trace."""
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, res) in enumerate(results.items()):
        cum = res.cumulative_mispredictions
        ax.plot(range(len(cum)), cum, label=name, color=_COLORS[i % len(_COLORS)])
    ax.set_xlabel("Branch index (scored region)")
    ax.set_ylabel("Cumulative mispredictions")
    ax.set_title("Cumulative Mispredictions (lower slope is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(result: EvalResult, path: str | Path) -> Path:
    """Heatmap of a single predictor's 2x2 confusion matrix."""
    path = _ensure_parent(path)
    m = result.metrics
    cm = [[m.tn, m.fp], [m.fn, m.tp]]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred NT", "pred T"])
    ax.set_yticks([0, 1], labels=["true NT", "true T"])
    ax.set_title(f"Confusion Matrix — {result.name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > (m.total / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_training_curve(history: list[dict[str, float]], path: str | Path) -> Path:
    """Plot training/validation loss and validation accuracy over epochs."""
    path = _ensure_parent(path)
    epochs = [h["epoch"] for h in history]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", color=_COLORS[0], label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], "s-", color=_COLORS[1], label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_acc"] for h in history], "^-", color=_COLORS[2], label="val acc")
    ax2.set_ylabel("Validation accuracy")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Perceptron Training Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
