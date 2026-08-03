"""Training pipeline for the perceptron predictor.

Offline training is used only to *warm-start* the per-PC perceptron weights on the
training portion of the trace; the resulting model is then evaluated online (with
optional continued adaptation) through the shared harness. Training is:

* reproducible (seeded via the config),
* config-driven (all hyperparameters come from :class:`Config`),
* early-stopped on validation accuracy, keeping the best checkpoint.

The module also exposes small factories (:func:`build_model`,
:func:`build_perceptron_adapter`) and trace preparation (:func:`get_trace`) so the
CLI scripts stay thin.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.feature_engineering import Features
from ..data.trace_generator import generate_trace
from ..data.trace_loader import load_trace, save_trace
from ..models.ml_predictor import PerceptronPredictor
from ..models.perceptron import PerceptronPredictorModel
from ..models.torch_dataset import BranchDataset
from ..utils.logging import get_logger
from .config import Config

logger = get_logger(__name__)


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_val_acc: float
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)


# -- data ----------------------------------------------------------------------
def get_trace(config: Config) -> pd.DataFrame:
    """Return the trace for this config, generating or loading it as configured.

    For ``source == 'synthetic'`` a deterministic trace is generated and cached to
    ``data.trace_path`` (so downstream steps can reuse it); for ``source == 'file'``
    the trace is loaded from ``data.trace_path``.
    """
    if config.data.source == "file":
        logger.info("Loading trace from %s", config.data.trace_path)
        return load_trace(config.data.trace_path)

    logger.info("Generating synthetic trace (%d branches)", config.data.num_branches)
    df = generate_trace(
        num_branches=config.data.num_branches,
        pattern_mix=config.data.pattern_mix,
        seed=config.seed,
        global_history_length=config.features.global_history_length,
    )
    try:
        save_trace(df, config.data.trace_path)
        logger.info("Cached trace to %s", config.data.trace_path)
    except OSError as exc:  # pragma: no cover - caching is best-effort
        logger.warning("Could not cache trace: %s", exc)
    return df


# -- factories -----------------------------------------------------------------
def build_model(config: Config, features: Features) -> PerceptronPredictorModel:
    """Instantiate a perceptron model sized to the given features."""
    return PerceptronPredictorModel(
        feature_dim=features.feature_dim,
        num_pc_buckets=features.num_pc_buckets,
    )


def build_perceptron_adapter(
    model: PerceptronPredictorModel, config: Config
) -> PerceptronPredictor:
    """Wrap a model in the online BasePredictor adapter using the config's spec."""
    return PerceptronPredictor(
        model,
        global_history_length=config.features.global_history_length,
        local_history_length=config.features.local_history_length,
        pc_hash_bits=config.features.pc_hash_bits,
        online_update=config.model.online_update,
        learning_rate=config.model.learning_rate,
    )


# -- training ------------------------------------------------------------------
@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader (teacher-forced features)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for history, bucket, label in loader:
        logits = model(history, bucket)
        total_loss += criterion(logits, label).item() * label.size(0)
        correct += int(((logits >= 0).float() == label).sum().item())
        n += label.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_perceptron(
    model: PerceptronPredictorModel,
    train_ds: BranchDataset,
    val_ds: BranchDataset,
    config: Config,
) -> TrainResult:
    """Train ``model`` on ``train_ds`` with early stopping on ``val_ds`` accuracy."""
    if len(train_ds) == 0:
        raise ValueError("Training set is empty; cannot train.")
    if len(val_ds) == 0:
        # This trainer selects the best checkpoint and early-stops on validation
        # accuracy. An empty validation set (e.g. val_frac=0, or a tiny trace whose
        # validation slice rounds to zero rows) would make _evaluate return (0, 0),
        # freezing epoch 1 as "best" and discarding all later improvement. Fail
        # fast rather than silently mis-selecting the model.
        raise ValueError(
            "Validation set is empty; increase data.val_frac or the trace size so "
            "the validation slice has at least one branch."
        )
    train_loader = DataLoader(train_ds, batch_size=config.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.model.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.model.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for hist, bucket, label in train_loader:
            optimizer.zero_grad()
            logits = model(hist, bucket)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running += loss.item() * label.size(0)
            seen += label.size(0)

        train_loss = running / max(seen, 1)
        val_loss, val_acc = _evaluate(model, val_loader, criterion)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )
        logger.info(
            "epoch %d | train_loss %.4f | val_loss %.4f | val_acc %.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.model.patience:
                logger.info("Early stopping at epoch %d (patience %d).", epoch, config.model.patience)
                break

    model.load_state_dict(best_state)
    return TrainResult(best_val_acc=best_val_acc, best_epoch=best_epoch, history=history)


# -- checkpointing -------------------------------------------------------------
def save_checkpoint(
    model: PerceptronPredictorModel, path: str | Path, config: Config | None = None
) -> Path:
    """Save a self-describing checkpoint (weights + shape + feature spec).

    When ``config`` is supplied, the *full* feature specification (global/local
    history lengths and PC-hash bits) is stored alongside the shape metadata, so a
    later load can verify that the evaluation config matches how the model was
    trained — not just that the total ``feature_dim`` happens to agree.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": model.feature_dim,
        "num_pc_buckets": model.num_pc_buckets,
    }
    if config is not None:
        payload["feature_spec"] = {
            "global_history_length": config.features.global_history_length,
            "local_history_length": config.features.local_history_length,
            "pc_hash_bits": config.features.pc_hash_bits,
        }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path) -> PerceptronPredictorModel:
    """Rebuild a perceptron model from a checkpoint saved by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PerceptronPredictorModel(
        feature_dim=ckpt["feature_dim"], num_pc_buckets=ckpt["num_pc_buckets"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_adapter_from_checkpoint(path: str | Path, config: Config) -> PerceptronPredictor:
    """Load a checkpoint and build the online adapter, verifying the feature spec.

    Guards against silently evaluating a mismatched model: if the checkpoint stored
    a feature spec (global/local history lengths, PC-hash bits), it must match the
    config exactly — otherwise the adapter would feed columns with different
    meanings or hash PCs into a different bucket range while ``feature_dim`` alone
    still agreed.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    spec = ckpt.get("feature_spec")
    if spec is not None:
        expected = {
            "global_history_length": config.features.global_history_length,
            "local_history_length": config.features.local_history_length,
            "pc_hash_bits": config.features.pc_hash_bits,
        }
        if spec != expected:
            raise ValueError(
                "Checkpoint feature spec does not match the config.\n"
                f"  checkpoint: {spec}\n  config    : {expected}\n"
                "Re-train with this config or evaluate with the training config."
            )
    model = PerceptronPredictorModel(
        feature_dim=ckpt["feature_dim"], num_pc_buckets=ckpt["num_pc_buckets"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return build_perceptron_adapter(model, config)
