"""branchpred.training subpackage — configuration and the training pipeline."""

from .config import Config
from .trainer import (
    TrainResult,
    build_model,
    build_perceptron_adapter,
    get_trace,
    load_checkpoint,
    save_checkpoint,
    train_perceptron,
)

__all__ = [
    "Config",
    "TrainResult",
    "build_model",
    "build_perceptron_adapter",
    "get_trace",
    "load_checkpoint",
    "save_checkpoint",
    "train_perceptron",
]
