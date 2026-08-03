"""High-level orchestration tying every layer into one comparison run.

Keeps the CLI scripts thin: they parse arguments and call these functions. The
end-to-end flow is:

    trace -> causal features -> chronological split -> train perceptron
          -> evaluate {2-bit, gshare, perceptron} on the identical held-out tail
          -> metrics table + figures.

All predictors warm up over the same train+val history and are scored on the same
test slice, so the numbers are directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .baselines.gshare_predictor import GsharePredictor
from .baselines.two_bit_predictor import TwoBitPredictor
from .data.feature_engineering import Features, build_features, chronological_split
from .evaluation.evaluator import EvalResult, evaluate_online
from .models.torch_dataset import BranchDataset
from .training.config import Config
from .training.trainer import (
    TrainResult,
    build_model,
    build_perceptron_adapter,
    get_trace,
    save_checkpoint,
    train_perceptron,
)
from .utils.logging import get_logger
from .utils.seed import set_seed
from .visualization import plots

logger = get_logger(__name__)


@dataclass
class ComparisonOutput:
    """Everything a comparison run produces."""

    config: Config
    trace: pd.DataFrame
    features: Features
    train_result: TrainResult
    results: dict[str, EvalResult]  # predictor name -> result
    test_start: int


def build_features_for(config: Config, df: pd.DataFrame) -> Features:
    """Build causal features according to ``config.features``."""
    return build_features(
        df,
        global_history_length=config.features.global_history_length,
        local_history_length=config.features.local_history_length,
        pc_hash_bits=config.features.pc_hash_bits,
    )


def train_perceptron_model(config: Config, feats: Features):
    """Train and return ``(model, TrainResult)`` on the config's chronological split."""
    tr, va, _ = chronological_split(len(feats), config.data.train_frac, config.data.val_frac)
    model = build_model(config, feats)
    result = train_perceptron(
        model, BranchDataset(feats, tr), BranchDataset(feats, va), config
    )
    return model, result


def run_comparison(config: Config, save_artifacts: bool = True) -> ComparisonOutput:
    """Run the full train-and-compare pipeline, optionally writing artifacts."""
    set_seed(config.seed)

    df = get_trace(config)
    feats = build_features_for(config, df)
    _, _, te = chronological_split(len(feats), config.data.train_frac, config.data.val_frac)
    test_start = te.start

    logger.info("Training perceptron on the training split...")
    model, train_result = train_perceptron_model(config, feats)
    if save_artifacts:
        save_checkpoint(model, config.output.checkpoint_path, config)

    predictors = {
        "2-bit": TwoBitPredictor(table_bits=config.baselines.two_bit.table_bits),
        "gshare": GsharePredictor(
            table_bits=config.baselines.gshare.table_bits,
            history_bits=config.baselines.gshare.history_bits,
        ),
        "perceptron": build_perceptron_adapter(model, config),
    }

    ipb = config.evaluation.instructions_per_branch
    results = {
        name: evaluate_online(pred, df, score_from=test_start, instructions_per_branch=ipb)
        for name, pred in predictors.items()
    }

    if save_artifacts:
        _write_artifacts(config, results, train_result)

    return ComparisonOutput(
        config=config,
        trace=df,
        features=feats,
        train_result=train_result,
        results=results,
        test_start=test_start,
    )


def format_results_table(results: dict[str, EvalResult]) -> str:
    """Render a Markdown comparison table, best accuracy first."""
    rows = sorted(results.items(), key=lambda kv: kv[1].metrics.accuracy, reverse=True)
    lines = [
        "| Predictor | Accuracy | MPKI | Precision | Recall | F1 | Mispredicts |",
        "|-----------|---------:|-----:|----------:|-------:|---:|------------:|",
    ]
    for name, res in rows:
        m = res.metrics
        lines.append(
            f"| {name} | {m.accuracy:.4f} | {m.mpki:.3f} | {m.precision:.3f} | "
            f"{m.recall:.3f} | {m.f1:.3f} | {m.mispredictions}/{m.total} |"
        )
    return "\n".join(lines)


def _write_artifacts(
    config: Config, results: dict[str, EvalResult], train_result: TrainResult
) -> None:
    """Persist metrics (JSON + Markdown) and all figures."""
    metrics_dir = Path(config.output.metrics_dir)
    figures_dir = Path(config.output.figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = {name: res.metrics.as_dict() for name, res in results.items()}
    (metrics_dir / "comparison.json").write_text(json.dumps(metrics_json, indent=2))
    (metrics_dir / "comparison.md").write_text(format_results_table(results) + "\n")

    plots.plot_accuracy_comparison(results, figures_dir / "accuracy.png")
    plots.plot_mpki_comparison(results, figures_dir / "mpki.png")
    plots.plot_cumulative_mispredictions(results, figures_dir / "cumulative_mispredictions.png")
    for name, res in results.items():
        plots.plot_confusion_matrix(res, figures_dir / f"confusion_{name}.png")
    if train_result.history:
        plots.plot_training_curve(train_result.history, figures_dir / "training_curve.png")

    logger.info("Wrote metrics to %s and figures to %s", metrics_dir, figures_dir)
