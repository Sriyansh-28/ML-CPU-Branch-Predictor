"""Tests for the plotting functions.

These are smoke tests: they confirm each plot renders headlessly and writes a
non-empty image file. They deliberately avoid asserting on pixels.
"""

from __future__ import annotations

import pandas as pd

from branchpred.baselines.two_bit_predictor import TwoBitPredictor
from branchpred.evaluation.evaluator import evaluate_online
from branchpred.visualization.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_cumulative_mispredictions,
    plot_mpki_comparison,
    plot_training_curve,
)


def _results():
    df = pd.DataFrame({"pc": [0, 4, 0, 4, 0, 4, 0, 4], "outcome": [1, 0, 1, 0, 1, 1, 0, 0]})
    r1 = evaluate_online(TwoBitPredictor(table_bits=6), df, instructions_per_branch=1.0)
    r2 = evaluate_online(TwoBitPredictor(table_bits=8), df, instructions_per_branch=1.0)
    return {"a": r1, "b": r2}


def _assert_written(path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_accuracy_plot(tmp_path) -> None:
    _assert_written(plot_accuracy_comparison(_results(), tmp_path / "acc.png"))


def test_mpki_plot(tmp_path) -> None:
    _assert_written(plot_mpki_comparison(_results(), tmp_path / "mpki.png"))


def test_cumulative_plot(tmp_path) -> None:
    _assert_written(plot_cumulative_mispredictions(_results(), tmp_path / "cum.png"))


def test_confusion_plot(tmp_path) -> None:
    res = _results()["a"]
    _assert_written(plot_confusion_matrix(res, tmp_path / "cm.png"))


def test_training_curve_plot(tmp_path) -> None:
    history = [
        {"epoch": 1, "train_loss": 0.6, "val_loss": 0.62, "val_acc": 0.70},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.45, "val_acc": 0.80},
        {"epoch": 3, "train_loss": 0.3, "val_loss": 0.42, "val_acc": 0.83},
    ]
    _assert_written(plot_training_curve(history, tmp_path / "curve.png"))


def test_plot_creates_parent_dirs(tmp_path) -> None:
    _assert_written(plot_accuracy_comparison(_results(), tmp_path / "deep" / "nested" / "acc.png"))
