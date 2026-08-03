"""Tests for the metrics functions and the PredictionMetrics bundle."""

from __future__ import annotations

import numpy as np
import pytest

from branchpred.evaluation.metrics import (
    PredictionMetrics,
    accuracy,
    misprediction_count,
    mpki,
)


def test_accuracy_basic() -> None:
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75


def test_misprediction_count() -> None:
    assert misprediction_count([1, 1, 0, 0], [1, 0, 0, 1]) == 2


def test_mpki_per_1000_branches() -> None:
    # 2 mispredictions in 4 branches, 1 instruction/branch -> 500 per 1000.
    assert mpki([1, 1, 0, 0], [0, 0, 0, 0], instructions_per_branch=1.0) == 500.0
    # With 5 instructions/branch the rate is 5x lower.
    assert mpki([1, 1, 0, 0], [0, 0, 0, 0], instructions_per_branch=5.0) == 100.0


def test_perfect_prediction_zero_mpki() -> None:
    y = [1, 0, 1, 1, 0]
    assert mpki(y, y) == 0.0
    assert accuracy(y, y) == 1.0


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        mpki([], [])


def test_prediction_metrics_bundle() -> None:
    # y_true: 1 1 0 0 ; y_pred: 1 0 0 0
    m = PredictionMetrics.from_arrays([1, 1, 0, 0], [1, 0, 0, 0], instructions_per_branch=1.0)
    assert m.accuracy == 0.75
    assert m.mispredictions == 1
    assert m.total == 4
    assert m.mpki == 250.0
    # Confusion: tp=1 (row1->1), fn=1 (row1->0), tn=2, fp=0
    assert (m.tp, m.fn, m.tn, m.fp) == (1, 1, 2, 0)
    assert 0.0 <= m.precision <= 1.0
    d = m.as_dict()
    assert set(d) >= {"accuracy", "mpki", "precision", "recall", "f1", "tp", "tn"}


def test_confusion_matrix_handles_single_class() -> None:
    # All not-taken, predicted all not-taken -> tn only, no crash.
    m = PredictionMetrics.from_arrays([0, 0, 0], [0, 0, 0])
    assert m.tn == 3
    assert m.tp == m.fp == m.fn == 0
    assert m.accuracy == 1.0


def test_numpy_inputs() -> None:
    yt = np.array([1, 0, 1, 1])
    yp = np.array([1, 1, 1, 0])
    assert accuracy(yt, yp) == 0.5
