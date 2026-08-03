"""Tests for the online evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from branchpred.baselines.base_predictor import BasePredictor
from branchpred.baselines.gshare_predictor import GsharePredictor
from branchpred.baselines.two_bit_predictor import TwoBitPredictor
from branchpred.evaluation.evaluator import evaluate_online, evaluate_predictors


class AlwaysTaken(BasePredictor):
    name = "always-taken"

    def predict(self, pc: int) -> int:
        return 1

    def update(self, pc: int, outcome: int) -> None:  # no state
        pass


class NoLeakChecker(BasePredictor):
    """Asserts predict() is always called before the matching update()."""

    name = "checker"

    def __init__(self) -> None:
        self.awaiting_update = False

    def predict(self, pc: int) -> int:
        assert not self.awaiting_update, "predict called twice before update (protocol violation)"
        self.awaiting_update = True
        return 1

    def update(self, pc: int, outcome: int) -> None:
        assert self.awaiting_update, "update called before predict (leak risk)"
        self.awaiting_update = False


def _trace(outcomes, pcs=None) -> pd.DataFrame:
    if pcs is None:
        pcs = list(range(len(outcomes)))
    return pd.DataFrame({"pc": pcs, "outcome": outcomes})


def test_always_taken_accuracy_matches_taken_rate() -> None:
    df = _trace([1, 1, 0, 1, 0, 1])  # 4/6 taken
    res = evaluate_online(AlwaysTaken(), df, instructions_per_branch=1.0)
    assert res.metrics.accuracy == pytest.approx(4 / 6)
    assert res.metrics.total == 6


def test_predict_precedes_update_protocol() -> None:
    df = _trace([1, 0, 1, 0, 1, 1, 0])
    # Assertions inside the checker fire if the harness ever violates the order.
    evaluate_online(NoLeakChecker(), df)


def test_score_from_restricts_metrics_but_warms_up() -> None:
    df = _trace([1, 0] * 50)
    res = evaluate_online(AlwaysTaken(), df, score_from=40)
    assert res.metrics.total == len(df) - 40
    assert res.y_true.shape == (len(df) - 40,)


def test_cumulative_mispredictions_monotonic() -> None:
    df = _trace([1, 0, 0, 1, 0])
    res = evaluate_online(AlwaysTaken(), df, instructions_per_branch=1.0)
    cum = res.cumulative_mispredictions
    assert np.all(np.diff(cum) >= 0)
    assert cum[-1] == res.metrics.mispredictions


def test_score_from_out_of_range() -> None:
    df = _trace([1, 0, 1])
    with pytest.raises(ValueError):
        evaluate_online(AlwaysTaken(), df, score_from=3)


def test_evaluate_predictors_returns_all() -> None:
    df = _trace([1, 1, 0, 1, 0, 0, 1, 1], pcs=[0, 4, 0, 4, 0, 4, 0, 4])
    results = evaluate_predictors(
        [TwoBitPredictor(table_bits=8), GsharePredictor(table_bits=8, history_bits=4)],
        df,
        instructions_per_branch=1.0,
    )
    assert set(results) == {"2-bit", "gshare"}
    for res in results.values():
        assert 0.0 <= res.metrics.accuracy <= 1.0
