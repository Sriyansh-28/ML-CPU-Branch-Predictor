"""The online evaluation harness — the heart of the fair comparison.

Every predictor (classical or ML) is driven through the *same* single pass over
the trace: for each branch we call ``predict`` (using only prior state), record the
guess, then call ``update`` with the resolved outcome. Because the protocol is
identical for all predictors and no outcome is ever revealed before it is
predicted, the resulting accuracy/MPKI numbers are directly comparable and free of
future-information leakage.

To mirror the ML model's train/test split, a predictor may stream over the whole
trace (warming up its state) while metrics are only recorded from ``score_from``
onward — so all predictors are scored on the identical held-out tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..baselines.base_predictor import BasePredictor
from ..utils.logging import get_logger
from .metrics import DEFAULT_INSTRUCTIONS_PER_BRANCH, PredictionMetrics

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Per-predictor evaluation output."""

    name: str
    metrics: PredictionMetrics
    y_true: np.ndarray
    y_pred: np.ndarray
    cumulative_mispredictions: np.ndarray  # length == number of scored branches


def evaluate_online(
    predictor: BasePredictor,
    df: pd.DataFrame,
    score_from: int = 0,
    instructions_per_branch: float = DEFAULT_INSTRUCTIONS_PER_BRANCH,
) -> EvalResult:
    """Run ``predictor`` over the trace, scoring branches at index >= ``score_from``.

    Parameters
    ----------
    predictor:
        Any :class:`BasePredictor` (2-bit, gshare, or the perceptron adapter).
    df:
        Canonical ``(pc, outcome)`` trace in dynamic order.
    score_from:
        Metrics are recorded only from this index onward; earlier branches still
        drive ``update`` so the predictor is warmed up identically for everyone.
    instructions_per_branch:
        Passed through to the MPKI calculation.
    """
    pcs = df["pc"].to_numpy(dtype=np.int64)
    outcomes = df["outcome"].to_numpy(dtype=np.int64)
    n = len(df)
    if not (0 <= score_from < n):
        raise ValueError(f"score_from must be in [0, {n}); got {score_from}.")

    scored = n - score_from
    y_true = np.empty(scored, dtype=np.int64)
    y_pred = np.empty(scored, dtype=np.int64)

    j = 0
    for i in range(n):
        pc = int(pcs[i])
        outcome = int(outcomes[i])
        pred = predictor.predict(pc)
        if i >= score_from:
            y_true[j] = outcome
            y_pred[j] = pred
            j += 1
        predictor.update(pc, outcome)

    cumulative = np.cumsum(y_true != y_pred)
    metrics = PredictionMetrics.from_arrays(y_true, y_pred, instructions_per_branch)
    logger.info(
        "%-11s | acc %.4f | MPKI %.3f | %d/%d mispredicted",
        predictor.name, metrics.accuracy, metrics.mpki, metrics.mispredictions, metrics.total,
    )
    return EvalResult(
        name=predictor.name,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        cumulative_mispredictions=cumulative,
    )


def evaluate_predictors(
    predictors: dict[str, BasePredictor] | list[BasePredictor],
    df: pd.DataFrame,
    score_from: int = 0,
    instructions_per_branch: float = DEFAULT_INSTRUCTIONS_PER_BRANCH,
) -> dict[str, EvalResult]:
    """Evaluate several predictors on the same trace, returning results by name."""
    if isinstance(predictors, dict):
        items = list(predictors.items())
    else:
        items = [(p.name, p) for p in predictors]

    results: dict[str, EvalResult] = {}
    for name, predictor in items:
        results[name] = evaluate_online(predictor, df, score_from, instructions_per_branch)
    return results
