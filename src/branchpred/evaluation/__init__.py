"""branchpred.evaluation subpackage — metrics and the online evaluator."""

from .evaluator import EvalResult, evaluate_online, evaluate_predictors
from .metrics import PredictionMetrics, accuracy, misprediction_count, mpki

__all__ = [
    "PredictionMetrics",
    "accuracy",
    "mpki",
    "misprediction_count",
    "EvalResult",
    "evaluate_online",
    "evaluate_predictors",
]
