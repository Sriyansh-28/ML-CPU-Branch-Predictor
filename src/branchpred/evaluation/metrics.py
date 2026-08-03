"""Branch-prediction metrics.

Two numbers headline the comparison:

* **Accuracy** — fraction of branches predicted correctly.
* **MPKI** — *mispredictions per 1000 instructions*, the standard metric computer
  architects use, because a misprediction costs a pipeline flush and its impact is
  naturally expressed per instruction. Our traces contain *branches only*, so MPKI
  is computed as ``mispredictions / total_instructions * 1000`` where
  ``total_instructions = num_branches * instructions_per_branch``. The
  ``instructions_per_branch`` factor (default 5.0, a common average for general-
  purpose code) makes the number comparable to published MPKI figures; set it to
  1.0 to read the metric as mispredictions-per-1000-branches instead.

Precision/recall/F1 and the confusion matrix expose *how* a predictor errs (e.g. a
bias toward predicting "taken").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DEFAULT_INSTRUCTIONS_PER_BRANCH = 5.0


def _as_int_array(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64).ravel()


def accuracy(y_true, y_pred) -> float:
    yt, yp = _as_int_array(y_true), _as_int_array(y_pred)
    if yt.size == 0:
        raise ValueError("Cannot compute accuracy on empty arrays.")
    return float((yt == yp).mean())


def misprediction_count(y_true, y_pred) -> int:
    yt, yp = _as_int_array(y_true), _as_int_array(y_pred)
    return int((yt != yp).sum())


def mpki(
    y_true,
    y_pred,
    instructions_per_branch: float = DEFAULT_INSTRUCTIONS_PER_BRANCH,
) -> float:
    """Mispredictions per 1000 instructions (see module docstring)."""
    if instructions_per_branch <= 0:
        raise ValueError("instructions_per_branch must be positive.")
    yt = _as_int_array(y_true)
    if yt.size == 0:
        raise ValueError("Cannot compute MPKI on empty arrays.")
    mispreds = misprediction_count(y_true, y_pred)
    total_instructions = yt.size * instructions_per_branch
    return mispreds / total_instructions * 1000.0


@dataclass
class PredictionMetrics:
    """A bundle of the metrics for one predictor on one trace."""

    accuracy: float
    mpki: float
    mispredictions: int
    total: int
    precision: float
    recall: float
    f1: float
    # Confusion matrix counts.
    tn: int
    fp: int
    fn: int
    tp: int

    @classmethod
    def from_arrays(
        cls,
        y_true,
        y_pred,
        instructions_per_branch: float = DEFAULT_INSTRUCTIONS_PER_BRANCH,
    ) -> PredictionMetrics:
        yt, yp = _as_int_array(y_true), _as_int_array(y_pred)
        # labels=[0,1] guarantees a 2x2 matrix even if one class is absent.
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in cm.ravel())
        precision, recall, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=[0, 1], average="binary", zero_division=0
        )
        return cls(
            accuracy=accuracy(yt, yp),
            mpki=mpki(yt, yp, instructions_per_branch),
            mispredictions=misprediction_count(yt, yp),
            total=int(yt.size),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "mpki": self.mpki,
            "mispredictions": self.mispredictions,
            "total": self.total,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }
