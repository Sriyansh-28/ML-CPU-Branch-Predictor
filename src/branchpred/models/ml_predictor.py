"""Adapter exposing the PyTorch perceptron through the ``BasePredictor`` API.

The perceptron model is trained offline on engineered features (see the trainer).
At evaluation time it must run through the *same* online ``predict`` -> ``update``
harness as the classical predictors. This adapter bridges the two worlds by
maintaining, internally, exactly the state ``feature_engineering`` uses — a global
history register and a per-PC local history — so it reconstructs the identical
feature vector for each branch, on the fly, with no future leakage.

Optionally it keeps *learning online*: on each ``update`` it takes a single
manual gradient step (BCE) on the relevant per-PC weights, mimicking a real
adaptive hardware perceptron. The manual update touches only the one PC's weight
row, so it is O(feature_dim) per branch and fast.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch

from ..baselines.base_predictor import BasePredictor
from ..data.feature_engineering import hash_pc
from .perceptron import PerceptronPredictorModel


class PerceptronPredictor(BasePredictor):
    """Online adapter around a (typically pre-trained) perceptron model.

    Parameters
    ----------
    model:
        A :class:`PerceptronPredictorModel`. Its ``feature_dim`` must equal
        ``global_history_length + local_history_length``.
    global_history_length, local_history_length, pc_hash_bits:
        Must match the values used to build the training features.
    online_update:
        If ``True``, take a manual BCE gradient step on each ``update`` so the
        predictor keeps adapting during evaluation.
    learning_rate:
        Step size for the online update.
    """

    name = "perceptron"

    def __init__(
        self,
        model: PerceptronPredictorModel,
        global_history_length: int,
        local_history_length: int,
        pc_hash_bits: int,
        online_update: bool = True,
        learning_rate: float = 0.01,
    ) -> None:
        expected = global_history_length + local_history_length
        if model.feature_dim != expected:
            raise ValueError(
                f"model.feature_dim ({model.feature_dim}) != "
                f"global+local history ({expected})."
            )
        self.model = model.eval()  # inference mode; we update weights manually
        self.gh_len = global_history_length
        self.lh_len = local_history_length
        self.pc_hash_bits = pc_hash_bits
        self.online_update = online_update
        self.lr = learning_rate

        self._global: deque[float] = deque(maxlen=global_history_length)
        self._local: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=local_history_length)
        )

    # -- feature reconstruction ------------------------------------------------
    def _history_vector(self, pc: int) -> np.ndarray:
        """Rebuild the causal ±1 history vector for ``pc`` from internal state.

        Matches ``feature_engineering.build_features``: column j holds the (j+1)-th
        most recent prior outcome, 0 where no such history exists.
        """
        vec = np.zeros(self.gh_len + self.lh_len, dtype=np.float32)
        gh = self._global
        for j in range(len(gh)):
            vec[j] = gh[-(j + 1)]  # most recent first
        if self.lh_len:
            lh = self._local[pc]
            for j in range(len(lh)):
                vec[self.gh_len + j] = lh[-(j + 1)]
        return vec

    def _bucket(self, pc: int) -> int:
        return hash_pc(pc, self.pc_hash_bits)

    # -- BasePredictor API -----------------------------------------------------
    def predict(self, pc: int) -> int:
        hist = torch.from_numpy(self._history_vector(pc)).unsqueeze(0)
        bucket = torch.tensor([self._bucket(pc)], dtype=torch.int64)
        with torch.no_grad():
            logit = self.model(hist, bucket).item()
        return int(logit >= 0.0)

    def update(self, pc: int, outcome: int) -> None:
        if self.online_update:
            self._online_step(pc, outcome)
        # Record the resolved outcome (±1) into global and per-PC history.
        signed = 1.0 if outcome else -1.0
        self._global.append(signed)
        if self.lh_len:
            self._local[pc].append(signed)

    @torch.no_grad()
    def _online_step(self, pc: int, outcome: int) -> None:
        """One manual BCE gradient step on this PC's weight row (pre-update state)."""
        hist = torch.from_numpy(self._history_vector(pc))
        bucket = self._bucket(pc)
        w_row = self.model.w_table.weight[bucket]     # (feature_dim,)
        b_row = self.model.b_table.weight[bucket]     # (1,)
        logit = torch.dot(w_row, hist) + b_row[0]
        prob = torch.sigmoid(logit)
        grad = prob - float(outcome)                  # d(BCE)/d(logit)
        w_row.sub_(self.lr * grad * hist)
        b_row.sub_(self.lr * grad)

    def reset(self) -> None:
        self._global.clear()
        self._local.clear()
