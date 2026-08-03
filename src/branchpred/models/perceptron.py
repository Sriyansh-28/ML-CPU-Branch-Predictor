"""The perceptron branch predictor (PyTorch).

This is the software analog of the hardware perceptron predictor (Jiménez & Lin,
HPCA 2001). The hardware design keeps a *table of perceptrons* indexed by branch
PC; each perceptron is a vector of integer weights that is dotted with the branch
history (encoded as ±1) to produce a signed score — taken iff the score is
non-negative.

We express exactly that as a compact ``nn.Module``:

* ``w_table`` — an ``Embedding(num_pc_buckets, feature_dim)`` giving the per-PC
  weight vector.
* ``b_table`` — an ``Embedding(num_pc_buckets, 1)`` giving the per-PC bias (the
  classic ``x_0 = 1`` bias input).

The forward pass returns a **logit** ``score = w_pc · history + b_pc`` so it pairs
directly with ``BCEWithLogitsLoss`` for gradient training, while still reducing to
a plain dot-product at inference — just like the hardware. Weights are initialised
to zero (as the hardware perceptron starts), giving a neutral 0.5 initial
probability.
"""

from __future__ import annotations

import torch
from torch import nn


class PerceptronPredictorModel(nn.Module):
    """A table of per-PC perceptrons over the branch-history features.

    Parameters
    ----------
    feature_dim:
        Length of the history feature vector (global + local bits).
    num_pc_buckets:
        Number of PC hash buckets = size of the perceptron table.
    zero_init:
        If ``True`` (default) start all weights and biases at zero, mirroring the
        hardware perceptron and giving deterministic, neutral initial predictions.
    """

    def __init__(self, feature_dim: int, num_pc_buckets: int, zero_init: bool = True) -> None:
        super().__init__()
        if feature_dim <= 0 or num_pc_buckets <= 0:
            raise ValueError("feature_dim and num_pc_buckets must be positive.")
        self.feature_dim = feature_dim
        self.num_pc_buckets = num_pc_buckets

        self.w_table = nn.Embedding(num_pc_buckets, feature_dim)
        self.b_table = nn.Embedding(num_pc_buckets, 1)
        if zero_init:
            nn.init.zeros_(self.w_table.weight)
            nn.init.zeros_(self.b_table.weight)

    def forward(self, history: torch.Tensor, pc_bucket: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch,)`` for a batch of branches.

        Parameters
        ----------
        history:
            Float tensor of shape ``(batch, feature_dim)`` in {-1, 0, +1}.
        pc_bucket:
            Long tensor of shape ``(batch,)`` with values in ``[0, num_pc_buckets)``.
        """
        weights = self.w_table(pc_bucket)          # (batch, feature_dim)
        bias = self.b_table(pc_bucket).squeeze(-1)  # (batch,)
        return (weights * history).sum(dim=-1) + bias

    @torch.no_grad()
    def predict_proba(self, history: torch.Tensor, pc_bucket: torch.Tensor) -> torch.Tensor:
        """Return taken-probabilities of shape ``(batch,)``."""
        return torch.sigmoid(self.forward(history, pc_bucket))
