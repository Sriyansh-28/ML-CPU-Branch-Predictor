"""Tests for the perceptron model and its online BasePredictor adapter.

The most important test proves the adapter reconstructs, online, *exactly* the
feature vectors that ``build_features`` produces offline — otherwise the trained
model would see different inputs at evaluation time. The rest cover forward
shapes, zero-init behaviour, and that the online adapter actually learns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from branchpred.data.feature_engineering import build_features
from branchpred.models.ml_predictor import PerceptronPredictor
from branchpred.models.perceptron import PerceptronPredictorModel


def test_forward_shape_and_zero_init() -> None:
    model = PerceptronPredictorModel(feature_dim=8, num_pc_buckets=16)
    hist = torch.randn(5, 8)
    bucket = torch.randint(0, 16, (5,))
    logits = model(hist, bucket)
    assert logits.shape == (5,)
    # Zero init -> all logits exactly zero -> 0.5 probability.
    assert torch.allclose(logits, torch.zeros(5))
    assert torch.allclose(model.predict_proba(hist, bucket), torch.full((5,), 0.5))


def test_adapter_rejects_dim_mismatch() -> None:
    model = PerceptronPredictorModel(feature_dim=10, num_pc_buckets=16)
    try:
        PerceptronPredictor(model, global_history_length=4, local_history_length=4, pc_hash_bits=4)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on feature_dim mismatch")


def test_adapter_reconstructs_offline_features() -> None:
    # Step the adapter through a trace and confirm the history vector it builds at
    # each branch matches the corresponding row from build_features exactly.
    rng = np.random.default_rng(1)
    pcs = rng.integers(0, 6, size=120)
    outcomes = rng.integers(0, 2, size=120)
    df = pd.DataFrame({"pc": pcs, "outcome": outcomes})

    gh, lh, bits = 8, 4, 6
    feats = build_features(df, global_history_length=gh, local_history_length=lh, pc_hash_bits=bits)

    model = PerceptronPredictorModel(feature_dim=gh + lh, num_pc_buckets=1 << bits)
    adapter = PerceptronPredictor(
        model, gh, lh, bits, online_update=False
    )

    for i in range(len(df)):
        pc = int(pcs[i])
        built = adapter._history_vector(pc)
        np.testing.assert_array_equal(built, feats.history[i])
        # Bucket must match too.
        assert adapter._bucket(pc) == feats.pc_bucket[i]
        adapter.update(pc, int(outcomes[i]))


def test_predict_threshold_at_zero_logit() -> None:
    model = PerceptronPredictorModel(feature_dim=4, num_pc_buckets=8)
    adapter = PerceptronPredictor(model, 4, 0, 3, online_update=False)
    # Zero weights -> logit 0 -> predict taken (>= 0).
    assert adapter.predict(0x10) == 1


def test_online_adapter_learns_biased_branch() -> None:
    # A single always-taken PC: online updates should push predictions to taken.
    model = PerceptronPredictorModel(feature_dim=8, num_pc_buckets=16)
    adapter = PerceptronPredictor(model, 4, 4, 4, online_update=True, learning_rate=0.1)
    pc = 0x20
    correct = 0
    n = 500
    for _ in range(n):
        pred = adapter.predict(pc)
        correct += int(pred == 1)
        adapter.update(pc, 1)
    assert correct >= n - 5  # essentially always right after warm-up


def test_online_adapter_learns_history_correlation() -> None:
    # Alternating T,N,... for one PC — learnable from global history, not from a
    # PC-only counter. The perceptron should exceed chance comfortably.
    model = PerceptronPredictorModel(feature_dim=8, num_pc_buckets=16)
    adapter = PerceptronPredictor(model, 8, 0, 4, online_update=True, learning_rate=0.1)
    pc = 0x30
    seq = [1, 0] * 1500
    correct = 0
    warmup = 200
    for i, o in enumerate(seq):
        pred = adapter.predict(pc)
        if i >= warmup:
            correct += int(pred == o)
        adapter.update(pc, o)
    acc = correct / (len(seq) - warmup)
    assert acc > 0.9


def test_reset_clears_history() -> None:
    model = PerceptronPredictorModel(feature_dim=4, num_pc_buckets=8)
    adapter = PerceptronPredictor(model, 2, 2, 3, online_update=False)
    adapter.update(0x10, 1)
    adapter.update(0x10, 0)
    adapter.reset()
    assert np.all(adapter._history_vector(0x10) == 0)
