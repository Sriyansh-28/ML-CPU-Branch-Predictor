"""Tests for causal feature engineering.

The headline test is the *leakage* check: a feature row for branch i must not
change when any outcome at position >= i changes. The rest verify exact global /
local history values, PC hashing, and the chronological split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from branchpred.data.feature_engineering import (
    build_features,
    chronological_split,
    hash_pc,
)


def _trace(pcs, outcomes) -> pd.DataFrame:
    return pd.DataFrame({"pc": pcs, "outcome": outcomes})


def test_global_history_exact_values() -> None:
    # Single PC so global == local content; check the ±1 encoding and padding.
    df = _trace([0, 0, 0, 0], [1, 0, 1, 1])
    feats = build_features(df, global_history_length=2, local_history_length=0)
    # Row 0: no history -> [0, 0]
    # Row 1: prev outcome 1 -> [+1, 0]
    # Row 2: prev 0, prev-prev 1 -> [-1, +1]
    # Row 3: prev 1, prev-prev 0 -> [+1, -1]
    expected = np.array(
        [[0, 0], [1, 0], [-1, 1], [1, -1]], dtype=np.float32
    )
    np.testing.assert_array_equal(feats.history, expected)


def test_local_history_is_per_pc() -> None:
    # Interleave two PCs; local history must track each PC independently.
    df = _trace([10, 20, 10, 20, 10], [1, 0, 0, 1, 1])
    feats = build_features(df, global_history_length=0, local_history_length=1)
    # PC 10 outcomes in order: 1,0,1 -> local prev = [0, +1, -1]
    # PC 20 outcomes in order: 0,1   -> local prev = [0, -1]
    # Rows:      10 20 10 20 10
    expected = np.array([[0], [0], [1], [-1], [-1]], dtype=np.float32)
    np.testing.assert_array_equal(feats.history, expected)


def test_no_future_leakage() -> None:
    # Flipping outcome at position i must not alter any feature row <= i.
    rng = np.random.default_rng(0)
    pcs = rng.integers(0, 5, size=200)
    outcomes = rng.integers(0, 2, size=200)
    base = build_features(
        _trace(pcs, outcomes), global_history_length=8, local_history_length=4
    )

    for i in (10, 50, 199):
        flipped = outcomes.copy()
        flipped[i] = 1 - flipped[i]
        mod = build_features(
            _trace(pcs, flipped), global_history_length=8, local_history_length=4
        )
        # Rows 0..i inclusive depend only on outcomes < their own index, so
        # changing outcome[i] can only affect rows > i.
        np.testing.assert_array_equal(base.history[: i + 1], mod.history[: i + 1])


def test_feature_dim_and_shapes() -> None:
    df = _trace([1, 2, 3, 1, 2], [1, 0, 1, 0, 1])
    feats = build_features(df, global_history_length=4, local_history_length=3, pc_hash_bits=6)
    assert feats.feature_dim == 7
    assert feats.history.shape == (5, 7)
    assert feats.pc_bucket.shape == (5,)
    assert feats.labels.shape == (5,)
    assert feats.num_pc_buckets == 64
    assert feats.history.dtype == np.float32


def test_labels_match_outcomes() -> None:
    df = _trace([1, 1, 1], [1, 0, 1])
    feats = build_features(df, global_history_length=2, local_history_length=0)
    np.testing.assert_array_equal(feats.labels, np.array([1, 0, 1], dtype=np.float32))


def test_pc_hash_in_range_and_deterministic() -> None:
    for bits in (4, 8, 12):
        vals = [hash_pc(pc, bits) for pc in (0x400000, 0x400004, 0x123456)]
        assert all(0 <= v < (1 << bits) for v in vals)
    # Deterministic and consistent with build_features' bucketing.
    df = _trace([0x400000, 0x400004], [1, 0])
    feats = build_features(df, global_history_length=2, local_history_length=0, pc_hash_bits=10)
    assert feats.pc_bucket[0] == hash_pc(0x400000, 10)
    assert feats.pc_bucket[1] == hash_pc(0x400004, 10)


def test_chronological_split() -> None:
    tr, va, te = chronological_split(1000, 0.7, 0.1)
    assert (tr.start, tr.stop) == (0, 700)
    assert (va.start, va.stop) == (700, 800)
    assert (te.start, te.stop) == (800, 1000)


def test_split_validation() -> None:
    with pytest.raises(ValueError):
        chronological_split(100, 0.9, 0.2)  # leaves no test set
    with pytest.raises(ValueError):
        chronological_split(100, 0.0, 0.1)


def test_requires_some_history() -> None:
    with pytest.raises(ValueError):
        build_features(_trace([1], [1]), global_history_length=0, local_history_length=0)
