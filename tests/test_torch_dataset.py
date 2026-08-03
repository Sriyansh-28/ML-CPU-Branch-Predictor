"""Tests for the PyTorch dataset wrapper over engineered features."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from branchpred.data.feature_engineering import build_features
from branchpred.models.torch_dataset import BranchDataset


def _features(n: int = 50):
    df = pd.DataFrame(
        {"pc": [i % 5 for i in range(n)], "outcome": [i % 2 for i in range(n)]}
    )
    return build_features(df, global_history_length=6, local_history_length=2, pc_hash_bits=6)


def test_length_and_item_types() -> None:
    ds = BranchDataset(_features(40))
    assert len(ds) == 40
    hist, pc, label = ds[0]
    assert hist.shape == (8,)
    assert hist.dtype == torch.float32
    assert pc.dtype == torch.int64
    assert label.dtype == torch.float32


def test_slice_restricts_dataset() -> None:
    feats = _features(100)
    ds = BranchDataset(feats, index=slice(0, 30))
    assert len(ds) == 30


def test_dataloader_batches() -> None:
    ds = BranchDataset(_features(64))
    loader = DataLoader(ds, batch_size=16)
    batches = list(loader)
    assert len(batches) == 4
    hist, pc, label = batches[0]
    assert hist.shape == (16, 8)
    assert pc.shape == (16,)
    assert label.shape == (16,)
