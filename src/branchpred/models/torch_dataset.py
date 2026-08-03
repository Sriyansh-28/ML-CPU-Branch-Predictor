"""A ``torch.utils.data.Dataset`` over engineered branch features.

Wraps a :class:`branchpred.data.feature_engineering.Features` bundle (or a slice
of one) so it can be fed to a ``DataLoader`` during perceptron training. Each item
is the tuple ``(history_vector, pc_bucket, label)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.feature_engineering import Features


class BranchDataset(Dataset):
    """Dataset yielding ``(history, pc_bucket, label)`` tensors."""

    def __init__(self, features: Features, index: slice | None = None) -> None:
        history = features.history
        pc_bucket = features.pc_bucket
        labels = features.labels
        if index is not None:
            history = history[index]
            pc_bucket = pc_bucket[index]
            labels = labels[index]

        # Contiguous copies keep torch.from_numpy zero-copy and warning-free.
        self.history = torch.from_numpy(np.ascontiguousarray(history, dtype=np.float32))
        self.pc_bucket = torch.from_numpy(np.ascontiguousarray(pc_bucket, dtype=np.int64))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
        self.feature_dim = features.feature_dim
        self.num_pc_buckets = features.num_pc_buckets

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.history[idx], self.pc_bucket[idx], self.labels[idx]
