"""Causal feature engineering for branch prediction.

Given a canonical ``(pc, outcome)`` trace, build the model inputs used by the
perceptron predictor. Every feature for branch *i* is derived from information
available **strictly before** branch *i* resolves — never from ``outcome[i]`` or
any later branch. This causality is the single most important correctness
property of the whole benchmark and is asserted by the tests.

For each branch we produce:

* **global history** — the last ``global_history_length`` outcomes across *all*
  branches, encoded as ±1 (with 0 padding where history does not yet exist);
* **local history** — the last ``local_history_length`` outcomes of *this PC*,
  likewise ±1 with 0 padding;
* **pc_bucket** — the branch PC hashed into ``2 ** pc_hash_bits`` buckets, used to
  select a per-PC weight vector (the hardware perceptron is a *table* of
  perceptrons indexed by PC).

The history vectors are concatenated into a single ``(N + M)`` input per branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Knuth multiplicative-hash constant (32-bit). Used to spread PCs across buckets
# while keeping the mapping deterministic and shareable with the online adapter.
_KNUTH = 2654435761


def hash_pc(pc: int, bits: int) -> int:
    """Hash a PC into ``[0, 2**bits)`` deterministically (Knuth multiplicative)."""
    if bits <= 0:
        raise ValueError("bits must be positive.")
    return ((int(pc) * _KNUTH) & 0xFFFFFFFF) >> (32 - bits)


@dataclass
class Features:
    """Bundled, model-ready features for a trace.

    Attributes
    ----------
    history:
        ``float32`` array of shape ``(num_branches, feature_dim)`` in {-1, 0, +1}.
    pc_bucket:
        ``int64`` array of shape ``(num_branches,)`` — the hashed PC index.
    labels:
        ``float32`` array of shape ``(num_branches,)`` in {0.0, 1.0}.
    feature_dim:
        ``global_history_length + local_history_length``.
    num_pc_buckets:
        ``2 ** pc_hash_bits`` — size of the perceptron weight table.
    """

    history: np.ndarray
    pc_bucket: np.ndarray
    labels: np.ndarray
    feature_dim: int
    num_pc_buckets: int

    def __len__(self) -> int:
        return len(self.labels)


def _global_history_matrix(signed: np.ndarray, length: int) -> np.ndarray:
    """Columns k=1..length where entry [i, k-1] = signed[i-k] (0 where i < k)."""
    n = len(signed)
    mat = np.zeros((n, length), dtype=np.float32)
    for k in range(1, length + 1):
        if k < n:
            mat[k:, k - 1] = signed[:-k]
    return mat


def _local_history_matrix(
    pcs: np.ndarray, signed: np.ndarray, length: int
) -> np.ndarray:
    """Per-PC history: entry [i, j] = the j-th most recent prior outcome of pcs[i]."""
    n = len(signed)
    mat = np.zeros((n, length), dtype=np.float32)
    if length == 0:
        return mat
    # Group row positions by PC, preserving order, and shift within each group.
    order = np.argsort(pcs, kind="stable")
    sorted_pcs = pcs[order]
    # Boundaries between distinct PCs in the sorted view.
    boundaries = np.flatnonzero(np.diff(sorted_pcs)) + 1
    for group in np.split(order, boundaries):
        grp_signed = signed[group]
        for k in range(1, length + 1):
            if k < len(group):
                mat[group[k:], k - 1] = grp_signed[:-k]
    return mat


def build_features(
    df: pd.DataFrame,
    global_history_length: int = 16,
    local_history_length: int = 8,
    pc_hash_bits: int = 12,
) -> Features:
    """Build causal features from a canonical ``(pc, outcome)`` trace.

    Parameters
    ----------
    df:
        Canonical trace with columns ``[pc, outcome]`` in dynamic order.
    global_history_length, local_history_length:
        Number of global / per-PC history bits to include.
    pc_hash_bits:
        Number of PC hash buckets is ``2 ** pc_hash_bits``.
    """
    if global_history_length < 0 or local_history_length < 0:
        raise ValueError("history lengths must be non-negative.")
    if global_history_length + local_history_length == 0:
        raise ValueError("at least one of the history lengths must be positive.")

    pcs = df["pc"].to_numpy(dtype=np.int64)
    outcomes = df["outcome"].to_numpy(dtype=np.int8)
    signed = (outcomes.astype(np.float32) * 2.0) - 1.0  # {0,1} -> {-1,+1}

    g = _global_history_matrix(signed, global_history_length)
    lcl = _local_history_matrix(pcs, signed, local_history_length)
    history = np.concatenate([g, lcl], axis=1) if lcl.size else g

    mask = (1 << pc_hash_bits) - 1
    pc_bucket = ((pcs * _KNUTH) & 0xFFFFFFFF) >> (32 - pc_hash_bits)
    pc_bucket = (pc_bucket & mask).astype(np.int64)

    return Features(
        history=history,
        pc_bucket=pc_bucket,
        labels=outcomes.astype(np.float32),
        feature_dim=global_history_length + local_history_length,
        num_pc_buckets=1 << pc_hash_bits,
    )


def chronological_split(
    n: int, train_frac: float, val_frac: float
) -> tuple[slice, slice, slice]:
    """Return train/val/test slices for ``n`` items, split by time (no shuffling).

    Splitting chronologically reflects deployment — a predictor learns from a
    program's past to predict its future — and avoids leaking future behaviour
    into training. The test fraction is the remainder.
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1):
        raise ValueError("fractions must satisfy 0<train_frac<1 and 0<=val_frac<1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 to leave a test set.")
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)
