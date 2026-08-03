"""Tests for the synthetic branch-trace generator.

These focus on the properties that matter for a *correct* benchmark: exact
reproducibility, a valid canonical schema, and that the modelled behaviours
actually have the statistical character they claim (biased loops, learnable
correlation, near-random noise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from branchpred.data.trace_generator import generate_trace


def test_schema_and_length() -> None:
    df = generate_trace(num_branches=5000, seed=1)
    assert list(df.columns) == ["pc", "outcome"]
    assert len(df) == 5000
    assert df["outcome"].isin([0, 1]).all()
    assert (df["pc"] >= 0).all()


def test_is_reproducible() -> None:
    a = generate_trace(num_branches=3000, seed=123)
    b = generate_trace(num_branches=3000, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs() -> None:
    a = generate_trace(num_branches=3000, seed=1)
    b = generate_trace(num_branches=3000, seed=2)
    assert not a.equals(b)


def test_multiple_distinct_pcs() -> None:
    df = generate_trace(num_branches=5000, seed=7)
    # Several sites per pattern across five patterns -> many distinct PCs.
    assert df["pc"].nunique() >= 10


def test_loop_branches_are_strongly_biased() -> None:
    # A pure-loop mix should be overwhelmingly "taken" (only exits are not-taken).
    df = generate_trace(num_branches=5000, pattern_mix={"loop": 1.0}, seed=3)
    taken_rate = df["outcome"].mean()
    assert taken_rate > 0.75


def test_random_pattern_is_near_balanced() -> None:
    df = generate_trace(num_branches=20000, pattern_mix={"random": 1.0}, seed=5)
    taken_rate = df["outcome"].mean()
    assert 0.45 < taken_rate < 0.55


def test_correlated_pattern_is_learnable_from_pc_and_history() -> None:
    # The correlated outcome is a (mostly) deterministic function of the *PC* and
    # the recent *global* history. A predictor that keys on both (as gshare and the
    # perceptron do) should learn it well; a PC-blind bimodal predictor could not.
    df = generate_trace(num_branches=40000, pattern_mix={"correlated": 1.0}, seed=9)
    pcs = df["pc"].to_numpy()
    outcomes = df["outcome"].to_numpy()

    hist_bits = 6  # matches the correlation width the generator conditions on
    table: dict[tuple[int, int], list[int]] = {}
    correct = 0
    total = 0
    key = 0
    for i, (pc, o) in enumerate(zip(pcs, outcomes)):
        idx = (int(pc), key)
        if i >= hist_bits:
            counts = table.get(idx, [0, 0])
            pred = int(counts[1] >= counts[0])
            correct += int(pred == o)
            total += 1
        counts = table.setdefault(idx, [0, 0])
        counts[int(o)] += 1
        key = ((key << 1) | int(o)) & ((1 << hist_bits) - 1)

    accuracy = correct / total
    # Ceiling is ~0.97 (3% injected noise); comfortably above the 0.5 coin flip.
    assert accuracy > 0.8


def test_pattern_mix_is_normalised() -> None:
    # Un-normalised weights should behave the same as normalised ones.
    a = generate_trace(num_branches=4000, pattern_mix={"loop": 2, "random": 2}, seed=11)
    b = generate_trace(num_branches=4000, pattern_mix={"loop": 0.5, "random": 0.5}, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_invalid_num_branches_raises() -> None:
    try:
        generate_trace(num_branches=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-positive num_branches")


def test_outcomes_are_int_typed() -> None:
    df = generate_trace(num_branches=100, seed=1)
    assert np.issubdtype(df["outcome"].dtype, np.integer)
