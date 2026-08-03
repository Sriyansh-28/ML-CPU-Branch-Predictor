"""Tests for the 2-bit saturating (bimodal) predictor.

These pin down the exact state-machine semantics (saturation, hysteresis,
indexing) that make the baseline correct, plus a behavioural check that it learns
a strongly-biased branch.
"""

from __future__ import annotations

from branchpred.baselines.two_bit_predictor import TwoBitPredictor


def test_predict_from_initial_state() -> None:
    # Default init is Weakly-Taken (2) -> predicts taken.
    p = TwoBitPredictor(table_bits=4, init_state=2)
    assert p.predict(0) == 1
    p = TwoBitPredictor(table_bits=4, init_state=1)  # Weakly-Not-Taken
    assert p.predict(0) == 0


def test_saturation_upper() -> None:
    p = TwoBitPredictor(table_bits=4, init_state=2)
    for _ in range(10):
        p.update(0, 1)
    # Counter saturates at 3; still predicts taken.
    assert p.predict(0) == 1
    assert p._table[0] == 3


def test_saturation_lower() -> None:
    p = TwoBitPredictor(table_bits=4, init_state=1)
    for _ in range(10):
        p.update(0, 0)
    assert p.predict(0) == 0
    assert p._table[0] == 0


def test_hysteresis_single_anomaly_does_not_flip() -> None:
    # From Strongly-Taken (3), one not-taken drops to Weakly-Taken (2): still taken.
    p = TwoBitPredictor(table_bits=4, init_state=3)
    p.update(0, 0)
    assert p._table[0] == 2
    assert p.predict(0) == 1


def test_two_mispredictions_flip() -> None:
    # From Strongly-Taken, two consecutive not-taken outcomes cross the threshold.
    p = TwoBitPredictor(table_bits=4, init_state=3)
    p.update(0, 0)  # -> 2, still taken
    p.update(0, 0)  # -> 1, now not-taken
    assert p.predict(0) == 0


def test_indexing_uses_low_bits() -> None:
    # PCs that collide in the low table_bits share a counter (aliasing).
    p = TwoBitPredictor(table_bits=4)  # 16 entries, mask 0xF
    assert p._index(0x10) == p._index(0x20) == 0
    assert p._index(0x11) == 1


def test_distinct_pcs_are_independent() -> None:
    p = TwoBitPredictor(table_bits=8, init_state=1)
    for _ in range(5):
        p.update(0x10, 1)  # train one PC toward taken
    assert p.predict(0x10) == 1
    assert p.predict(0x11) == 0  # untouched neighbour keeps its init state


def test_learns_biased_branch() -> None:
    p = TwoBitPredictor(table_bits=8, init_state=1)
    correct = 0
    for _ in range(1000):
        pred = p.predict(0x40)
        outcome = 1  # always taken
        correct += int(pred == outcome)
        p.update(0x40, outcome)
    # After a two-step warm-up it should be right essentially every time.
    assert correct >= 997


def test_reset_restores_initial_state() -> None:
    p = TwoBitPredictor(table_bits=4, init_state=2)
    p.update(0, 0)
    p.update(0, 0)
    p.reset()
    assert p.predict(0) == 1
    assert int(p._table[0]) == 2


def test_invalid_params() -> None:
    for bad in (0, -1):
        try:
            TwoBitPredictor(table_bits=bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")
    try:
        TwoBitPredictor(init_state=4)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for init_state")
