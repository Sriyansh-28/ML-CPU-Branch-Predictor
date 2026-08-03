"""Tests for the gshare predictor.

Covers the counter semantics it shares with the bimodal predictor, the global
history register mechanics and index mixing, and — the whole point of gshare — that
it learns a history-correlated branch that a bimodal predictor provably cannot.
"""

from __future__ import annotations

from branchpred.baselines.gshare_predictor import GsharePredictor
from branchpred.baselines.two_bit_predictor import TwoBitPredictor


def test_history_bits_capped_to_table_bits() -> None:
    p = GsharePredictor(table_bits=8, history_bits=20)
    assert p.history_bits == 8


def test_ghr_updates_with_outcomes() -> None:
    p = GsharePredictor(table_bits=8, history_bits=4)
    p.update(0, 1)
    p.update(0, 0)
    p.update(0, 1)
    # GHR low bits should be ...101 = 5.
    assert p._ghr == 0b101


def test_ghr_masked_to_history_bits() -> None:
    p = GsharePredictor(table_bits=8, history_bits=3)
    for _ in range(10):
        p.update(0, 1)
    assert p._ghr == 0b111  # saturated to 3 ones


def test_index_mixes_pc_and_history() -> None:
    p = GsharePredictor(table_bits=8, history_bits=8)
    assert p._index(0x12) == (0x12 ^ 0) & p.mask  # ghr starts at 0
    p.update(0, 1)  # ghr -> 1
    assert p._index(0x12) == (0x12 ^ 1) & p.mask


def test_saturation() -> None:
    p = GsharePredictor(table_bits=6, history_bits=0, init_state=2)
    for _ in range(10):
        p.update(0, 1)
    assert p.predict(0) == 1
    for _ in range(10):
        p.update(0, 0)
    assert p.predict(0) == 0


def test_learns_history_correlated_branch_better_than_bimodal() -> None:
    # A single static branch whose outcome alternates T, N, T, N, ...
    # Bimodal (PC-only) sees one counter thrashing and cannot exceed ~50%.
    # gshare indexes by (PC ^ history), separating the two phases, so it learns it.
    gshare = GsharePredictor(table_bits=10, history_bits=4)
    bimodal = TwoBitPredictor(table_bits=10)
    pc = 0x80

    seq = [1, 0] * 3000  # perfectly history-correlated
    g_correct = b_correct = 0
    warmup = 100
    for i, outcome in enumerate(seq):
        gp = gshare.predict(pc)
        bp = bimodal.predict(pc)
        if i >= warmup:
            g_correct += int(gp == outcome)
            b_correct += int(bp == outcome)
        gshare.update(pc, outcome)
        bimodal.update(pc, outcome)

    n = len(seq) - warmup
    g_acc = g_correct / n
    b_acc = b_correct / n
    assert g_acc > 0.95              # gshare nails the alternation
    assert b_acc < 0.6               # bimodal cannot
    assert g_acc > b_acc + 0.3       # and the gap is large


def test_reset() -> None:
    p = GsharePredictor(table_bits=6, history_bits=4)
    p.update(0, 1)
    p.update(0, 1)
    p.reset()
    assert p._ghr == 0
    assert p.predict(0) == 1  # back to init_state 2 (Weakly-Taken)


def test_invalid_params() -> None:
    try:
        GsharePredictor(table_bits=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
