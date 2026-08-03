"""The gshare branch predictor (McFarling, 1993).

gshare strengthens the bimodal predictor by folding *global* branch history into
the index, so the same static branch can map to different counters depending on
the path that led to it — capturing correlation the bimodal predictor cannot.

    index   = (pc ^ global_history) & mask
    predict = counter[index] >= 2                 # same 2-bit saturating counter
    update  : saturating counter update, then
              global_history = ((global_history << 1) | outcome) & history_mask

A single global history register (GHR) is shared across all branches. The XOR mix
of PC and history spreads correlated branches across the table while still letting
purely-biased branches converge, making gshare a strong yet classical baseline to
put between the bimodal predictor and the learned perceptron.
"""

from __future__ import annotations

import numpy as np

from .base_predictor import BasePredictor

_TAKEN_THRESHOLD = 2
_MAX_COUNTER = 3
_MIN_COUNTER = 0


class GsharePredictor(BasePredictor):
    """A gshare (global-history XOR PC) predictor.

    Parameters
    ----------
    table_bits:
        The counter table holds ``2 ** table_bits`` entries.
    history_bits:
        Width of the global history register folded into the index. Capped to
        ``table_bits`` because more history than index bits cannot be used.
    init_state:
        Initial counter value (0..3); defaults to 2 (Weakly-Taken).
    """

    name = "gshare"

    def __init__(self, table_bits: int = 14, history_bits: int = 14, init_state: int = 2) -> None:
        if table_bits <= 0:
            raise ValueError("table_bits must be positive.")
        if history_bits < 0:
            raise ValueError("history_bits must be non-negative.")
        if not _MIN_COUNTER <= init_state <= _MAX_COUNTER:
            raise ValueError("init_state must be in [0, 3].")

        self.table_bits = table_bits
        self.history_bits = min(history_bits, table_bits)
        self.size = 1 << table_bits
        self.mask = self.size - 1
        self.history_mask = (1 << self.history_bits) - 1
        self.init_state = init_state

        self._table = np.full(self.size, init_state, dtype=np.int8)
        self._ghr = 0

    def _index(self, pc: int) -> int:
        return (pc ^ self._ghr) & self.mask

    def predict(self, pc: int) -> int:
        return int(self._table[self._index(pc)] >= _TAKEN_THRESHOLD)

    def update(self, pc: int, outcome: int) -> None:
        idx = self._index(pc)
        if outcome:
            if self._table[idx] < _MAX_COUNTER:
                self._table[idx] += 1
        else:
            if self._table[idx] > _MIN_COUNTER:
                self._table[idx] -= 1
        # Shift the resolved outcome into the global history register.
        self._ghr = ((self._ghr << 1) | (outcome & 1)) & self.history_mask

    def reset(self) -> None:
        self._table.fill(self.init_state)
        self._ghr = 0
