"""The classical 2-bit saturating (bimodal) branch predictor.

This is the canonical hardware baseline. A Pattern History Table (PHT) of 2-bit
saturating counters is indexed by the low bits of the branch PC:

    counter states:  0 = Strongly-Not-Taken   1 = Weakly-Not-Taken
                     2 = Weakly-Taken          3 = Strongly-Taken

    predict:  taken iff counter >= 2 (i.e. the high bit is set)
    update :  taken -> increment (saturating at 3)
              not   -> decrement (saturating at 0)

The two bits give *hysteresis*: a single anomalous outcome nudges the counter but
does not immediately flip the prediction, which is why the bimodal predictor is
robust on strongly-biased branches (e.g. loops) yet — because it ignores global
history entirely — cannot capture correlation between branches. That gap is
exactly what gshare and the perceptron exploit.
"""

from __future__ import annotations

import numpy as np

from .base_predictor import BasePredictor

# Counter is taken when its value is >= this threshold (high bit set).
_TAKEN_THRESHOLD = 2
_MAX_COUNTER = 3
_MIN_COUNTER = 0


class TwoBitPredictor(BasePredictor):
    """A 2-bit saturating counter (bimodal) predictor.

    Parameters
    ----------
    table_bits:
        The PHT holds ``2 ** table_bits`` counters; the PC is masked to this many
        low bits to index it. Larger tables reduce destructive aliasing.
    init_state:
        Initial counter value for every entry. Defaults to 2 (Weakly-Taken), a
        common choice given that taken branches (loops) dominate typical code.
    """

    name = "2-bit"

    def __init__(self, table_bits: int = 14, init_state: int = 2) -> None:
        if table_bits <= 0:
            raise ValueError("table_bits must be positive.")
        if not _MIN_COUNTER <= init_state <= _MAX_COUNTER:
            raise ValueError("init_state must be in [0, 3].")
        self.table_bits = table_bits
        self.size = 1 << table_bits
        self.mask = self.size - 1
        self.init_state = init_state
        self._table = np.full(self.size, init_state, dtype=np.int8)

    def _index(self, pc: int) -> int:
        return pc & self.mask

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

    def reset(self) -> None:
        self._table.fill(self.init_state)
