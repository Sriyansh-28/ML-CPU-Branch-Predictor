"""Generate realistic, reproducible synthetic branch traces.

Real branch-trace suites (CBP/ChampSim) are large, access-gated, and stored in a
simulator-specific binary format (see ``docs/datasets.md``). To keep this project
reproducible out of the box, we synthesise a *program-like* branch stream instead.

The generator models a small synthetic program as a collection of **branch sites**
(each with its own PC and behaviour), and produces a dynamic instruction stream by
interleaving branches from those sites. The behaviours are chosen to exercise the
strengths and weaknesses of the different predictors:

* ``loop``         — a loop back-edge: taken ``T-1`` times then one exit. Strongly
  biased; the easy case that even a bimodal predictor nails.
* ``nested_loop``  — a longer periodic pattern (inner/outer trip counts). Still
  perfectly predictable, but only for predictors that use history.
* ``correlated``   — outcome is a deterministic function of the *global* branch
  history. The bimodal predictor (which ignores global history) cannot capture
  this, while gshare and the perceptron can — this is where ML earns its keep.
* ``function``     — a biased branch (call/return-like context). Handled well by
  biased counters.
* ``random``       — a fair coin: the irreducible noise floor for every predictor.

Everything is driven by a single seeded ``numpy`` generator, so a given
``(num_branches, pattern_mix, seed)`` always yields the identical trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Base address for synthetic branch sites (word-aligned, realistic-looking PCs).
_PC_BASE = 0x0040_0000
_PC_STRIDE = 0x4

# How many distinct branch sites to instantiate per pattern category.
_SITES_PER_PATTERN = 4

# Width of the global history the ``correlated`` sites condition on.
_CORRELATION_BITS = 6


@dataclass
class _Site:
    """A single branch site with a PC and a stateful outcome model."""

    pc: int
    pattern: str
    rng: np.random.Generator
    # Per-pattern state (only the relevant fields are used).
    trip_count: int = 0
    counter: int = 0
    period_pattern: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    bias: float = 0.5
    table: dict = field(default_factory=dict)
    noise: float = 0.0

    def next(self, global_history: deque[int]) -> int:
        """Return this site's next outcome, advancing any internal state."""
        if self.pattern == "loop":
            self.counter += 1
            if self.counter >= self.trip_count:
                self.counter = 0
                return 0  # loop exit (fall-through / not-taken)
            return 1  # back-edge taken

        if self.pattern == "nested_loop":
            bit = int(self.period_pattern[self.counter % len(self.period_pattern)])
            self.counter += 1
            return bit

        if self.pattern == "correlated":
            key = 0
            for i, h in enumerate(list(global_history)[-_CORRELATION_BITS:]):
                key |= (h & 1) << i
            base = self.table.setdefault(key, int(self.rng.integers(0, 2)))
            if self.noise and self.rng.random() < self.noise:
                return 1 - base  # occasional flip -> realistic imperfect learnability
            return base

        if self.pattern == "function":
            return int(self.rng.random() < self.bias)

        # "random"
        return int(self.rng.random() < 0.5)


def _build_sites(pattern_mix: dict[str, float], rng: np.random.Generator) -> tuple[list[_Site], np.ndarray]:
    """Instantiate branch sites and their per-step selection weights.

    Returns the sites plus a normalised weight vector, where each site's weight is
    ``pattern_mix[pattern] / sites_in_pattern`` so that the *aggregate* fraction of
    dynamic branches drawn from each pattern matches the requested mix.
    """
    sites: list[_Site] = []
    weights: list[float] = []
    pc = _PC_BASE

    total = sum(pattern_mix.values())
    if total <= 0:
        raise ValueError("pattern_mix weights must sum to a positive value.")

    for pattern, weight in pattern_mix.items():
        share = weight / total
        for _ in range(_SITES_PER_PATTERN):
            site = _Site(pc=pc, pattern=pattern, rng=rng)
            pc += _PC_STRIDE

            if pattern == "loop":
                site.trip_count = int(rng.integers(4, 65))
            elif pattern == "nested_loop":
                inner = int(rng.integers(3, 9))
                outer = int(rng.integers(3, 9))
                pat = np.ones(inner * outer, dtype=np.int8)
                pat[inner - 1 :: inner] = 0  # inner-loop exits
                pat[-1] = 0  # outer-loop exit
                site.period_pattern = pat
            elif pattern == "function":
                site.bias = float(rng.uniform(0.8, 0.97))
            elif pattern == "correlated":
                site.noise = 0.03  # small, keeps it realistically imperfect

            sites.append(site)
            weights.append(share / _SITES_PER_PATTERN)

    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    return sites, w


def generate_trace(
    num_branches: int,
    pattern_mix: dict[str, float] | None = None,
    seed: int = 42,
    global_history_length: int = 32,
) -> pd.DataFrame:
    """Generate a synthetic branch trace.

    Parameters
    ----------
    num_branches:
        Number of dynamic branch records to emit.
    pattern_mix:
        Mapping of pattern name -> relative weight. Weights are auto-normalised
        and control the fraction of dynamic branches drawn from each category.
        Defaults to a balanced, loop-heavy mix.
    seed:
        Seed for the internal RNG; a fixed seed yields an identical trace.
    global_history_length:
        Length of the global history maintained while generating (must be at least
        the width the correlated sites condition on).

    Returns
    -------
    pandas.DataFrame
        Canonical ``[pc, outcome]`` trace in dynamic order.
    """
    if num_branches <= 0:
        raise ValueError("num_branches must be positive.")
    if pattern_mix is None:
        pattern_mix = {
            "loop": 0.35,
            "nested_loop": 0.20,
            "correlated": 0.25,
            "function": 0.10,
            "random": 0.10,
        }

    rng = np.random.default_rng(seed)
    sites, weights = _build_sites(pattern_mix, rng)
    site_indices = np.arange(len(sites))

    global_history: deque[int] = deque(maxlen=max(global_history_length, _CORRELATION_BITS))

    # Pre-draw the site selection sequence in one vectorised call for speed.
    chosen = rng.choice(site_indices, size=num_branches, p=weights)

    pcs = np.empty(num_branches, dtype=np.int64)
    outcomes = np.empty(num_branches, dtype=np.int8)

    for step in range(num_branches):
        site = sites[chosen[step]]
        outcome = site.next(global_history)
        pcs[step] = site.pc
        outcomes[step] = outcome
        global_history.append(outcome)

    return pd.DataFrame({"pc": pcs, "outcome": outcomes})
