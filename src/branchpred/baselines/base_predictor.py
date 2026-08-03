"""The predictor interface shared by every predictor in the project.

Both the classical hardware predictors (2-bit, gshare) and the ML perceptron
implement this one contract, which mirrors how a predictor operates inside a real
pipeline:

    1. ``predict(pc)``          — guess taken/not-taken using only information
                                  available *before* the branch resolves.
    2. ...the branch executes and its true outcome becomes known...
    3. ``update(pc, outcome)``  — update internal state (counters / history /
                                  weights) with the resolved outcome.

Driving both paradigms through the identical ``predict`` → ``update`` protocol is
what makes the ML-vs-classical comparison fair and free of future-information
leakage: no predictor ever sees an outcome before it has predicted it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """Abstract base class for all branch predictors."""

    #: Human-readable name used in results tables and plots.
    name: str = "base"

    @abstractmethod
    def predict(self, pc: int) -> int:
        """Return the predicted outcome for the branch at ``pc`` (1=taken, 0=not)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, pc: int, outcome: int) -> None:
        """Update internal state with the resolved ``outcome`` for ``pc``."""
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027  (intentional optional no-op hook)
        """Reset the predictor to its initial state.

        Stateless predictors need do nothing; stateful ones override this. It is
        deliberately concrete (not abstract) so subclasses are not forced to
        implement it.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r})"
