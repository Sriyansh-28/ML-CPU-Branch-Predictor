"""Consistent, timestamped logging for scripts and library code.

Use :func:`get_logger` everywhere instead of ``print`` so that CLI output,
training progress, and evaluation summaries share one readable format and can be
silenced or redirected centrally.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def _configure_root(level: int) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("branchpred")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a namespaced logger under the ``branchpred`` root.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.
    level:
        Logging level for the shared root handler (applied on first call).
    """
    _configure_root(level)
    # Namespace everything under "branchpred" so the single handler applies.
    child = name if name.startswith("branchpred") else f"branchpred.{name}"
    return logging.getLogger(child)
