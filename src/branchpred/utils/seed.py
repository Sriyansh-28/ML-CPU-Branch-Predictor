"""Global reproducibility helpers.

A single :func:`set_seed` call fixes every source of randomness the project can
touch (Python ``random``, NumPy, and — if installed — PyTorch), so that trace
generation, training, and evaluation are all bit-for-bit reproducible from a
config's ``seed`` field.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed all RNGs used by the project.

    Parameters
    ----------
    seed:
        The integer seed to apply everywhere.
    deterministic_torch:
        When ``True`` (default) and PyTorch is available, also request
        deterministic cuDNN behaviour. This trades a little speed for exact
        reproducibility, which is what we want for a benchmark.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is a heavy import; keep seeding usable without it.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch always present in this project
        pass
