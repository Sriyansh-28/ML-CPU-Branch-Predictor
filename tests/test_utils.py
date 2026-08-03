"""Smoke tests for the foundational utilities and package wiring.

These keep CI meaningful from the very first milestone: they prove the package
imports, versioning is exposed, and seeding is genuinely reproducible.
"""

from __future__ import annotations

import numpy as np

import branchpred
from branchpred.utils.logging import get_logger
from branchpred.utils.seed import set_seed


def test_package_exposes_version() -> None:
    assert isinstance(branchpred.__version__, str)
    assert branchpred.__version__.count(".") >= 1


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = np.random.rand(5)
    set_seed(123)
    second = np.random.rand(5)
    assert np.allclose(first, second)


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b)


def test_get_logger_is_namespaced() -> None:
    logger = get_logger(__name__)
    assert logger.name.startswith("branchpred")
