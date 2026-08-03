"""branchpred — ML-based CPU branch predictor benchmarked against classical baselines.

The package is organised into independently testable layers:

* ``data``          — trace generation, loading, and causal feature engineering.
* ``baselines``     — classical hardware predictors (2-bit saturating, gshare).
* ``models``        — the PyTorch perceptron predictor and its evaluator adapter.
* ``training``      — configuration handling and the training loop.
* ``evaluation``    — metrics (accuracy, MPKI, ...) and the online evaluator.
* ``visualization`` — matplotlib comparison plots.
* ``utils``         — reproducibility (seeding) and logging helpers.

All predictors — classical and ML — implement the same
:class:`branchpred.baselines.base_predictor.BasePredictor` interface so they can
be driven through one shared, leak-free online evaluation harness.
"""

__version__ = "0.1.0"
