"""branchpred.visualization subpackage — matplotlib comparison plots."""

from .plots import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_cumulative_mispredictions,
    plot_mpki_comparison,
    plot_training_curve,
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_mpki_comparison",
    "plot_cumulative_mispredictions",
    "plot_confusion_matrix",
    "plot_training_curve",
]
