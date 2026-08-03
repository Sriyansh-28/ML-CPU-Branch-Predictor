#!/usr/bin/env python
"""Run the full 2-bit vs gshare vs perceptron comparison and write results.

Trains the perceptron, evaluates all three predictors on the identical held-out
test tail, prints a Markdown results table, and writes metrics + figures under
results/.

Usage:
    python scripts/run_comparison.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make branchpred importable when run without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from branchpred.pipeline import format_results_table, run_comparison  # noqa: E402
from branchpred.training.config import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write metrics/figures to disk."
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    out = run_comparison(config, save_artifacts=not args.no_save)

    print("\n=== Branch Predictor Comparison (held-out test set) ===\n")
    print(format_results_table(out.results))
    print(
        f"\nPerceptron best validation accuracy: {out.train_result.best_val_acc:.4f} "
        f"(epoch {out.train_result.best_epoch})"
    )
    if not args.no_save:
        print(f"\nMetrics + figures written under: {config.output.metrics_dir} / "
              f"{config.output.figures_dir}")


if __name__ == "__main__":
    main()
