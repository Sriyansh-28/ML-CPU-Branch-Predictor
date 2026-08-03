#!/usr/bin/env python
"""Train the perceptron predictor and save a checkpoint.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make branchpred importable when run without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from branchpred.pipeline import build_features_for, train_perceptron_model  # noqa: E402
from branchpred.training.config import Config  # noqa: E402
from branchpred.training.trainer import get_trace, save_checkpoint  # noqa: E402
from branchpred.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)

    df = get_trace(config)
    feats = build_features_for(config, df)
    model, result = train_perceptron_model(config, feats)
    path = save_checkpoint(model, config.output.checkpoint_path, config)

    print(f"Best validation accuracy : {result.best_val_acc:.4f} (epoch {result.best_epoch})")
    print(f"Checkpoint saved to      : {path}")


if __name__ == "__main__":
    main()
