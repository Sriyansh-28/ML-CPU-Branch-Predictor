#!/usr/bin/env python
"""Evaluate a single predictor on the held-out test tail of the trace.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --predictor gshare

The perceptron requires a checkpoint (run scripts/train.py first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make branchpred importable when run without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from branchpred.baselines.gshare_predictor import GsharePredictor  # noqa: E402
from branchpred.baselines.two_bit_predictor import TwoBitPredictor  # noqa: E402
from branchpred.data.feature_engineering import chronological_split  # noqa: E402
from branchpred.evaluation.evaluator import evaluate_online  # noqa: E402
from branchpred.pipeline import build_features_for  # noqa: E402
from branchpred.training.config import Config  # noqa: E402
from branchpred.training.trainer import (  # noqa: E402
    build_perceptron_adapter,
    get_trace,
    load_checkpoint,
)
from branchpred.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--predictor", choices=["2-bit", "gshare", "perceptron"], default="perceptron"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)

    df = get_trace(config)
    feats = build_features_for(config, df)
    _, _, te = chronological_split(len(feats), config.data.train_frac, config.data.val_frac)

    if args.predictor == "2-bit":
        predictor = TwoBitPredictor(table_bits=config.baselines.two_bit.table_bits)
    elif args.predictor == "gshare":
        predictor = GsharePredictor(
            table_bits=config.baselines.gshare.table_bits,
            history_bits=config.baselines.gshare.history_bits,
        )
    else:
        model = load_checkpoint(config.output.checkpoint_path)
        predictor = build_perceptron_adapter(model, config)

    res = evaluate_online(
        predictor,
        df,
        score_from=te.start,
        instructions_per_branch=config.evaluation.instructions_per_branch,
    )
    m = res.metrics
    print(f"Predictor : {res.name}")
    print(f"Accuracy  : {m.accuracy:.4f}")
    print(f"MPKI      : {m.mpki:.3f}")
    print(f"F1        : {m.f1:.3f}")
    print(f"Mispredict: {m.mispredictions}/{m.total}")


if __name__ == "__main__":
    main()
